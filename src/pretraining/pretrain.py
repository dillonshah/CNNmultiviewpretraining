import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms.v2 as T
import torchvision.transforms.functional as TF
from torchsampler import ImbalancedDatasetSampler
from torchmetrics.functional import auroc
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from skimage.util import img_as_ubyte
from skimage.io import imread

import pandas as pd
from tqdm import tqdm
import numpy as np

import cv2
import random
import os
import numbers

# Transfrom taken from biomedia's mammo-net.py
class GammaCorrectionTransform:
    def __init__(self, gamma=0.5):
        self.gamma = self._check_input(gamma, 'gammacorrection')   
        
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        # if value is 0 or (1., 1.) for gamma correction do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, img):
        gamma_factor = None if self.gamma is None else float(torch.empty(1).uniform_(self.gamma[0], self.gamma[1]))
        if gamma_factor is not None:
            img = TF.adjust_gamma(img, gamma_factor, gain=1)
        return img

class EMBEDData(pl.LightningDataModule):
    def __init__(self, val_percent, test_percent, batch_size, num_workers, probing=False):
        super().__init__()
        self.test_percent = test_percent
        self.val_percent = val_percent
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.df = pd.read_csv("data/EMBED/tables/merged_df.csv",
         low_memory=False)

        # FFDM only
        self.df = self.df[self.df['FinalImageType'] == '2D']

        # Female only
        self.df = self.df[self.df['GENDER_DESC'] == 'Female']

        # Remove unclear breast density cases
        self.df = self.df[self.df['tissueden'].notna()]
        self.df = self.df[self.df['tissueden'] < 5]

        # MLO and CC only        
        self.df = self.df[self.df['ViewPosition'].isin(['MLO','CC'])]

        # Remove spot compression or magnificiation
        self.df = self.df[self.df['spot_mag'].isna()]

        self.df.dropna(inplace=True, subset='asses')
        self.df.drop(self.df[self.df['asses'] == 'X'].index, inplace=True)

        self.df['label'] = self.df['asses']
        self.df.loc[self.df['asses'] == 'N', 'label'] = 0
        self.df.loc[self.df['asses'] == 'B', 'label'] = 0
        self.df.loc[self.df['asses'] == 'P', 'label'] = 0
        self.df.loc[self.df['asses'] == 'A', 'label'] = 0
        self.df.loc[self.df['asses'] == 'S', 'label'] = 0
        self.df.loc[self.df['asses'] == 'M', 'label'] = 0
        self.df.loc[self.df['asses'] == 'K', 'label'] = 1

        # Restructure data for multi-view task
        self.df = self.restructure_data(self.df)

        self.df['mlo_path'] = self.df.MLO_path.values
        self.df['cc_path'] = self.df.CC_path.values
        self.df['study_id'] = [str(study_id) for study_id in self.df.study_id.values]
        
        # Making sure images from the same subject are within the same set
        self.df['split'] = 'test'

        unique_study_ids_all = self.df.study_id.unique()
        unique_study_ids_all = shuffle(unique_study_ids_all)
        num_test = (round(len(unique_study_ids_all) * self.test_percent))
        
        dev_sub_id = unique_study_ids_all[num_test:]
        self.df.loc[self.df.study_id.isin(dev_sub_id), 'split'] = 'training'
        
        self.dev_data = self.df[self.df['split'] == 'training']
        self.test_data = self.df[self.df['split'] == 'test']        

        unique_study_ids_dev = self.dev_data.study_id.unique()

        unique_study_ids_dev = shuffle(unique_study_ids_dev)
        num_train = (round(len(unique_study_ids_dev) * (1.0 - self.val_percent)))

        valid_sub_id = unique_study_ids_dev[num_train:]
        
        self.dev_data.loc[self.dev_data.study_id.isin(valid_sub_id), 'split'] = 'validation'
        
        self.train_data = self.dev_data[self.dev_data['split'] == 'training']
        self.val_data = self.dev_data[self.dev_data['split'] == 'validation']

        self.train_set = MultiViewDataset(self.train_data, augment=True)
        self.val_set = MultiViewDataset(self.val_data)
        self.test_set = MultiViewDataset(self.test_data)

        ## Calculate class weights
        train_labels = self.train_set.get_labels()        
        self.train_class_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
        self.weights = 1. / self.train_class_count

        print('samples (train): ',len(self.train_set))
        print('samples (val):   ',len(self.val_set))
        print('samples (test):  ',len(self.test_set))
    
    def train_dataloader(self):
        train_labels = self.train_set.get_labels()
        train_class_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
        weights = 1. / train_class_count
        weight = torch.from_numpy(weights)
        sampler = WeightedRandomSampler(weight, len(weight))
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        if probing:
            val_labels = self.val_set.get_labels()
            val_class_count = np.array([len(np.where(val_labels == t)[0]) for t in np.unique(val_labels)])
            weights = 1. / val_class_count
            weight = torch.from_numpy(weights)
            sampler = WeightedRandomSampler(weight, len(weight))
            return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False, sampler=sampler, num_workers=self.num_workers)
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    
    # Creates a new df with each row containing both MLO and CC views of an single side from an individual examination.
    def restructure_data(self, df_in):
        print("Restructuring data...")
        
        result_dict = {}
        views = []
        for _, row in df_in.iterrows():
            side = row['side']
            sideid_anon = str(row['acc_anon']) + side
            view_type = row['ViewPosition']
            path = row['path_1024png']
            label = row['label']
            if sideid_anon not in result_dict:
                result_dict[sideid_anon] = {'label':row['label'],  'study_id': row['empi_anon'], 'MLO_path': None, 'CC_path': None, 'MLO_lab': None, 'CC_lab': None}
            result_dict[sideid_anon][view_type + '_path'] = path
            result_dict[sideid_anon][view_type + '_lab'] = label

        # Convert the dictionary into a DataFrame
        return pd.DataFrame.from_dict(result_dict, orient='index').dropna()

class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, df, augment = False):
        self.augment = augment
        self.df = df

        # photometric data augmentation
        self.photometric_augment = T.Compose([
            GammaCorrectionTransform(gamma=0.2),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ])
        
        # geometric data augmentation
        self.geometric_augment = T.Compose([
            T.RandomApply(transforms=[T.RandomAffine(degrees=10, scale=(0.9, 1.1))], p=0.5),
        ])

        self.mlo_paths = df.MLO_path.to_numpy()
        self.cc_paths = df.CC_path.to_numpy()
        self.mlo_labels = df.MLO_lab.to_numpy()
        self.cc_labels = df.CC_lab.to_numpy()
        self.labels = df.label.to_numpy()

    def preprocess(self, image):
        # breast mask
        image_norm = image - np.min(image)
        image_norm = image_norm / np.max(image_norm)
        thresh = cv2.threshold(img_as_ubyte(image_norm), 5, 255, cv2.THRESH_BINARY)[1]

        # Connected components with stats.
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=4)

        # Find the largest non background component.
        max_label, _ = max(
            [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)],
            key=lambda x: x[1],
        )
        mask = output == max_label
        image[mask == 0] = 0
        
        return image
        
    def __getitem__(self, index): 
        # Randomly choose between mlo and cc, and unpack the paths accordingly
        source_path, target_path, label = random.choice([
            (self.mlo_paths[index], self.cc_paths[index], self.mlo_labels[index]),
            (self.cc_paths[index], self.mlo_paths[index], self.cc_labels[index])
        ])
        
        source = imread(source_path).astype(np.float32) / 65535.0
        target = imread(target_path).astype(np.float32) / 65535.0
        
        source = self.preprocess(source)
        target = self.preprocess(target)
        
        source = torch.from_numpy(source).unsqueeze(0)
        target = torch.from_numpy(target).unsqueeze(0)

        if self.augment:
            source = self.geometric_augment(self.photometric_augment(source))
        
        source = source.repeat(3,1,1)

        # Pad to square image
        pad_height = 1024 - 768 
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        source = torch.nn.functional.pad(source, (pad_top, pad_bottom, 0, 0), mode='constant', value=0)
        target = torch.nn.functional.pad(target, (pad_top, pad_bottom, 0, 0), mode='constant', value=0)

        return {'source':source, 'target':target, 'label': label}
    
    def __len__(self):
        return len(self.df)

    def get_labels(self):
        # Inaccurate: Only used to calculate approximate class weights.
        return self.labels

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convT1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)  
        self.bn1    = nn.BatchNorm2d(256)
        self.conv1a = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu1a = nn.ReLU() 

        self.convT2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0) 
        self.bn2    = nn.BatchNorm2d(128)
        self.conv2a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu2a = nn.ReLU() 

        self.convT3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.bn3    = nn.BatchNorm2d(64) 
        self.conv3a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3a = nn.ReLU() 

        self.convT4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
        self.bn4    = nn.BatchNorm2d(32)
        self.conv4a = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.relu4a = nn.ReLU() 

        self.convT5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0)
        self.bn5    = nn.BatchNorm2d(16)
        self.conv5a = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.relu5a = nn.ReLU() 

        self.convT6 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=0)
        self.bn6    = nn.BatchNorm2d(8)
        self.conv6a = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.relu6a = nn.ReLU() 

        self.convT7 = nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.convT1(x)
        x = self.bn1(x)
        x = nn.ReLU()(x)
        x = self.conv1a(x)
        x = self.relu1a(x)
        x = self.convT2(x)
        x = self.bn2(x)
        x = nn.ReLU()(x)
        x = self.conv2a(x)
        x = self.relu2a(x)
        x = self.convT3(x)
        x = self.bn3(x)
        x = nn.ReLU()(x)
        x = self.conv3a(x)
        x = self.relu3a(x)
        x = self.convT4(x)
        x = self.bn4(x)
        x = nn.ReLU()(x)
        x = self.conv4a(x)
        x = self.relu4a(x)
        x = self.convT5(x)
        x = self.bn5(x)
        x = nn.ReLU()(x)
        x = self.conv5a(x)
        x = self.relu5a(x)
        x = self.convT6(x)
        x = self.bn6(x)
        x = nn.ReLU()(x)
        x = self.conv6a(x)
        x = self.relu6a(x)
        x = self.convT7(x)
        return x

# class Decoder(nn.Module):
#     def block(self, in_channels, out_channels):
#         return [
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(),
#             nn.Conv2d(out_channels,out_channels, kernel_size=3, stride=1, padding=1),
#             nn.ReLU()
#         ]

#     def __init__(self, in_channels=512):
#         super().__init__()
#         self.layers = []
#         while in_channels != 8:
#             self.layers += self.block(in_channels, in_channels // 2)
#             in_channels = in_channels // 2
#         self.layers += [nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2, padding=0)]
#         self.decoder = nn.Sequential(*self.layers)

#     def forward(self, x):   
#         return self.decoder(x)

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(512, 512), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(512, 512), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Remove fc and avgpool layers to preserve spatial information
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])
        # Downsample using maxpool
        self.maxpool = nn.MaxPool2d(kernel_size=4, stride=4)
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.maxpool(x)
        x = self.decoder(x)
        return x

class MultiViewModel(pl.LightningModule):
    def __init__(self, class_weights, learning_rate=0.0001, checkpoint=None, predictions_dir=None, probing=False):
        super().__init__()
        
        self.predictions_dir = predictions_dir

        self.class_weights = torch.from_numpy(class_weights).float().cuda()
        self.class_weights = self.class_weights / self.class_weights.sum() * 2
        self.num_classes = 2

        self.model = EncoderDecoder()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) 

        if probing:
            self.probe = nn.Linear(512, self.num_classes)
            self.automatic_optimization = False

        if checkpoint is not None:
            print(self.model.load_state_dict(state_dict=checkpoint, strict=False))

        self.lr = learning_rate
        self.ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
        self.vgg_perceptual = VGGPerceptualLoss() 

        self.val_preds = []
        self.val_targets = []

    def probe_criterion(self, output, label):
        return torch.nn.functional.cross_entropy(output, label, weight=self.class_weights)

    def forward(self, x):
        return self.model.encoder(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        if probing:
            probe_optimizer = torch.optim.Adam(self.probe.parameters(), lr=self.lr)
            return [optimizer, probe_optimizer]
        return optimizer

    def reconstruction_loss(self, output, target, alpha = 0.15, beta = 0.15):
        normalised_output = (output-output.min())/(output.max()-output.min()) 
        l1_loss = torch.nn.functional.l1_loss(normalised_output, target)
        ms_ssim_loss = 1 - self.ms_ssim(normalised_output, target)
        perceptual_loss = self.vgg_perceptual(normalised_output, target)
        return ((1 - alpha - beta) * l1_loss + alpha * ms_ssim_loss + beta * perceptual_loss, \
         (l1_loss, ms_ssim_loss, perceptual_loss))

    def log_images(self, source_image, target_image, output):
        # Combine images into grids for logging
        target_grid = torchvision.utils.make_grid(target_image[0:4, ...], nrow=2, normalize=True)
        output_grid = torchvision.utils.make_grid(output[0:4, ...], nrow=2, normalize=True)

        # Log images to TensorBoard
        self.logger.experiment.add_image('Target Images', target_grid)
        self.logger.experiment.add_image('Generated Images', output_grid)

    def training_step(self, batch, batch_idx):
        source_image = batch['source']
        target_image = batch['target']
        labels = batch['label']

        if probing:
            opt1, opt2 = self.optimizers()

        # Self-supervised task
        output = self.model(source_image)  
        reconstruction_loss, losses = self.reconstruction_loss(output, target_image)
        if probing:
            self.manual_backward(reconstruction_loss)
            opt1.step()
            opt1.zero_grad()

            Linear probe training
            features = self(source_image)
            pooled_features = torch.flatten(self.avgpool(features) , 1)
            logits = self.probe(pooled_features)
            preds = torch.softmax(logits, dim=1)
            probe_loss = self.probe_criterion(preds, labels.to(torch.int64))
            self.manual_backward(probe_loss)
            opt2.step()
            opt2.zero_grad()

        if batch_idx % 50 == 0: 
            self.log_images(source_image, target_image, output)
        self.log('train_loss', reconstruction_loss, batch_size=batch_size)
        self.log('Smooth L1 Loss', losses[0], batch_size=batch_size)
        self.log('MS-SSIM Loss', losses[1], batch_size=batch_size)
        self.log('Perceptual Loss', losses[2], batch_size=batch_size)

        if probing:
            self.log('probe_loss', probe_loss, batch_size=batch_size)

        return reconstruction_loss

    def validation_step(self, batch, batch_idx):
        source_image = batch['source']
        target_image = batch['target']
        labels = batch['label']

        output = self.model(source_image)  
        loss, _ = self.reconstruction_loss(output, target_image)

        if probing:
            with torch.no_grad():
                features = self(source_image)
                pooled_features = torch.flatten(self.avgpool(features) , 1)
                logits = self.probe(pooled_features)
                preds = torch.softmax(logits, dim=1)
            probe_loss = self.probe_criterion(preds, labels.to(torch.int64))
            self.log('val_probe_loss', loss.mean(), batch_size=batch_size)
            self.val_preds.append(preds)
            self.val_targets.append(labels)


        self.log('val_reconstruction_loss', loss.mean(), batch_size=batch_size)

    def on_validation_epoch_end(self):
        if probing:
            preds = torch.cat(self.val_preds, dim=0)
            labels = torch.cat(self.val_targets, dim=0).to(torch.int64)

            auc = auroc(preds, labels, num_classes=2, average='macro', task='multiclass')
            self.log('probe_auc', auc, batch_size=len(preds))

            # Save Predictions
            cols_names = ['class_' + str(i) for i in range(0, 2)]
            df = pd.DataFrame(data=preds.cpu().detach(), columns=cols_names)    
            df['target'] = labels.cpu().detach()
            df.to_csv(self.predictions_dir + "/epoch_" + str(self.current_epoch) + ".csv", index=False)

            self.val_preds.clear()
            self.val_targets.clear()


    def test_step(self, batch, batch_idx):
        source_image = batch['source']
        target_image = batch['target']

        output = self(source_image) 
        loss, _ = self.reconstruction_loss(output, target_image)

        self.log('test_loss', loss.mean(), batch_size=batch_size)
        

def main(hparams):
    torch.set_float32_matmul_precision('high')
    pl.seed_everything(hparams.seed, workers=True)

    data_module = EMBEDData(val_percent=hparams.val, test_percent=hparams.test, batch_size=hparams.batch_size, num_workers=hparams.num_workers, probing=hparams.linear_probing)
        
    output_base = 'output'
    output_name = 'pretrain'
    output_preds ='predictions'
    output_dir = os.path.join(output_base, output_name)
    predictions_dir = os.path.join(output_dir, output_preds)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    
    model = MultiViewModel(class_weights=data_module.weights, predictions_dir=predictions_dir, probing=hparams.linear_probing)

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='mammogram-reconstruction-{epoch:02d}-{val_reconstruction_loss:.3f}',
        save_top_k=50,
        monitor='val_reconstruction_loss',
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        devices=hparams.num_devices,
        callbacks=[checkpoint_callback],
        logger=pl.loggers.TensorBoardLogger(output_base, name=output_name),
        log_every_n_steps=5,
    )

    trainer.fit(model, data_module)

    trainer.test(model, datamodule=data_module) 

    torch.save(model.model.state_dict(), 'MVSpretrained.pth')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--test', type=float, default=0.2)
    parser.add_argument('--val', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--batch_alpha', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--linear_probing', action='store_true')

    args = parser.parse_args()

    main(args)
