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
import sys
sys.path.append('../')

from pretraining.pretrain import EncoderDecoder

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
    def __init__(self, val_percent, test_percent, batch_size, num_workers, density, train_cut, proportional_val, seed):
        super().__init__()
        self.test_percent = test_percent
        self.val_percent = val_percent
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_cut = train_cut

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

        self.df['density_label'] = 0
        self.df.loc[self.df['tissueden'] == 1, 'density_label'] = 0
        self.df.loc[self.df['tissueden'] == 2, 'density_label'] = 1
        self.df.loc[self.df['tissueden'] == 3, 'density_label'] = 2
        self.df.loc[self.df['tissueden'] == 4, 'density_label'] = 3

        
        self.df['image_path'] = self.df.path_1024png.values
        self.df['study_id'] = [str(study_id) for study_id in self.df.empi_anon.values]
        self.df['label'] = self.df.label.values
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

        self.train_data = self.train_data.sample(frac=train_cut, random_state=seed)
        if proportional_val: self.val_data = self.val_data.sample(frac=train_cut, random_state=seed)

        ## DELETE ME
        # self.test_data = self.test_data.sample(frac=train_cut, random_state=seed)

        self.train_set = MammoDataset(self.train_data, augment=True, density=density)
        self.val_set = MammoDataset(self.val_data, density=density)
        self.test_set = MammoDataset(self.test_data, density=density)

        train_labels = self.train_set.get_labels()        
        self.train_class_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
        self.weights = 1. / self.train_class_count

        val_labels = self.val_set.get_labels()        
        val_class_count = np.array([len(np.where(val_labels == t)[0]) for t in np.unique(val_labels)])

        test_labels = self.test_set.get_labels()        
        test_class_count = np.array([len(np.where(test_labels == t)[0]) for t in np.unique(test_labels)])

        print('samples (train): ',len(self.train_set))
        print('samples (val):   ',len(self.val_set))
        print('samples (test):  ',len(self.test_set))
        if density:
            print('class_0/class_1/class_2/class_3 (train): {}/{}/{}/{}'.format(self.train_class_count[0], self.train_class_count[1], self.train_class_count[2], self.train_class_count[3]))
            print('class_0/class_1/class_2/class_3 (val):   {}/{}/{}/{}'.format(val_class_count[0], val_class_count[1], val_class_count[2], val_class_count[3]))
            print('class_0/class_1/class_2/class_3 (test):  {}/{}/{}/{}'.format(test_class_count[0], test_class_count[1], test_class_count[2], test_class_count[3]))
        else:
            print('pos/neg (train): {}/{}'.format(self.train_class_count[1], self.train_class_count[0]))
            print('pos/neg (val):   {}/{}'.format(val_class_count[1], val_class_count[0]))
            print('pos/neg (test):  {}/{}'.format(test_class_count[1], test_class_count[0]))
            print('pos (train):     {:0.2f}%'.format(self.train_class_count[1]/len(train_labels)*100.0))
            print('pos (val):       {:0.2f}%'.format(val_class_count[1]/len(val_labels)*100.0))
            print('pos (test):      {:0.2f}%'.format(test_class_count[1]/len(test_labels)*100.0))

    def train_dataloader(self):
        train_labels = self.train_set.get_labels()        
        samples_weight = np.array([self.weights[t] for t in train_labels])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return DataLoader(dataset=self.train_set, batch_size=self.batch_size, sampler=sampler, num_workers=self.num_workers, pin_memory=True)
   
    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    
class MammoDataset(torch.utils.data.Dataset):
    def __init__(self, df, augment = False, density=False):
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

        self.image_path = df.path_1024png.to_numpy()
        self.label = df.density_label.to_numpy() if density else df.label.to_numpy()
    

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
        image_path = self.image_path[index]
        label = self.label[index]
        
        image = imread(image_path).astype(np.float32) / 65535.0
        
        image = self.preprocess(image)
        
        image = torch.from_numpy(image).unsqueeze(0)

        if self.augment:
            image = self.geometric_augment(self.photometric_augment(image))
        
        image = image.repeat(3,1,1)

        # Pad to square image
        pad_height = 1024 - 768 
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        image = torch.nn.functional.pad(image, (pad_top, pad_bottom, 0, 0), mode='constant', value=0)

        return {'image':image, 'label':label}

    def __len__(self):
        return len(self.df)

    def get_labels(self):
        return self.label


class MammogramModel(pl.LightningModule):
    def __init__(self, class_weights, num_classes, batch_size, pretrained=False, model_path=None, linear_probing=False, learning_rate=0.001):
        super().__init__()

        #Normalise weights
        self.class_weights = torch.from_numpy(class_weights).float().cuda()
        self.class_weights = self.class_weights / self.class_weights.sum() * 2
        self.num_classes = num_classes
        self.batch_size = batch_size

        self.pretrained = pretrained
        
        if self.pretrained:
            self.fc_in_features = 512
            self.model = EncoderDecoder()

            if model_path: self.model.load_state_dict(torch.load(model_path))

            # Discard decoder head
            self.model = self.model.encoder

            # Freeze the backbone
            if linear_probing:
                for param in self.model.parameters():
                    param.requires_grad = False

            # Replace avgpool layer
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(self.fc_in_features, self.num_classes)
        else: 
            self.model = models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            self.fc_in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        

        self.lr = learning_rate
        
        self.predictions = []
        self.targets = []
        self.train_step_preds = []
        self.train_step_trgts = []
        self.val_step_preds = []
        self.val_step_trgts = []

    def forward(self, x):
        x = self.model(x)
        if self.pretrained:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def process_batch(self, batch):
        image, label = batch['image'], batch['label']
        output = self.forward(image)
        prd = torch.softmax(output, dim=1)
        loss = torch.nn.functional.cross_entropy(output, label, weight=self.class_weights)
        return loss, prd, label

    def training_step(self, batch, batch_idx):
        loss, prd, label = self.process_batch(batch)
        self.train_step_preds.append(prd)
        self.train_step_trgts.append(label)
        self.log('train_loss', loss, batch_size=self.batch_size)
        batch_ratio = 0 if len(np.where(label.cpu().numpy() == 0)[0]) == 0 else len(np.where(label.cpu().numpy() == 1)[0]) / len(np.where(label.cpu().numpy() == 0)[0])
        self.log('batch_ratio', batch_ratio, batch_size=self.batch_size) 
        return loss

    def on_train_epoch_end(self):
        all_preds = torch.cat(self.train_step_preds, dim=0)
        all_trgts = torch.cat(self.train_step_trgts, dim=0)
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')
        self.log('train_auc', auc, batch_size=len(all_preds))
        self.train_step_preds.clear()
        self.train_step_trgts.clear()


    def validation_step(self, batch, batch_idx):
        loss, prd, label = self.process_batch(batch)
        self.val_step_preds.append(prd)
        self.val_step_trgts.append(label)
        self.log('val_loss', loss, batch_size=self.batch_size)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_step_preds, dim=0)
        all_trgts = torch.cat(self.val_step_trgts, dim=0)
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')
        self.log('val_auc', auc, batch_size=len(all_preds))
        self.val_step_preds.clear()
        self.val_step_trgts.clear()

    def on_test_start(self):
        self.predictions = []
        self.targets = []

    def test_step(self, batch, batch_idx):
        _, prd, label = self.process_batch(batch)
        self.predictions.append(prd.detach().cpu())
        self.targets.append(label.squeeze().cpu())

class MammoEmbeddings(MammogramModel):
    def __init__(self, class_weights, num_classes, batch_size, pretrained, init=None):
        super().__init__(class_weights=class_weights, num_classes=num_classes, batch_size=batch_size, pretrained=pretrained)
        if init is not None:
            self.model = init.model
    
    def on_test_start(self):
        self.fc = nn.Identity(512)
        self.embeddings = []

    def test_step(self, batch, batch_idx):
        emb = self.forward(batch['image'])
        self.embeddings.append(emb.detach().cpu())

def save_embeddings(model, output_fname):
    embeddings = torch.cat(model.embeddings, dim=0)
    df = pd.DataFrame(data=embeddings)
    df.to_csv(output_fname, index=False)

def save_predictions(model, output_fname, num_classes):
    predictions = torch.cat(model.predictions, dim=0)
    targets = torch.cat(model.targets, dim=0)
    cols_names = ['class_' + str(i) for i in range(0, num_classes)]
    df = pd.DataFrame(data=predictions, columns=cols_names)    
    df['target'] = targets
    df.to_csv(output_fname, index=False)
        
def main(hparams):
    print(f""" 
    {65*'*'}
    Performing {'Breast Density Prediction' if hparams.density else 'Breast Cancer Detection'} using {'pretrained backbone' if hparams.pretrained else 'ImageNet weighted ResNet-18'} 
    with {100 * hparams.train_percent}% of training data{ 'with Linear Probing' if hparams.linear_probing else ''}.
    No. Epochs: {hparams.epochs}
    Batch Size: {hparams.batch_size}
    Test percentage: {100*hparams.test}%
    Val percentage: {100*hparams.val}%
    Seed: {hparams.seed}
    {65*'*'}
    """)

    torch.set_float32_matmul_precision('high')
    pl.seed_everything(hparams.seed, workers=True)

    num_classes = 4 if hparams.density else 2

    data_module = EMBEDData(
        val_percent=hparams.val,
        test_percent=hparams.test,
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        density=hparams.density,
        train_cut=hparams.train_percent,
        proportional_val=hparams.proportional_val,
        seed=hparams.seed
    )

    model = MammogramModel(
        model_path=hparams.model_path if hparams.pretrained else None,
        num_classes=num_classes,
        batch_size=hparams.batch_size,
        class_weights=data_module.weights,
        pretrained=hparams.pretrained,
        linear_probing=hparams.linear_probing,
    )

    output_base = 'output'
    output_name = 'resnet18_' + str(hparams.train_percent)
    output_dir = os.path.join(output_base, output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints_{'density' if hparams.density else 'malignancy'}_classification{'_pretrained' if hparams.pretrained else ''}_{hparams.train_percent}/",
        filename='mammogram-classification'+'-{epoch:02d}-{val_loss:.3f}',
        save_top_k=3,
        monitor='val_loss',
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

    torch.save(model.model.state_dict(), f"{'density' if hparams.density else 'malignancy'}_classification{'_pretrained' if hparams.pretrained else ''}_{hparams.train_percent}.pth")

    trainer.test(model, datamodule=data_module) 

    save_predictions(model=model, output_fname=os.path.join(output_dir, 'predictions.csv'), num_classes=num_classes)

    model_modified = MammoEmbeddings(class_weights=data_module.weights, num_classes=num_classes, batch_size=hparams.batch_size, pretrained=hparams.pretrained, init=model)
    trainer.test(model=model_modified, datamodule=data_module, ckpt_path=trainer.checkpoint_callback.best_model_path)#
    save_embeddings(model=model_modified, output_fname=os.path.join(output_dir, 'embeddings.csv'))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--test', type=float, default=0.2)
    parser.add_argument('--val', type=float, default=0.1)
    parser.add_argument('--train_percent', type=float, default=1)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--batch_alpha', type=float, default=1.0)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_devices', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--density', action='store_true')
    parser.add_argument('--proportional_val', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--linear_probing', action='store_true')

    args = parser.parse_args()

    main(args)
