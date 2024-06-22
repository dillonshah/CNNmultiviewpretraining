import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score

def show_results(downstream, density, percentage):
    backbone = 'pretrained' if downstream else 'supervised'
    ctype = 'density' if density else 'malignancy'
    sub_dir2 = 'resnet18_' + (format(percentage, ".4g") if percentage != 1.0 else '1.0')
    print(f"Using {'pretrained' if downstream else 'supervised'} with {percentage}%")
    path = f"{ctype}_{backbone}/output/{sub_dir2}/predictions.csv"
    data = pd.read_csv(path)

    if density:
        preds = np.stack([data['class_0'], data['class_1'], data['class_2'], data['class_3']]).transpose()
    else:
        preds = np.stack([data['class_0'],data['class_1']]).transpose()
    targets = np.array(data['target'])
    print(np.unique(targets))   
    if density:
        # Binarize the output for multi-class ROC-AUC calculation
        n_classes = preds.shape[1]
        targets_bin = label_binarize(targets, classes=[0, 1, 2, 3])

        auc_scores = [] 
        auc_pr_scores = []
        roc_curves = []
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(targets_bin[:, i], preds[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)
            roc_curves.append((fpr, tpr, roc_auc))

            precision, recall, _ = precision_recall_curve(targets_bin[:, i], preds[:, i])
            auc_pr = average_precision_score(targets_bin[:, i], preds[:, i])
            auc_pr_scores.append(auc_pr)

        # Compute the macro-averaged AUC-ROC and AUC-PR
        average_auc = np.mean(auc_scores)
        average_auc_pr = np.mean(auc_pr_scores)
        print("Macro-Averaged AUC-ROC:", average_auc)
        print("Macro-Averaged AUC-PR:", average_auc_pr)
    else:
        fpr, tpr, _ = roc_curve(targets, preds[:,1])
        roc_auc = auc(fpr, tpr)
        print("ROC-AUC: ", roc_auc)
        roc_curves = [(fpr, tpr, roc_auc)]

        precision, recall, _ = precision_recall_curve(targets, preds[:, 1])
        auc_pr = average_precision_score(targets, preds[:, 1])
        print("AUC-PR: ", auc_pr)

    predicted_classes = np.argmax(np.abs(preds), axis=1)
    
    confusion = confusion_matrix(targets, predicted_classes)
    accuracy = accuracy_score(targets, predicted_classes)
    precision = precision_score(targets, predicted_classes, average='macro')
    sensitivity = recall_score(targets, predicted_classes, average='macro')  
    specificity = confusion[0, 0] / (confusion[0, 0] + confusion[0, 1])
    f1 = f1_score(targets, predicted_classes, average='macro')

    if density:
        class_labels = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
    else:
        class_labels = ['Benign', 'Malignant']

    print('F1-Score:', f1)

    return roc_curves

def plot_roc_curves(pretrained_roc_curves, supervised_roc_curves, density, percentage):
    fig, ax = plt.subplots(figsize=(7, 4))
    
    if density:
        for i in range(len(pretrained_roc_curves)):
            fpr, tpr, roc_auc = pretrained_roc_curves[i]
            plt.plot(fpr, tpr, lw=1, alpha=.8, label=f'MVS Class {i} AUC=%0.2f' % roc_auc)
        for i in range(len(supervised_roc_curves)):
            fpr, tpr, roc_auc = supervised_roc_curves[i]
            plt.plot(fpr, tpr, lw=2, linestyle='dotted', alpha=.8, label=f'ImageNet Class {i} AUC=%0.2f' % roc_auc)
    else:
        fpr, tpr, roc_auc = pretrained_roc_curves[0]
        plt.plot(fpr, tpr, lw=1.5, alpha=.8, label='Pretrained AUC=%0.2f' % roc_auc)
        fpr, tpr, roc_auc = supervised_roc_curves[0]
        plt.plot(fpr, tpr, lw=1.5, alpha=.8, linestyle='--', label='Supervised AUC=%0.2f' % roc_auc)
    
    ax.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='k', label='Chance', alpha=.8)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.legend(loc="lower right", fontsize=12)
    plt.title(f'{"Malignancy Detection" if not density else "Density Prediction"}', fontsize=12)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    plt.savefig(f"evaluation/roc_curve_comparison_{'malignancy' if not density else 'density'}_{percentage}.png")
    plt.show()

def main():
    while True:
        density = input("Density (y/n): ").upper() == 'Y'
        percentage = float(input("Enter percentage: "))
        
        pretrained_roc_curves = show_results(downstream=True, density=density, percentage=percentage)
        supervised_roc_curves = show_results(downstream=False, density=density, percentage=percentage)
        
        plot_roc_curves(pretrained_roc_curves, supervised_roc_curves, density, percentage)

if __name__ == "__main__":
    main()

