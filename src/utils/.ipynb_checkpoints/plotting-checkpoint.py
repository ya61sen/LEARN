import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

def plot_loss(hist, keys=['train_loss', 'val_loss'], figsize=(16,8)):
    plt.figure(figsize=figsize)

    plt.plot(np.arange(len(hist[keys[0]])), hist[keys[0]], 
                 label=keys[0].replace('_', ' ').capitalize(), alpha=0.8)
    plt.plot(np.arange(len(hist[keys[1]])), hist[keys[1]], 
                 label=keys[1].replace('_', ' ').capitalize(), alpha=0.8)

    plt.xlabel('Epoches', fontsize=20)
    plt.xticks(range(len(hist[keys[0]])), fontsize=10, rotation=45)
    plt.ylabel('Loss', fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Loss curves', fontsize=20)
    plt.legend(fontsize=20)
    plt.show()

def plot_roc(y_true, y_pred_proba, lw = 2, figsize=(8, 8)):
    # Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=figsize)
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate", fontsize=20)
    plt.ylabel("True Positive Rate", fontsize=20)
    plt.title("Receiver operating characteristic (ROC)", fontsize=20)
    plt.legend(loc="lower right", fontsize=20)
    plt.show()

def plot_cm(confusion_matrix, classes, figsize=(5, 4)):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=figsize)
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes, annot_kws={"size": 20})
    plt.title("Confusion Matrix", fontsize=20)
    plt.xlabel("Predicted label", fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("True label", fontsize=20)
    plt.show()

