import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

def calculate_roc_auc(pred, target):
    pred = torch.cat(pred, dim=0).detach().numpy()
    target = torch.cat(target, dim=0)

    target = target.reshape((target.shape[0], 1))
    target_onehot = torch.zeros(target.shape[0], 3)
    target_onehot.scatter_(dim=1, index=target, value=1)
    target_onehot = np.array(target_onehot)

    # Micro
    fpr, tpr, roc_auc = dict(), dict(), dict()
    fpr["micro"], tpr["micro"], _ = roc_curve(target_onehot.ravel(), pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Per class
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(target_onehot[:, i], pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(3):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation
    mean_tpr /= 3
    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    return fpr, tpr, roc_auc


def plot_roc_curve(fpr, tpr, roc_auc, image_path):
    fig, ax = plt.subplots(figsize=(6, 6))

    # plt.plot(
    #     fpr["micro"],
    #     tpr["micro"],
    #     label=f"micro average (AUC={roc_auc['micro']:.4f})",
    #     color="deeppink",
    #     linewidth=2,
    # )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro average (AUC={roc_auc['macro']:.4f})",
        color="navy",
        linewidth=2,
    )

    colors = ["aqua", "darkorange", "cornflowerblue"]
    for class_id, color in zip(range(3), colors):
        plt.plot(
            fpr[class_id],
            tpr[class_id],
            label=f"class {class_id} (AUC={roc_auc[class_id]:.4f})",
            color=color,
            linewidth=2,
        )

    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC to One-vs-Rest multiclass")
    plt.legend()
    plt.savefig(image_path)
    plt.clf()
    plt.close()