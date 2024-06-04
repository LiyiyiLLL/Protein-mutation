import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pylab as plt



def caculate_metric(pred_y, labels, pred_prob):
    test_num = len(labels)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    ACC = float(tp + tn) / test_num
    # precision
    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)
    # SE
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)
    # SP
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)
    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    # F1-score
    if Recall + Precision == 0:
        F1 = 0
    else:
        F1 = 2 * Recall * Precision / (Recall + Precision)

    labels =labels.tolist()
    pred_prob = pred_prob.tolist()
    fpr, tpr, thresholds = roc_curve(labels, pred_prob, pos_label=1)
    AUC = auc(fpr, tpr)
    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(labels, pred_prob, pos_label=1)
    AUPR=auc(recall, precision)
    metric = (ACC, Precision, Sensitivity, Specificity, F1, AUC, MCC,AUPR)
    roc_data = [fpr, tpr, AUC]
    aupr_data = [recall, precision, AUPR]
    return metric,roc_data,aupr_data

def plot_roc(roc_data):
    plt.rc('font', family='Times New Roman')
    fpr1, tpr1=roc_data[0],roc_data[1]
    plt.plot(fpr1, tpr1, 'b', label='AUC = %0.2f' % roc_data[2])
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    save_path_pt =  'test_AUC.pdf'
    plt.savefig(save_path_pt)
    plt.close()

def plot_aupr(aupr_data):
    plt.rc('font', family='Times New Roman')
    fpr1, tpr1 = aupr_data[0], aupr_data[1]
    plt.plot(fpr1, tpr1, 'b', label='AUPR = %0.2f' % aupr_data[2])
    plt.legend(loc='lower left')
    plt.plot([1, 0], 'r--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall')
    save_path_pt =  'test_AUPR.pdf'
    plt.savefig(save_path_pt)
    plt.close()






def test(model: torch.nn.Module, test_loader, device,phase):
    model.eval()
    all_y_true = []
    all_y_pred_positive_probability = []
    with torch.no_grad():
        tbar = tqdm(enumerate(test_loader), disable=False, total=len(test_loader))
        for idx,(protein_batch, protein_lengths_batch,drug_batch, affinity_batch) in tbar:
            affinity_pred=model(protein_batch.to(device), protein_lengths_batch,drug_batch)
            # 将每个批次的真实值和预测值添加到列表中
            all_y_true.append(affinity_batch.cpu().numpy())
            all_y_pred_positive_probability.append(affinity_pred.squeeze().detach().cpu().numpy())
    all_y_true = np.concatenate(all_y_true)
    all_y_pred_positive_probability = np.concatenate(all_y_pred_positive_probability)
    all_y_pred = np.zeros((all_y_pred_positive_probability.shape[0],))
    # 将大于0.5的位置设置为1
    all_y_pred[all_y_pred_positive_probability > 0.5] = 1
    if phase=="Predicted_data":
        return all_y_pred
    else:
        metric,roc,pr= caculate_metric(all_y_pred, all_y_true, all_y_pred_positive_probability)
        plot_roc(roc)
        plot_aupr(pr)
        return metric


