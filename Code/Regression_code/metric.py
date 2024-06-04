import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import pearsonr


def calculate_r_squared(y_true, y_pred):
    total_sum_of_squares = np.sum((y_true - np.mean(y_true))**2)
    residual_sum_of_squares = np.sum((y_true - y_pred)**2)
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r_squared

def calculate_pearson_correlation(y_true, y_pred):
    correlation, _ = pearsonr(y_true, y_pred)
    return correlation


def test(model: torch.nn.Module, test_loader, device,phase):
    model.eval()
    all_y_true = []
    all_y_pred = []
    with torch.no_grad():
        tbar = tqdm(enumerate(test_loader), disable=False, total=len(test_loader))
        for idx,(protein_batch, protein_lengths_batch,drug_batch, affinity_batch) in tbar:
            affinity_pred=model(protein_batch.to(device), protein_lengths_batch,drug_batch)
            # 将每个批次的真实值和预测值添加到列表中
            all_y_true.append(affinity_batch.cpu().numpy())
            all_y_pred.append(affinity_pred.squeeze().detach().cpu().numpy())
    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    if phase=="Predicted_data":
        return all_y_pred
    else:
        r_squared = calculate_r_squared(all_y_true, all_y_pred)
        correlation = calculate_pearson_correlation(all_y_true, all_y_pred)
        return (correlation,r_squared)