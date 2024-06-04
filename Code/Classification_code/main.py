device_set="cuda:0"
DataInput_address="../data/protein_mutation_data.csv"
amino_acids = 'ACDEFGHIKLMNPQRSTVWY$'#氨基酸词汇表
vocab_size_protein=len(amino_acids)
vocab_size_drug=118#元素周期表最大序数
batch_size=32
protein_embedding_size=drug_embedding_size=256
LR=0.0001
Epoch=50
random_seed=1
split_ratio_train_test=0.7#训练集和测试集的划分比例


# 58.72609202067057

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
import os
from datetime import datetime
from dataset import Dataset_train_and_test
from model import Model
from metric import test
start = datetime.now()
print('start at ', start)
device = torch.device(device_set)





#固定随机种子，保证模型可复现
def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(1)
setup_seed(random_seed)





# 定义数据预处理函数
def collate_fn(batch):
    protein_sequence_number=[item[0] for item in batch]
    # 计算每个序列的长度，并找到批次中最大的序列长度
    sequence_lengths = [len(seq) for seq in protein_sequence_number]
    max_length = max(sequence_lengths)
    # 填充序列，使它们的长度都相同
    padded_sequences = [seq + [0] * (max_length - len(seq)) for seq in protein_sequence_number]
    # 将填充后的序列转换为张量
    # 创建一个与填充后的序列长度相对应的张量
    protein_lengths_batch = torch.tensor(sequence_lengths)
    protein_batch = torch.tensor(padded_sequences)
    drug_batch = [item[1] for item in batch if item[1] is not None]
    affinity_batch = torch.tensor([item[2] for item in batch if item[1] is not None], dtype=torch.float)
    return protein_batch,protein_lengths_batch, drug_batch, affinity_batch





# 实例化数据集，包括训练集和测试集
data_loaders = {phase_name: DataLoader(Dataset_train_and_test(DataInput_address,phase_name ,amino_acids,random_seed,split_ratio_train_test  ),
                                                                        batch_size=batch_size,pin_memory=True,shuffle=False,collate_fn=collate_fn)
                                                                        for phase_name in ['Train_data', 'Test_data','Predicted_data']}





# 定义预测模型，优化器和损失函数
model=Model(vocab_size_drug, drug_embedding_size,vocab_size_protein, protein_embedding_size,protein_embedding_size+drug_embedding_size,device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_function = nn.BCELoss(reduction='sum')





# 训练模型
metric_test_record=0
model.to(device)
scheduler = StepLR(optimizer, step_size=5, gamma=0.9)  # 每5个epoch后将学习率乘以0.9
for epoch in range(Epoch):
    model.train()
    tbar = tqdm(enumerate(data_loaders['Train_data']), disable=False,total=len(data_loaders['Train_data']))
    total_loss = 0
    for idx,(protein_batch, protein_lengths_batch,drug_batch, affinity_batch) in tbar:
        optimizer.zero_grad()
        affinity_pred=model(protein_batch.to(device), protein_lengths_batch,drug_batch)
        loss = loss_function(affinity_pred.squeeze(), affinity_batch.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(data_loaders)}')
    metric_test = test(model, data_loaders["Test_data"], device, "Test_data")
    if metric_test[0]>0.55 and metric_test[0]>metric_test_record:
        metric_test_acc_record=metric_test[0]

        torch.save(model.state_dict(), './best_model.pt')
        print("性能超过之前最好的epoch,进行模型保存")





model.load_state_dict(torch.load('./best_model.pt'))
#计算训练好的模型的预测指标
metric_train = test(model, data_loaders["Train_data"], device,"Train_data")
metric_test = test(model, data_loaders["Test_data"], device,"Test_data")
#保存指标结果
performance_result = pd.DataFrame([metric_train, metric_test], columns=['ACC', 'Pre','Sen','Spe','F1','AUC','MCC','AUPR'], index=['train_set', 'test_set'])
# 写入 Excel 文件
performance_result.to_excel('performance_result.xlsx')


#计算突变体的结合亲和力
result = test(model, data_loaders['Predicted_data'], device,'Predicted_data')
# 读取 CSV 文件
file_path = '../data/protein_mutation_data.csv'
df = pd.read_csv(file_path)
df['Label_cls_Mut'] = result
df.to_csv('./result.csv', index=False)



print('finished!!!!!!!!!!!!')
end = datetime.now()
print('end at:', end)
print('time used:', str(end - start))