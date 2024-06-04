from torch.utils.data import Dataset
from rdkit import Chem
import pandas as pd


# 辅助函数：将氨基酸转换为索引
def aa_to_index(sequence,amino_acids):
    Legal_amino_acids=amino_acids[:-1]
    index=[]
    for aa in sequence:
        if aa not in Legal_amino_acids:
            index.append(amino_acids.index(amino_acids[-1])+1)
        else:
            index.append (amino_acids.index(aa)+1)
    return index



# 定义数据集类
class Dataset_train_and_test(Dataset):
    def __init__(self,data_path, phase,amino_acids,random_seed,split_ratio ):
        data = pd.read_csv(data_path)
        data_shuffled=data.sample(frac=1, random_state=random_seed)
        train_size = int(split_ratio * len(data_shuffled))
        affinity=(data_shuffled.iloc[:, 5]).tolist()
        self.amino_acids = amino_acids

        if phase == 'Train_data':
            self.data=data_shuffled.iloc[:train_size]
            self.protein_seq = (self.data.iloc[:, 1]).tolist()
            self.drug_smiles = (self.data.iloc[:, 3]).tolist()
            self.affinity = affinity[:train_size]
            # 将药物SMILES转换为分子图
            self.mol = [Chem.MolFromSmiles(smiles) for smiles in self.drug_smiles]
            # 将蛋白质序列转换为索引
            self.protein_idx = [aa_to_index(sequence, self.amino_acids) for sequence in self.protein_seq]
        elif phase == 'Test_data':
            self.data =data_shuffled.iloc[train_size:]
            self.protein_seq = (self.data.iloc[:, 1]).tolist()
            self.drug_smiles = (self.data.iloc[:, 3]).tolist()
            self.affinity = affinity[train_size:]
            # 将药物SMILES转换为分子图
            self.mol = [Chem.MolFromSmiles(smiles) for smiles in self.drug_smiles]
            # 将蛋白质序列转换为索引
            self.protein_idx = [aa_to_index(sequence, self.amino_acids) for sequence in self.protein_seq]
        elif phase=='Predicted_data':

            self.data = data
            self.protein_seq = (self.data.iloc[:, 0]).tolist()
            self.drug_smiles = (self.data.iloc[:, 3]).tolist()
            self.affinity = (self.data.iloc[:, 5]).tolist()#占位作用
            # 将药物SMILES转换为分子图
            self.mol = [Chem.MolFromSmiles(smiles) for smiles in self.drug_smiles]
            # 将蛋白质序列转换为索引
            self.protein_idx = [aa_to_index(sequence, self.amino_acids) for sequence in self.protein_seq]

    def __getitem__(self, idx):
        return self.protein_idx[idx], self.mol[idx], self.affinity[idx]

    def __len__(self):
        return len(self.data)