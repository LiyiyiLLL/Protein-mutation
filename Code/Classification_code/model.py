import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data



class Embedding(nn.Module):
    def __init__(self,vocab_size,embedding_size):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, embedding_size)
    def forward(self, x):
        embedding = self.tok_embed(x)
        return embedding


# 定义蛋白质序列编码模型
class ProteinEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(ProteinEncoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_size)
        self.RNN = nn.RNN(input_size=embedding_size, hidden_size=embedding_size, batch_first=True)
        self.relu = nn.ReLU()
    def forward(self, x,lengths_tensor):
        x_embedding=self.embedding(x)
        packed_input = pack_padded_sequence(x_embedding, lengths=lengths_tensor, batch_first=True, enforce_sorted=False)
        # 将packed_input传递给RNN模型
        output, hidden = self.RNN (packed_input)
        # 解压缩模型输出
        unpacked_output, _ = pad_packed_sequence(output, batch_first=True)
        last_non_padding_indices = lengths_tensor - 1
        # 使用索引获取最后一个非填充单词的表示
        protein_representations = self.relu(torch.stack([unpacked_output[i, idx] for i, idx in enumerate(last_non_padding_indices)]))
        return protein_representations


# 辅助函数：将原子转换为原子序数
def get_adge_index(mol):
    edge_index = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[start, end], [end, start]])
    edge_index = torch.tensor(edge_index).t().contiguous()
    return edge_index


# 定义药物分子图编码模型
class DrugEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size,device):
        super(DrugEncoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size//2)
        self.conv2 = GCNConv(embedding_size//2, embedding_size//2)
        self.conv3=GCNConv(embedding_size//2, embedding_size)
        self.relu = nn.ReLU()
        self.device=device
    # 将药物SMILES转换为分子图
    def forward(self, data_drug):
        edge_index=get_adge_index(data_drug)
        x = []  # Node features
        num_atoms = data_drug.GetNumAtoms()
        for i in range(num_atoms):
            atom = data_drug.GetAtomWithIdx(i)
            atom_type = atom.GetAtomicNum()
            atom_embedding = self.embedding(torch.tensor(atom_type).to(self.device))
            x.append(atom_embedding)
        x = torch.stack(x, dim=0)
        data=(Data(x=x, edge_index=edge_index)).to(self.device)
        x, edge_index = data.x, data.edge_index
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x.mean(dim=0)



# 定义亲和力预测模型
class AffinityPredictor(nn.Module):
    def __init__(self, embedding_size):
        super(AffinityPredictor, self).__init__()
        self.fc = nn.Linear(embedding_size, embedding_size//2)
        self.classifier = torch.nn.Sequential(
         nn.Linear(embedding_size//2, 1),
         nn.Sigmoid())
    def forward(self, x):
        x = self.fc(x)
        x = self.classifier(x)
        return x


class Model(nn.Module):
    def __init__(self,vocab_size_drug,embedding_size_drug,vocab_size_protein, embedding_size_protein,embedding_size_classifier,device ):
        super(Model,self).__init__()
        self.drug_encoder=DrugEncoder(vocab_size_drug,embedding_size_drug,device)
        self.protein_encoder=ProteinEncoder(vocab_size_protein,embedding_size_protein)
        self.classifier=AffinityPredictor(embedding_size_classifier)

    def forward(self, protein_batch,protein_lengths_batch,drug_batch):
        protein_encoded =  self.protein_encoder(protein_batch, protein_lengths_batch)
        drug_encoded = torch.stack([self.drug_encoder(drug) for drug in drug_batch])
        # 将编码后的蛋白质和药物拼接起来
        combined = torch.cat((protein_encoded, drug_encoded), dim=1)
        affinity_pred = self.classifier(combined)
        return affinity_pred