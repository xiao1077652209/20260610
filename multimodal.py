import torch
import torch.nn as nn
from .seq_encoder import EnhancedCNN
from .attention import BiCrossAttention
from .cnn import SimpleCNN

class MultiModalModel(nn.Module):
    def __init__(self, text_dim=384, seq_inchannel=1, num_classes=7, hidden_dim=128):
        super().__init__()
        
        # 光谱特征提取
        self.seq_encoder = EnhancedCNN(seq_inchannel,num_classes)
        
        # 双向交叉注意力融合（文本+光谱）
        self.bi_cross_att = BiCrossAttention(text_dim, 128, hidden_dim)
        
        # 分类头（仅融合特征）
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, 64),  #(128, 64),  #
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, text_feat, seq_data):
        """
        text_feat: [batch_size, text_dim] 文本特征
        seq_data: [batch_size, 1, seq_len] 光谱数据
        return: logits, fusion_feat
        """
        # 1. 提取光谱特征
        seq_feat = self.seq_encoder(seq_data)  # [B, 128]
        #print('text_feat',text_feat.shape)
        #print('seq_feat',seq_feat.shape)
        # 2. 文本-光谱融合
        fusion_feat = self.bi_cross_att(text_feat, seq_feat)  # [B, 2*hidden_dim]
        #fusion_feat = torch.cat((text_feat, seq_feat), dim=-1) 
        #fusion_feat =seq_feat
        
        # 3. 分类预测
        logits = self.classifier(fusion_feat)
        
        return logits, fusion_feat