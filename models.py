import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_1D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            Mish(),
            nn.Dropout(0.1),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            Mish(),
            nn.Dropout(0.1),
            
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            Mish(),
            nn.Dropout(0.1),

            nn.AdaptiveAvgPool1d(1)
        )            

    def forward(self, x):        
        features = self.conv_layers(x).squeeze(-1)  # [B, 64]
        return features
    
class CNN_SMNK1D(nn.Module):
    '''串行递进式多尺度特征提取'''
    def __init__(self, in_channels):
        super().__init__()
        # 窄核卷积特征提取块，串行递进式多尺度特征提取
        self.narrow_conv_block = nn.Sequential(
            # 第一阶段：单通道独立提取，分组卷积实现通道隔离
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=3,  # 小尺度窄核
                padding=1,
                groups=in_channels  # 每个输入通道单独卷积
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第二阶段：跨通道融合 + 中尺度窄核
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,  # 中尺度窄核
                padding=2,
                groups=1  # 跨通道融合
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            
            # 第三阶段：大尺度窄核捕捉长时序依赖
            nn.Conv1d(
                in_channels=64,
                out_channels=64,
                kernel_size=7,  # 大尺度窄核
                padding=3,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第四阶段：全局池化提取全局特征
            nn.AdaptiveAvgPool1d(1)  # 全局池化得到固定维度特征
        )
    
    def forward(self, x):
        features = self.narrow_conv_block(x).squeeze(-1) # 输出特征shape: [B, 64]
        return features

class CNN_PMNK1D(nn.Module):
    '''并行多尺度特征提取'''
    def __init__(self, in_channels):
        super().__init__()
        
        # 并行多尺度分支（输出特征保持时序长度，便于后续1×N卷积融合）
        # 分支1：小尺度：kernel_size=3
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),  # 输出32通道
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # 分支2：中尺度（kernel_size=5
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),  # 输出32通道
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # 分支3：中大尺度：kernel_size=7
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),  # 输出32通道
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # 1×N卷积融合模块（N=4，同时融合通道和时序特征）
        # 输入：3个分支拼接 → [B, 32+32+32=96, L]，输出：64通道特征 → 再通过全局池化压缩为64维
        self.fusion_conv = nn.Sequential(
            # 1×3卷积（这里用Conv1d(kernel_size=3)实现，等价于1×3二维卷积的时序融合）
            nn.Conv1d(96, 64, kernel_size=3, padding=1),  # 融合96通道→64通道，同时融合3个时序点
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.AdaptiveAvgPool1d(1)
        )
        
    def forward(self, x):
        # 输入x：[B, 1, L]
        feat1 = self.branch1(x)  # [B, 32, L]
        feat2 = self.branch2(x)  # [B, 32, L]
        feat3 = self.branch3(x)  # [B, 32, L]
        
        # 拼接多分支特征（通道维度）→ [B, 96, L]
        fused = torch.cat([feat1, feat2, feat3], dim=1)
        
        # 1×3卷积融合（同时融合通道和时序特征）→ [B, 64, 1]
        output_feat = self.fusion_conv(fused).squeeze(-1)  # [B, 64]
        
        return output_feat

class CNN_SPMNK1D(nn.Module):
    '''串并行混合多尺度分支'''
    def __init__(self, in_channels):
        super().__init__()
        # 第一部分：并行多尺度特征提取
        self.branch1 = nn.Sequential(  # 小尺度
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            Mish(),
            nn.Dropout(0.2),
            
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            Mish(),
            nn.Dropout(0.2),
        )
        self.branch2 = nn.Sequential(  # 中尺度
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            Mish(),
            nn.Dropout(0.2),

            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            Mish(),
            nn.Dropout(0.2),
        )
        
        self.branch3 = nn.Sequential(  # 大尺度
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            Mish(),
            nn.Dropout(0.2),

            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            Mish(),
            nn.Dropout(0.2),
        )
        
        self.branch4 = nn.Sequential(  # 更大尺度
            nn.Conv1d(in_channels, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            Mish(),
            nn.Dropout(0.2),
            
            nn.Conv1d(32, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            Mish(),
            nn.Dropout(0.2),
        )
        
        # 并行特征融合（1×3卷积初步融合通道与时序）
        self.parallel_fusion = nn.Sequential(
            nn.Conv1d(32*4, 64, kernel_size=3, padding=1),  # 32×4=128输入通道
            nn.BatchNorm1d(64),
            Mish(),
            nn.Dropout(0.2)
        )
        
        # 第二部分：串行递进式特征深化
        self.serial_conv = nn.Sequential(
            # 深化中尺度特征（基于并行融合后的特征）
            nn.Conv1d(64, 64, kernel_size=5, padding=2),  # 中尺度窄核
            nn.BatchNorm1d(64),
            Mish(),
            nn.MaxPool1d(2),  # 压缩时序维度，聚焦关键特征
            nn.Dropout(0.2),
            
            # 深化大尺度特征（捕捉长程依赖）
            nn.Conv1d(64, 64, kernel_size=7, padding=3),  # 大尺度窄核
            nn.BatchNorm1d(64),
            Mish(),
            nn.Dropout(0.2),
            
            # 全局池化输出64维特征
            nn.AdaptiveAvgPool1d(1)
        )
    
    def forward(self, x):
        # 输入x: [B, in_channels, L]
        
        # 1. 并行分支提取多尺度特征
        feat1 = self.branch1(x)  # [B, 16, L]
        feat2 = self.branch2(x)  # [B, 16, L]
        feat3 = self.branch3(x)  # [B, 16, L]
        feat4 = self.branch4(x)  # [B, 16, L]
        
        # 2. 拼接并行特征并初步融合
        parallel_feat = torch.cat([feat1, feat2, feat3, feat4], dim=1)  # [B, 64, L]
        fused_parallel = self.parallel_fusion(parallel_feat)  # [B, 64, L]
        
        # 3. 串行卷积深化特征
        deep_feat = self.serial_conv(fused_parallel)  # [B, 64, 1]
        
        # 输出64维特征
        return deep_feat.squeeze(-1)  # [B, 64]

class CNN_SBPMNK1D(nn.Module):
    '''分支并行提取多尺度特征，每个分支内部串行卷积深化特征提取'''
    def __init__(self, in_channels):
        super().__init__()
        # 每个分支内部包含两层串行卷积（同尺度窄核），深化该尺度的特征提取
        # 分支1：小尺度（kernel_size=3→3）
        self.branch1 = nn.Sequential(
            # 第一层：捕捉该尺度的基础特征
            nn.Conv1d(in_channels, 32, kernel_size=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),  
            # 第二层：深化该尺度的特征，同核尺寸，提取更抽象的细节
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # 分支2：中尺度（kernel_size=5→5）
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),  

            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # 分支3：大尺度（kernel_size=7→7）
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(32, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # 分支4：大尺度（kernel_size=9→9）
        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv1d(32, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 1×3卷积融合模块（融合4个分支的特征，输出64维）
        self.fusion_conv = nn.Sequential(
            # 输入：4个分支拼接 → 32×4=128通道
            nn.Conv1d(128, 64, kernel_size=3, padding=1),  # 融合通道与时序特征
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(1)  # 全局池化得到64维特征
        )

        # 注意力机制（可选）
        self.attention = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(1)
        )
        
    def forward(self, x):
        # 输入x: [B, in_channels, L]
        
        # 各分支提取深化后的多尺度特征（每个分支输出[B, 32, L]）
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)
        
        # 拼接所有分支特征 → [B, 32×4=128, L]
        fused = torch.cat([feat1, feat2, feat3, feat4], dim=1)
        
        # 融合并输出64维特征
        output_feat = self.fusion_conv(fused).squeeze(-1)  # [B, 64]
        # 应用注意力机制（可选）
        attention_weights = self.attention(output_feat.unsqueeze(-1))  # [B, 64, 1]
        output_feat = output_feat * attention_weights.squeeze(-1)  # [B, 64]
        
        return output_feat

class CNN_2D(nn.Module):
    def __init__(self, in_channels):
        # 改为多尺度卷积
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm2d(32),
            Mish(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            Mish(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
    def forward(self, x):
        features = self.features(x).view(x.size(0), -1)  # [B,64]
        return (features) # 展平为 (batch_size, 128) 的特征向量

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # 短路连接（仅维度不匹配时使用）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet_2D(nn.Module):
    def __init__(self, in_channels, layers=[1, 1], out_features=32):
        super(ResNet_2D, self).__init__()
        self.in_channels = 32

        # 初始卷积层（适配64x64输入）
        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        
        # 残差块层
        self.layer1 = self._make_layer(BasicBlock, out_channels=32, num_blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, out_channels=64, num_blocks=layers[1], stride=2)
        
        # 全局平均池化（输出特征向量）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入形状: [B, 1, 64, 64]（适配你的2D图像输入）
        x = self.conv1(x)       # [B, 32, 64, 64]
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)      # [B, 32, 64, 64]
        x = self.layer2(x)      # [B, 64, 32, 32]
        
        x = self.avgpool(x)     # [B, 64, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 64] 输出64维特征向量

        return x

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

# 1. 注意力机制
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),
            Mish(),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x_ca = x * ca
        sa_input = torch.cat([x_ca.mean(dim=1, keepdim=True), 
                             x_ca.max(dim=1, keepdim=True)[0]], dim=1)
        sa = self.spatial_attention(sa_input)
        x_sa = x_ca * sa
        return x_sa

# 2. 多尺度块
class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 双分支设计：1×1（捕捉全局信息）+ 3×3（捕捉局部信息）
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels//2),
            Mish()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels//2),
            Mish()
        )
        # 短连接（缓解梯度消失，适配浅层网络）
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        self.bn_shortcut = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # 双分支特征拼接
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        concat_out = torch.cat([branch1_out, branch2_out], dim=1)  # 拼接通道
        # 短连接残差融合
        shortcut_out = self.bn_shortcut(self.shortcut(x))
        return Mish()(concat_out + shortcut_out)  # 残差+激活

# 3. 完整模型
class HMCNN_2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super(HMCNN_2D, self).__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            Mish(),
            nn.MaxPool2d(2)
        )
        self.ms_block1 = MultiScaleBlock(16, 32)
        self.ms_block2 = MultiScaleBlock(32, 64)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.feat_dim = out_channels

        self.light_pool = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            Mish(),
            nn.AdaptiveAvgPool2d(1)  # 压缩为[B, 64, 1, 1]
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.ms_block1(x)
        x = self.ms_block2(x)
        x = self.light_pool(x)      # 得到[B, 64, 1, 1]的特征
        x = x.view(x.size(0), -1)   # 将特征展平为向量形式 [B, 64]

        return x

class CrossModalAttention(nn.Module):
    def __init__(self, dim1, dim2, hidden_dim=None):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.hidden_dim = hidden_dim or max(dim1, dim2)

        # 1. 模态特征投影（统一维度并增加非线性）
        self.proj1 = nn.Sequential(
            nn.Linear(dim1, self.hidden_dim),
            Mish(),
        )
        self.proj2 = nn.Sequential(
            nn.Linear(dim2, self.hidden_dim),
            Mish(),
        )
        
        # 2. 双向注意力机制
        self.cross_attn1 = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.2,
        )  # 模态1关注模态2
        self.cross_attn2 = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=8,
            batch_first=True,
            dropout=0.2,
        )  # 模态2关注模态1
        
        # 3. 模态共享特征编码
        self.share = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # 输入是两个hidden_dim拼接
            Mish()
        )

        # 3.1 模态关联编码
        self.relation = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
        # 3.2 门控融合权重
        self.gate = nn.Sequential(
            nn.Linear(self.hidden_dim, 2),  # 两个模态的权重
            nn.Softmax(dim=-1)
        )
        
        # 归一化层（增强稳定性）
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.norm_final = nn.LayerNorm(self.hidden_dim)

    def forward(self, x1, x2):
        # 确保输入是三维的 [batch, seq_len, dim]
        x1 = x1.unsqueeze(1)  # 添加序列维度
        x2 = x2.unsqueeze(1)  # 添加序列维度
        
        # 步骤1：特征投影（统一维度）
        x1_proj = self.proj1(x1)  # [B, L, hidden_dim]
        x2_proj = self.proj2(x2)  # [B, L, hidden_dim]
        
        # 步骤2：双向交叉注意力（模态互关注）
        attn1, _ = self.cross_attn1(query=x1_proj, key=x2_proj, value=x2_proj)
        x1_attn = self.norm1(x1_proj + attn1)  # 残差+归一化
        
        attn2, _ = self.cross_attn2(query=x2_proj, key=x1_proj, value=x1_proj)
        x2_attn = self.norm2(x2_proj + attn2)  # 残差+归一化
        
        # 步骤3：计算模态间关联度
        x1_global = torch.mean(x1_attn, dim=1)  # [B, hidden_dim]
        x2_global = torch.mean(x2_attn, dim=1)  # [B, hidden_dim]
        
        # 计算每对样本的关联分数
        shared = self.share(torch.cat([x1_global, x2_global], dim=-1))  # [batch, hidden_dim]
        relation = self.relation(shared)  # [batch, 1]

        # 步骤4：动态权重融合
        gate_weights = self.gate(shared)  # [B, 2]
        fused_global = (
            gate_weights[:, 0:1] * x1_global * relation +
            gate_weights[:, 1:2] * x2_global * relation 
        )

        return self.norm_final(fused_global), relation

class Classifier(nn.Module):# 分类器
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.BatchNorm1d(32),
            Mish(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)
