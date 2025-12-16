import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.io as sio
import os
import json
from sklearn.preprocessing import StandardScaler
import torch
from utils.to2D import wavetoimagetransf
from models import *

# 字体配置
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']    # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False      # 修复负号显示问题

def tSNE_visualize_modality_comparison(features1, features2, labels, labels_Cn, title="模态的特征分布对比"):
    """
    可视化两个模态的特征分布，按模态和类别着色
    ，同类聚集性：若同一类别的点（同颜色）在两种模态中都紧密聚集，说明该模态对这类路面的特征提取能力强。
    ，类别区分度：不同颜色的点在两种模态中是否明显分离。分离越清晰，说明该模态的特征对类别区分更有效。
    ，模态一致性：两种模态中，同类别的点分布是否相似（如聚集区域是否重合）。若相似，说明两种模态捕捉到的特征具有一致性；若差异大，说明两种模态可能互补（各有侧重）。
    """
    # 合并特征
    all_features = np.vstack([features1, features2])
    modality_labels = ['Modality1'] * len(features1) + ['Modality2'] * len(features2)
    
    # 创建完整的标签数组，将原始标签重复两次分别对应两个模态
    all_labels = np.concatenate([labels, labels])

    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_features = tsne.fit_transform(all_features)
    
    # 绘制图形
    fig_vmc, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：按模态着色
    # 若两种颜色的点明显分离，说明两种模态的特征分布差异较大；若大量重叠，说明两种模态的特征在低维空间中表现相似。
    # 可用于判断两种模态是否捕获了相似的信息，重叠度高则信息相似，分离度高则信息差异大。
    for modality, color in zip(['Modality1', 'Modality2'], ['blue', 'red']):
        mask = [m == modality for m in modality_labels]
        ax1.scatter(reduced_features[mask, 0], reduced_features[mask, 1], 
                   c=color, label=modality, alpha=0.6)
    ax1.set_title(f'{title} - By Modality')
    ax1.legend()
    
    # 右图：按真实标签着色
    # 若同一颜色的点聚集在一起，说明特征能有效区分该类别；若不同颜色的点混杂，说明特征对这些类别的区分能力较弱。
    # 可用于评估两种模态的特征是否保留了类别信息，聚集性越好，类别区分能力越强
    unique_labels = np.unique(labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = all_labels == label
        ax2.scatter(reduced_features[mask, 0], reduced_features[mask, 1], 
                   c=[color], label=labels_Cn[label], alpha=0.6)
    ax2.set_title(f'{title} - By Class')
    ax2.legend()
    
    plt.tight_layout()
    
    return fig_vmc

def visualize_similarity_matrices(features1, features2):
    """
    可视化模态内和模态间的相似度矩阵，量化不同模态特征的相似性结构。
    1. 模态 1内部相似度
    图中存在多个明显的红色 / 橙色高相似度区块（相似度接近 1.0），且区块边界清晰。
    说明 1D 时序特征的内部一致性强：同类样本的 1D 特征相似度高，能形成紧密的 “特征集群”，对不同路面类别的区分能力较显著。
    2. 模态 2内部相似度
    整体以蓝色 / 浅蓝色为主（相似度多在 0.6 以下），红色高相似度区块极少且模糊。
    说明 2D 图像特征的内部一致性较弱：同类样本的 2D 特征相似度低，特征分布更分散，对类别区分的能力不如 1D 时序特征。
    3. 跨模态相似度
    仅左上角小区域存在少量红色 / 橙色高相似度区域，其余大部分区域为蓝色（低相似度）。
    说明两种模态的特征整体关联度低，但存在局部一致性：多数样本的 1D 和 2D 特征描述差异较大（互补性强），仅少数样本的两种模态特征表现出相似性。
    核心结论：1D 时序特征在同类样本一致性和类别区分上表现更优，2D 图像特征区分能力较弱但与 1D 特征互补性强，两种模态的融合有望结合各自优势，提升模型对路面类别的分类性能。
    """
    # 计算余弦相似度
    def cosine_similarity_matrix(features):
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        normalized = features / norms
        return np.dot(normalized, normalized.T)
    
    # 计算三个相似度矩阵
    sim_within_1 = cosine_similarity_matrix(features1)
    sim_within_2 = cosine_similarity_matrix(features2)
    sim_between = cosine_similarity_matrix(np.vstack([features1, features2]))
    
    # 绘制热力图
    fig_vsm, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im1 = axes[0].imshow(sim_within_1, cmap='RdYlBu', vmax=1, vmin=0)
    axes[0].set_title('Within Modality 1 Similarity')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(sim_within_2, cmap='RdYlBu', vmax=1, vmin=0)
    axes[1].set_title('Within Modality 2 Similarity')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(sim_between, cmap='RdYlBu', vmax=1, vmin=0)
    axes[2].set_title('Cross Modality Similarity')
    axes[2].axhline(y=len(features1)-0.5, color='black', linewidth=2)
    axes[2].axvline(x=len(features1)-0.5, color='black', linewidth=2)
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    return fig_vsm

def visualize_attention_weights(attention_weights, modality1_names, modality2_names):
    """
    可视化跨模态注意力权重
    """
    plt.figure(figsize=(12, 8))
    
    # 假设attention_weights形状为 [batch, seq_len1, seq_len2]
    # 取第一个样本的平均注意力权重
    avg_attn = attention_weights[0].mean(dim=0).cpu().numpy()
    
    plt.imshow(avg_attn, cmap='viridis', aspect='auto')
    plt.colorbar(label='Attention Weight')
    
    plt.xlabel('Modality 2 Features')
    plt.ylabel('Modality 1 Features')
    plt.title('Cross-Modal Attention Weights')
    
    # 添加特征标签（如果提供）
    if modality1_names is not None:
        plt.yticks(range(len(modality1_names)), modality1_names)
    if modality2_names is not None:
        plt.xticks(range(len(modality2_names)), modality2_names, rotation=45)
    
    plt.tight_layout()
    plt.show()

def visualize_modality_comparison(fold_f1score_array, results_dir, model_name, valid_dataset, labels_Cn):
    """
    找到F1分数最高的折，提取并对比CNN_1D和CNN_2D模型的特征分布
    
    参数:
        fold_f1score_array: 各折的F1分数数组
        results_dir: 结果保存目录
        model_name: 模型名称
        valid_dataset: 验证数据集
        labels_Cn: 类别中文名称列表
    """
    # 找到F1分数最高的折
    best_fold_idx = np.argmax(fold_f1score_array)
    best_fold = best_fold_idx + 1  # 折号从1开始计数
    print(f"\n选择F1分数最高的折 (第{best_fold}折, F1={fold_f1score_array[best_fold_idx]:.4f}) 进行模态特征对比可视化...")
    
    # 创建可视化结果保存目录
    viz_save_dir = os.path.join(results_dir, f'best_fold_{best_fold}_comparison-similarity')
    os.makedirs(viz_save_dir, exist_ok=True)
    
    # 加载最佳折的模型权重
    model_saved_path = os.path.join(results_dir, 'model_saved', f'fold_{best_fold}.pth')
    
    # 根据model_name初始化对应模型
    if 'LSTM' in model_name:MAMIL1D = LSTM_Model(in_channels = 1) .cuda() 
    elif 'CNN_1D' in model_name: MAMIL1D = CNN_1D(in_channels = 1) .cuda()
    elif 'CNN_SMNK1D' in model_name: MAMIL1D = CNN_SMNK1D(in_channels = 1) .cuda()
    elif 'CNN_PMNK1D' in model_name: MAMIL1D = CNN_PMNK1D(in_channels = 1) .cuda()
    elif 'CNN_SPMNK1D' in model_name: MAMIL1D = CNN_SPMNK1D(in_channels = 1) .cuda()
    elif 'CNN_SBPMNK1D' in model_name: MAMIL1D = CNN_SBPMNK1D(in_channels = 1) .cuda()
    else: MAMIL1D = None

    if 'CNN_2D' in model_name: MAMIL2D = CNN_2D(in_channels = 1).cuda()
    elif 'ResNet_2D' in model_name: MAMIL2D = ResNet_2D(in_channels = 1).cuda()
    elif 'HMCNN2D' in model_name: MAMIL2D = HMCNN_2D(in_channels = 1).cuda()
    else: MAMIL2D = None
    
    # 加载模型权重
    checkpoint = torch.load(model_saved_path)
    if MAMIL1D is not None and 'MAMIL1D' in checkpoint and checkpoint['MAMIL1D'] is not None:
        MAMIL1D.load_state_dict(checkpoint['MAMIL1D'])
        MAMIL1D.eval()
    if MAMIL2D is not None and 'MAMIL2D' in checkpoint and checkpoint['MAMIL2D'] is not None:
        MAMIL2D.load_state_dict(checkpoint['MAMIL2D'])
        MAMIL2D.eval()
    
    # 提取特征
    features_1d = []
    features_2d = []
    all_labels = []
    
    # 从valid_dataset创建DataLoader
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=256,  # 与训练时保持一致或适当调整
        shuffle=False,   # 评估时不需要打乱
        num_workers=4    # 根据系统情况调整
    )

    with torch.no_grad():
        for data_1D, data_2D, target in valid_loader:
            if MAMIL1D:data_1D = data_1D.cuda()
            if MAMIL2D:data_2D = data_2D.cuda()
            target = target.cuda()
            
            # 提取1D特征
            if MAMIL1D is not None:
                feat_1d = MAMIL1D(data_1D)
                features_1d.append(feat_1d.cpu().numpy())
            
            # 提取2D特征
            if MAMIL2D is not None:
                feat_2d = MAMIL2D(data_2D)
                features_2d.append(feat_2d.cpu().numpy())
            
            all_labels.append(target.cpu().numpy())
    
    # 合并特征和标签
    features_1d = np.concatenate(features_1d, axis=0) if features_1d else None
    features_2d = np.concatenate(features_2d, axis=0) if features_2d else None
    all_labels = np.concatenate(all_labels, axis=0)
    
    # 确保有两种模态的特征
    if len(features_1d) > 0 and len(features_2d) > 0: 
        print("开始模态特征对比可视化...")

        # 使用visualize_modality_comparison函数可视化两个模态的特征分布
        fig_vmc = tSNE_visualize_modality_comparison(
            features1=features_1d,
            features2=features_2d,
            labels=all_labels,
            labels_Cn=labels_Cn,
            title=f'第{best_fold}折 (F1={fold_f1score_array[best_fold_idx]:.4f}) - CNN_1D vs CNN_2D特征分布对比'
        )

        # 保存可视化结果
        plt.savefig(os.path.join(viz_save_dir, f'modality_comparison_fold_{best_fold}.png'), 
                    dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig_vmc)

        # 使用visualize_similarity_matrices函数可视化模态特征对比
        fig_vsm = visualize_similarity_matrices( 
            features1=features_1d,
            features2=features_2d
        )
        # 保存可视化结果
        plt.savefig(os.path.join(viz_save_dir, f'similarity_matrices_fold_{best_fold}.png'), 
                    dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig_vsm)
        
        print(f"模态特征对比、相似度矩阵可视化结果已保存至: {viz_save_dir}")
    else:
        print("警告：无法提取两种模态的特征进行对比")

def visualize_branch_weights(analysis_file, output_dir="visualize-results"):
    """可视化分支权重分析结果"""
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        weight_analysis = json.load(f)
    
    # 创建输出目录
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为DataFrame便于分析
    data_rows = []
    for fold_key, fold_analysis in weight_analysis.items():
        for road_type, branch_analysis in fold_analysis.items():
            for branch_name, stats in branch_analysis.items():
                data_rows.append({
                    'fold': fold_key,
                    'road_type': road_type,
                    'branch': branch_name,
                    'mean_weight': stats['mean_weight'],
                    'count': stats['count']
                })
    
    df = pd.DataFrame(data_rows)
    
    # 绘制权重分布图
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='road_type', y='mean_weight', hue='branch')
    plt.title('各分支在不同路面类型下的平均权重')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/branch_weights_by_road_type.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 重点关注FreRA分支
    frera_data = df[df['branch'] == '1D_fre']
    if not frera_data.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=frera_data, x='road_type', y='mean_weight')
        plt.title('FreRA分支在不同路面类型下的权重')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/frera_weights_by_road_type.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"权重分析图表已保存到 {output_dir} 目录")

class RoadFeatureHeatmapAnalyzer:
    """
    路面-特征响应热力图分析器
    通过量化分析一维时序特征和二维图像特征在不同路面类型上的响应强度，对比两种模态的表现差异，为跨模态融合提供依据。
    """
    
    def __init__(self):
        self.road_types = ['大碎石路面', '左右起伏路', '普通平整水泥路', '普通砖石路', '条纹型路面', '窨井盖路', '馒头圆饼路']
        self.force_types = ['Fx']
        
    def load_data(self, force_type='Fx', window_config='ori-64-16'):
        """加载车轮力数据和标签"""
        data_path = os.path.join('data', '不同窗口尺寸和重叠率的车轮力数据', window_config) 
        
        # 加载数据
        data_file = os.path.join(data_path, f'data_{force_type}.mat')
        label_file = os.path.join(data_path, f'label_{force_type}.mat')
        
        data = sio.loadmat(data_file)[f'data_{force_type}']
        labels = sio.loadmat(label_file)[f'label_{force_type}'].flatten()
            
        print(f"加载数据成功: {data.shape}, 标签: {labels.shape}")

        return data, labels

    def extract_1d_features(self, data):
        """提取一维时序特征"""
        print("提取一维时序特征...")
        
        # 基本统计特征
        features_1d = []
        for i in range(data.shape[0]):
            sample = data[i]
            
            # 时域特征
            time_features = [
                np.mean(sample),           # 均值
                np.std(sample),            # 标准差
                np.max(sample),            # 最大值
                np.min(sample),            # 最小值
                np.ptp(sample),            # 峰峰值
                np.median(sample),         # 中位数
                np.var(sample),            # 方差
                np.sqrt(np.mean(sample**2)), # RMS
                np.mean(np.abs(sample - np.mean(sample))), # 平均绝对偏差
            ]
            
            # 频域特征 (通过FFT)
            fft_vals = np.abs(np.fft.fft(sample))
            freq_features = [
                np.mean(fft_vals),         # 频域均值
                np.std(fft_vals),          # 频域标准差
                np.max(fft_vals),          # 频域最大值
                np.sum(fft_vals[:len(fft_vals)//2]), # 低频能量
                np.sum(fft_vals[len(fft_vals)//2:]), # 高频能量
            ]
            
            # 峰值特征
            peaks, _ = self._find_peaks(sample)
            peak_features = [
                len(peaks),                 # 峰值数量
                np.mean(peaks) if len(peaks) > 0 else 0, # 平均峰值
                np.std(peaks) if len(peaks) > 0 else 0,  # 峰值标准差
            ]
            
            all_features = time_features + freq_features + peak_features
            features_1d.append(all_features)
        
        features_1d = np.array(features_1d)
        print(f"一维特征形状: {features_1d.shape}")
       # 添加特征名称
        feature_names = [
            # 时域特征
            '时域均值', '时域标准差', '时域最大值', '时域最小值', '时域峰峰值', 
            '时域中位数', '时域方差', '时域RMS', '时域平均绝对偏差',
            # 频域特征
            '频域均值', '频域标准差', '频域最大值', '低频能量', '高频能量',
            # 峰值特征
            '峰值数量', '平均峰值', '峰值标准差'
        ]
        
        return features_1d, feature_names
    
    def extract_2d_features(self, data):
        """提取二维图像特征（基于现有的二维化方法）"""
        print("提取二维图像特征...")

        # 转换为适合二维化的格式
        data_tensor = torch.tensor(data).unsqueeze(1)  # [B, 1, L]

        # 生成二维特征图像
        image_features = wavetoimagetransf(data_tensor.numpy(), [0, 1], save=False)
        
        # 提取图像统计特征
        features_2d = []
        for i in range(image_features.shape[0]):
            sample_features = []
            for channel in range(image_features.shape[1]):
                img = image_features[i, channel]
                
                # 图像纹理特征
                img_features = [
                    np.mean(img),           # 平均亮度
                    np.std(img),            # 对比度
                    np.max(img),            # 最大亮度
                    np.min(img),            # 最小亮度
                    np.var(img),            # 方差
                    self._image_entropy(img), # 熵（纹理复杂度）
                    self._image_energy(img),  # 能量
                ]
                sample_features.extend(img_features)
            
            features_2d.append(sample_features)
        
        features_2d = np.array(features_2d)
        print(f"二维特征形状: {features_2d.shape}")
        # 添加特征名称
        feature_2d_names = [
            # GAF通道
            'GAF平均亮度', 'GAF对比度', 'GAF最大亮度', 'GAF最小亮度', 
            'GAF方差', 'GAF熵', 'GAF能量',
            # STFT通道
            'STFT平均亮度', 'STFT对比度', 'STFT最大亮度', 'STFT最小亮度', 
            'STFT方差', 'STFT熵', 'STFT能量',
        ]
        return features_2d, feature_2d_names
    
    def _find_peaks(self, signal, height_threshold=0.1):
        """简单的峰值检测"""
        peaks = []
        for i in range(1, len(signal)-1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                if signal[i] > height_threshold * np.max(signal):
                    peaks.append(signal[i])
        return np.array(peaks), []
    
    def _image_entropy(self, image):
        """计算图像熵"""
        hist, _ = np.histogram(image.flatten(), bins=256, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))
    
    def _image_energy(self, image):
        """计算图像能量"""
        return np.sum(image**2)
    
    def compute_feature_response(self, features, labels):
        """计算特征在不同路面类型上的响应强度"""
        print("计算特征响应强度...")
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # 按路面类型分组计算特征均值
        response_matrix = []
        
        for road_type in range(len(self.road_types)):
            mask = labels == road_type
            if np.sum(mask) > 0:
                road_features = features_scaled[mask]
                mean_response = np.mean(road_features, axis=0)
                response_matrix.append(mean_response)
            else:
                response_matrix.append(np.zeros(features.shape[1]))
        
        response_matrix = np.array(response_matrix)
        return response_matrix
    
    def plot_heatmap(self, response_matrix_1d, response_matrix_2d, feature_names_1d, feature_names_2d, 
                    save_path='visualize-results/road_feature_heatmaps.png'):
        """绘制路面-特征响应热力图"""
        print("绘制热力图...")
        print(f"一维响应矩阵形状: {response_matrix_1d.shape}")
        print(f"二维响应矩阵形状: {response_matrix_2d.shape}")
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        # 一维特征热力图
        sns.heatmap(response_matrix_1d.T, 
                   xticklabels=self.road_types,
                   yticklabels=feature_names_1d,
                   cmap='RdBu_r', center=0,
                   ax=axes[0, 0], cbar_kws={'label': '响应强度'})
        axes[0, 0].set_title('一维时序特征 - 路面响应热力图', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('路面类型')
        axes[0, 0].set_ylabel('时序特征')
        
        # 二维特征热力图
        sns.heatmap(response_matrix_2d.T,
                   xticklabels=self.road_types,
                   yticklabels=feature_names_2d,
                   cmap='RdBu_r', center=0,
                   ax=axes[0, 1], cbar_kws={'label': '响应强度'})
        axes[0, 1].set_title('二维图像特征 - 路面响应热力图', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('路面类型')
        axes[0, 1].set_ylabel('图像特征')
        
        # 使用主对角线上的特征强度比较或其他可视化方式
        # 我们可以可视化每个路面类型的整体响应差异
        road_diff = np.mean(response_matrix_2d, axis=1) - np.mean(response_matrix_1d, axis=1)
        
        # 创建一个比较热图
        sns.heatmap(np.expand_dims(road_diff, axis=0),
                   xticklabels=self.road_types,
                   yticklabels=['模态响应差异'],
                   cmap='RdBu_r', center=0,
                   ax=axes[1, 0], cbar_kws={'label': '响应差异'})
        axes[1, 0].set_title('路面类型的模态响应差异 (二维 - 一维)', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('路面类型')
        axes[1, 0].set_ylabel('差异类型')
        
        # 响应强度统计
        road_response_1d = np.mean(np.abs(response_matrix_1d), axis=1)
        road_response_2d = np.mean(np.abs(response_matrix_2d), axis=1)
        
        x = np.arange(len(self.road_types))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, road_response_1d, width, label='一维特征', alpha=0.7)
        axes[1, 1].bar(x + width/2, road_response_2d, width, label='二维特征', alpha=0.7)
        axes[1, 1].set_xlabel('路面类型')
        axes[1, 1].set_ylabel('平均响应强度')
        axes[1, 1].set_title('不同路面类型的平均响应强度对比', fontsize=14, fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(self.road_types, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"热力图已保存到: {save_path}")
        plt.close()
    
    def analyze_feature_regions(self, response_matrix_1d, response_matrix_2d):
        """分析特征响应集中区域"""
        print("分析特征响应集中区域...")
        
        # 找出每种路面类型响应最强的特征
        strong_features_1d = {}
        strong_features_2d = {}
        
        for i, road_type in enumerate(self.road_types):
            # 一维特征最强响应
            idx_1d = np.argmax(np.abs(response_matrix_1d[i]))
            strength_1d = response_matrix_1d[i, idx_1d]
            
            # 二维特征最强响应  
            idx_2d = np.argmax(np.abs(response_matrix_2d[i]))
            strength_2d = response_matrix_2d[i, idx_2d]
            
            strong_features_1d[road_type] = (idx_1d, strength_1d)
            strong_features_2d[road_type] = (idx_2d, strength_2d)
        
        # 打印分析结果
        print("\n=== 特征响应集中区域分析 ===")
        print("一维时序特征响应集中区域:")
        for road_type, (feature_idx, strength) in strong_features_1d.items():
            feature_region = self._get_1d_feature_region(feature_idx)
            print(f"  {road_type}: 特征F{feature_idx+1} ({feature_region}), 响应强度: {strength:.3f}")
        
        print("\n二维图像特征响应集中区域:")
        for road_type, (feature_idx, strength) in strong_features_2d.items():
            feature_region = self._get_2d_feature_region(feature_idx)
            print(f"  {road_type}: 特征F{feature_idx+1} ({feature_region}), 响应强度: {strength:.3f}")
    
    def _get_1d_feature_region(self, feature_idx):
        """获取一维特征对应的区域描述"""
        if feature_idx < 9:
            return "时域统计区"
        elif feature_idx < 14:
            return "频域能量区" 
        else:
            return "时域峰值区"
    
    def _get_2d_feature_region(self, feature_idx):
        """获取二维特征对应的区域描述"""
        if feature_idx < 7:
            return "GAF纹理区"
        else:
            return "STFT频域区"
    
    def run_analysis(self, force_type='Fx', window_config='ori-64-16'):
        """运行完整的特征响应分析"""
        '''响应强度高通常意味着对应模态对特定路面类型的区分性信息更敏感—— 即该模态能更显著地捕捉
        到该路面类型的独特特征（如结构、纹理、时序模式等）这些特征可能是区分该路面类型与其他类型的关键依据
        '''
        print(f"开始分析 {force_type} 数据的特征响应...")
        
        # 加载数据
        data, labels = self.load_data(force_type, window_config)
        
        # 提取特征
        features_1d, feature_names_1d = self.extract_1d_features(data)
        features_2d, feature_names_2d = self.extract_2d_features(data)
        
        # 计算响应矩阵
        response_1d = self.compute_feature_response(features_1d, labels)
        response_2d = self.compute_feature_response(features_2d, labels)
        
        # 绘制热力图
        save_path = f'visualize-results/road_feature_heatmaps_{force_type}.png'
        self.plot_heatmap(response_1d, response_2d, feature_names_1d, feature_names_2d, save_path)
        '''
        正值，红色：表示该特征在特定路面类型上的取值高于所有样本的平均水平（相对于标准化后的平均值）
        负值，蓝色：表示该特征在特定路面类型上的取值低于所有样本的平均水平（相对于标准化后的平均值）'''
        
        # 分析特征响应集中区域
        self.analyze_feature_regions(response_1d, response_2d)
        
        return response_1d, response_2d