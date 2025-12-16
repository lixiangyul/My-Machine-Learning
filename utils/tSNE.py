import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch, os
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler

# 字体配置
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']    # 使用文泉驿正黑
plt.rcParams['axes.unicode_minus'] = False                 # 修复负号显示问题

def extract_features(model, data_loader, device, model_type='MAMIL1D'):
    """
    从模型中提取特征
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 运行设备
        model_type: 模型类型 ('MAMIL1D', 'MAMIL2D', 'fusionmodel')
    
    Returns:
        features: 提取的特征
        labels: 对应的标签
    """
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            # 根据数据格式确定如何获取数据和标签
            if len(batch_data) == 3:
                # 1D数据, 2D数据, 标签
                batch_data_1D, batch_data_2D, batch_labels = batch_data
                batch_data_1D = batch_data_1D.to(device)
                batch_data_2D = batch_data_2D.to(device)
                batch_labels = batch_labels.to(device)
            else:
                # 其他情况，可能只有一种数据类型
                if model_type == 'MAMIL1D' and len(batch_data) == 2:
                    batch_data_1D, batch_labels = batch_data
                    batch_data_1D = batch_data_1D.to(device)
                    batch_labels = batch_labels.to(device)
                elif model_type == 'MAMIL2D' and len(batch_data) == 2:
                    batch_data_2D, batch_labels = batch_data
                    batch_data_2D = batch_data_2D.to(device)
                    batch_labels = batch_labels.to(device)

            # 根据模型类型选择不同的特征提取方式
            if model_type == 'MAMIL1D':
                # 提取1D特征
                if isinstance(model, torch.nn.Sequential):
                    outputs = model[0](batch_data_1D)
                else:
                    outputs = model(batch_data_1D)
            elif model_type == 'MAMIL2D':
                # 提取2D特征
                if isinstance(model, torch.nn.Sequential):
                    outputs = model[1](batch_data_2D)
                else:
                    outputs = model(batch_data_2D)
            elif model_type == 'fusionmodel':
                # 提取融合特征（注意力机制融合之后的特征）
                if isinstance(model, torch.nn.Sequential) and len(model) >= 3:
                    # 假设model[0]是1D特征提取器，model[1]是2D特征提取器，model[2]是注意力机制
                    feature_1d = model[0](batch_data_1D)
                    feature_2d = model[1](batch_data_2D)
                    # 获取注意力机制融合后的特征
                    fusion_feature, _ = model[2](feature_1d, feature_2d)
                    outputs = fusion_feature
                else:
                    # 如果模型结构不同，尝试直接调用
                    outputs = model(batch_data_1D, batch_data_2D)
            else:
                raise ValueError(f"未知的模型类型: {model_type}")

            # 确保输出是特征向量
            if outputs.dim() > 2:
                outputs = outputs.view(outputs.size(0), -1)
            
            features.append(outputs.cpu().numpy())
            labels.append(batch_labels.cpu().numpy())
    
    # 合并所有批次的特征和标签
    features = np.vstack(features)
    labels = np.hstack(labels)
    return features, labels

def visualize_wheel_force_features(model, data_loader, device, output_dir, n_components=2,
                                   model_name='CNN_1D', class_names=None, k=5, num_force=4, model_type='MAMIL1D'):
    """
    使用t-SNE可视化车轮力特征
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 运行设备
        output_dir: 结果保存目录
        n_components: 降维后的维度，2或3
        model_name: 模型名称
        class_names: 类别名称列表
        k: 当前折数
        num_force: 使用的力的数量
        model_type: 模型类型 ('MAMIL1D', 'MAMIL2D', 'fusionmodel')
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在提取{model_type}特征...")
    # 从模型中提取特征
    features, labels = extract_features(model, data_loader, device, model_type)
    
    print(f"特征提取完成，样本数: {features.shape[0]}, 特征维度: {features.shape[1]}")
    
    # 执行t-SNE降维
    print(f"执行t-SNE降维，将{features.shape[1]}维特征降维到{n_components}维...")
    
    # 特征标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    tsne = TSNE(
        n_components=n_components,
        perplexity=20,
        learning_rate=200.0,
        min_grad_norm=1e-07,
        metric='euclidean',
        init='pca',
        verbose=1,
        random_state=42
    )
    
    # 应用t-SNE
    features_tsne = tsne.fit_transform(features_scaled)
    
    # 可视化结果
    plot_tsne_results(features_tsne, labels, output_dir, n_components, model_name, 
                     class_names, k, num_force, model_type)
    
    return features_tsne, labels

def plot_tsne_results(features_tsne, labels_np, output_dir, n_components, model_name,
                      class_names, k, num_force, model_type):
    """
    绘制t-SNE结果
    
    Args:
        features_tsne: t-SNE降维后的特征
        labels_np: 对应的标签
        output_dir: 结果保存目录
        n_components: 降维后的维度，2或3
        model_name: 模型名称
        class_names: 类别名称列表
        k: 当前折数
        num_force: 使用的力的数量
        model_type: 模型类型
    """
    # 使用更鲜明的颜色映射 - 使用tab20确保颜色差异更大
    num_classes = len(np.unique(labels_np))
    colors = cm.tab20(np.linspace(0, 1, num_classes))
    
    plt.figure(figsize=(10, 8))
    
    if n_components == 2:
        # 2D可视化
        for i in range(num_classes):
            indices = labels_np == i
            if np.sum(indices) > 0:  # 确保该类别有样本
                plt.scatter(
                    features_tsne[indices, 0],
                    features_tsne[indices, 1],
                    c=[colors[i]],
                    label=class_names[i] if class_names else f'类别{i}',
                    alpha=0.7,
                    s=50
                )
        
        plt.xlabel('t-SNE 维度1')
        plt.ylabel('t-SNE 维度2')
    
    elif n_components == 3:
        # 3D可视化
        ax = plt.figure(figsize=(10, 8)).add_subplot(111, projection='3d')
        
        for i in range(num_classes):
            indices = labels_np == i
            if np.sum(indices) > 0:  # 确保该类别有样本
                ax.scatter(
                    features_tsne[indices, 0],
                    features_tsne[indices, 1],
                    features_tsne[indices, 2],
                    c=[colors[i]],
                    label=class_names[i] if class_names else f'类别{i}',
                    alpha=0.7,
                    s=50
                )
        
        ax.set_xlabel('t-SNE 维度1')
        ax.set_ylabel('t-SNE 维度2')
        ax.set_zlabel('t-SNE 维度3')
    
    plt.title(f't-SNE可视化 - {model_name} - 第{k+1}折', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存结果
    filename = f'tsne_{model_name}_fold_{k+1}_{n_components}D.png'
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"t-SNE可视化结果已保存至: {filepath}")
