import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

class DataAugmenter:
    def __init__(self): pass
    def __call__(self, data, labels, methods=['time_warp'], num_augmented=2, return_merged=False, visualize=False):
        """
        数据增强主函数
        :param data: 输入数据
        :param labels: 输入标签
        :param methods: 增强方法列表 ['amplitude', 'noise', 'time_warp', 'reverse', 'shift']
        :param num_augmented: 每个样本增强的次数
        -----------------------------------------------------------------------------------------------------------
        假设有一个原始样本x，我们选择了两种增强方法：添加噪声和振幅缩放，并且设置num_augmented=2
        那么我们会进行以下操作：
        第一次增强：对x先添加噪声，然后进行振幅缩放，得到增强样本x1
        第二次增强：再次对x先添加噪声，然后进行振幅缩放（注意：每次增强的噪声和缩放因子都是随机独立生成的），得到增强样本x2
        因此，每个原始样本会产生2个增强样本。
        如果return_merged=True，那么返回的数据集包括：
        所有原始样本 + 所有增强样本（每个原始样本对应num_augmented个增强样本）
        -----------------------------------------------------------------------------------------------------------
        :param return_merged: 是否返回合并后的数据
        :param visualize: 是否可视化增强后的样本
        :param method_kwargs: 字典，键为增强方法名，值为该方法的参数（如{'time_warp': {'warp_factor': 0.3}}）
        :return: 增强后的数据和标签
        """
        augmented_data = []
        augmented_labels = []
        
        # 保留原始数据
        if return_merged:            
            augmented_data.extend(data.clone())
            augmented_labels.extend(labels.clone())

        # 对每个样本应用增强
        for i in range(len(data)):
            original_signal = data[i].clone()     
            original_label = labels[i]

            for _ in range(num_augmented):
                augmented_signal = original_signal.clone()              
                # 应用所有选定的增强方法
                for method in methods:
                    if method == 'amplitude':
                        augmented_signal = self.amplitude_augment(augmented_signal)
                    elif method == 'noise':
                        augmented_signal = self.noise_augment(augmented_signal)
                    elif method == 'time_warp':
                        augmented_signal = self.time_warp_augment(augmented_signal)
                    elif method == 'reverse':
                        augmented_signal = self.reverse_augment(augmented_signal)
                    elif method == 'shift':
                        augmented_signal = self.shift_augment(augmented_signal)
                    # 可以继续添加其他增强方法
                
                augmented_data.append(augmented_signal)
                augmented_labels.append(original_label)

        result_data = torch.stack(augmented_data)
        result_labels = torch.tensor(augmented_labels)

        # 可视化（可选）
        if visualize:
            self.visualize_augmentation(data, result_data, methods)

        # 返回与输入相同类型的数据            
        return result_data, result_labels
    
    def amplitude_augment(self, signal, scale_range=(0.8, 1.2)):
        """振幅缩放增强"""
        scale_factor = torch.FloatTensor(1).uniform_(scale_range[0], scale_range[1]).item()
        return signal * scale_factor
    
    def noise_augment(self, signal, noise_level=0.1):
        """添加高斯噪声"""
        noise = torch.randn_like(signal) * noise_level * signal.std()
        return signal + noise

    def time_warp_augment(self, signal, warp_factor=0.1):
        """时间扭曲增强"""
        
        # 对于形状为(1, 64)的数据，调整维度处理方式
        if signal.ndim == 2 and signal.shape[0] == 1:
            # 提取时间序列
            seq_len = signal.shape[1]
            
            # 创建扭曲的时间点
            warp_amount = torch.normal(1, warp_factor, (seq_len,))
            warp_amount = torch.cumsum(warp_amount, dim=0)
            warp_amount = (warp_amount - warp_amount[0]) / (warp_amount[-1] - warp_amount[0])
            
            # 重新实现插值过程，确保正确处理维度
            warped_indices = (warp_amount * (seq_len - 1)).long()
            # 确保索引不超出范围
            warped_indices = torch.clamp(warped_indices, 0, seq_len - 1)
            
            # 直接使用索引选择实现扭曲
            warped_signal = signal[:, warped_indices]
            
            return warped_signal
        else:
            # 原来的实现，用于其他维度的数据
            n = len(signal)
            warp_amount = torch.normal(1, warp_factor, (n,))
            warp_amount = torch.cumsum(warp_amount, dim=0)
            warp_amount = (warp_amount - warp_amount[0]) / (warp_amount[-1] - warp_amount[0])
            
            try:
                warped_signal = F.interpolate(
                    signal.transpose(0, 1).unsqueeze(0),
                    size=n,
                    mode='linear',
                    align_corners=True
                ).squeeze(0).transpose(0, 1)
                return warped_signal
            except Exception as e:
                # 如果原来的实现失败，返回稍微修改过的信号以确保增强有效
                print(f"时间扭曲失败，使用备用方法: {e}")
                # 添加微小的随机偏移来确保数据有所不同
                jitter = torch.randn_like(signal) * 0.1 * signal.std()
                return signal + jitter

    def reverse_augment(self, signal):
        """时间反转增强"""
        return signal.flip(1)

    def shift_augment(self, signal, shift_range=(-0.1, 0.1)):
        """平移增强"""
        shift_factor = torch.FloatTensor(1).uniform_(shift_range[0], shift_range[1]).item()
        return signal + shift_factor
    
    def visualize_augmentation(self, data, augmented_data, methods):
        """可视化增强效果"""
        # 获取第一条原始数据
        original_sample = data[0].cpu().numpy()
        
        # 关键修改：从增强数据集中正确获取第一条增强数据
        # 由于数据结构是: [原始数据1, 原始数据2, ..., 增强数据1, 增强数据2, ...]
        # 所以第一条增强数据应该是在原始数据之后的第一个位置
        augmented_index = len(data)  # 跳过所有原始数据
        augmented_sample = augmented_data[augmented_index].cpu().numpy()
        
        # 检查并修正数据维度
        if original_sample.ndim == 2 and original_sample.shape[0] < original_sample.shape[1]:
            original_sample = original_sample.T
            augmented_sample = augmented_sample.T
        elif original_sample.ndim == 1:
            original_sample = original_sample.reshape(-1, 1)
            augmented_sample = augmented_sample.reshape(-1, 1)
        
        # 提取第一个通道的数据
        original_first_channel = original_sample[:, 0]
        augmented_first_channel = augmented_sample[:, 0]
        
        # 创建并配置图表
        plt.figure(figsize=(12, 6))
        plt.plot(original_first_channel, label='原始数据', color='blue', linewidth=2.5)
        plt.plot(augmented_first_channel, label='增强后数据', color='red', linewidth=1.5, linestyle='--')
        
        plt.legend(fontsize=12)
        plt.title(f'数据增强前后对比: {", ".join(methods)}', fontsize=14, pad=10)
        plt.grid(True, alpha=0.3)
        plt.xlabel('时序长度', fontsize=11)
        plt.ylabel('信号值', fontsize=11)
        
        plt.tight_layout()
        plt.savefig('visualize-results/augmentation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
