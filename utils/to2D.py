import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pywt  # 实现CWT
from matplotlib.gridspec import GridSpec
from pyts.image import GramianAngularField
from sklearn.metrics.pairwise import pairwise_distances
from scipy.signal import stft, hilbert
from scipy.ndimage import zoom
from PyEMD import EMD  # 用于HHT的EMD分解 pip install EMD-signal

def wavetoimagetransf(data, selected_channels, save=False):
    """生成多通道特征图像，支持GAF、STFT、CWT、HHT、MTF、RP特征组合"""
    # 确保selected_channels是列表
    if isinstance(selected_channels, int):
        selected_channels = [selected_channels]
    
    data = data.squeeze()  # [样本数, 1, L]->[样本数, L]
    
    # 定义所有处理函数
    def gaf_process(data):
        gaf = GramianAngularField(image_size=data.shape[0], method='summation')
        x_gaf = gaf.fit_transform(data.reshape(1, -1))
        x_gaf = x_gaf.squeeze()
        return (x_gaf + 1) / 2  # 归一化到[0,1]
    
    def stft_process(data):
        stft_feature = stft_transform(data)
        return (stft_feature - stft_feature.min()) / (stft_feature.max() - stft_feature.min() + 1e-8)
    
    def cwt_process(data):
        cwt_feature = cwt_transform(data)
        return (cwt_feature - cwt_feature.min()) / (cwt_feature.max() - cwt_feature.min() + 1e-8)
    
    def hht_process(data):
        hht_feature = hht_transform(data)
        return (hht_feature - hht_feature.min()) / (hht_feature.max() - hht_feature.min() + 1e-8)
    
    def mtf_process(data):
        return get_mtf(data)
    
    def rp_process(data):
        return recurrence_plot(data.reshape(-1, 1), steps=10)
    
    # 映射通道编号到处理函数
    process_funcs = {
        0: gaf_process,
        1: stft_process, 
        2: cwt_process,
        3: hht_process,
        4: mtf_process,
        5: rp_process
    }
    
    # 处理所有样本和所有选中的通道
    all_results = []
    for i in range(data.shape[0]):
        sample_results = []
        for channel in selected_channels:
            if channel in process_funcs:
                feature = process_funcs[channel](data[i])
                sample_results.append(feature)
        # 将单个样本的多通道特征堆叠起来
        sample_stack = np.stack(sample_results, axis=0)  # [通道数, L, L]
        all_results.append(sample_stack)
    
    results = np.array(all_results)  # 形状: [样本数, 通道数, L, L]
    
    # 可视化第一个样本的特征图
    if save:
        visualize_multichannel(results[:1], selected_channels)

    return results

def stft_transform(data: np.ndarray) -> np.ndarray:
    """STFT变换"""
    # 设置STFT参数：窗口长度16，重叠8
    nperseg = 16 # nperseg = len(data)这里选用更小的窗口长度更优
    noverlap = 8 # noverlap = 16
    target_size = 64
    _, _, Zxx = stft(data, nperseg=nperseg, noverlap=noverlap, return_onesided=True)
    stft_mag = np.abs(Zxx)  # 获取幅度谱，形状为(freq_bins, time_frames)
    
    # 计算当前时频矩阵的形状
    freq_bins, time_frames = stft_mag.shape
    # 缩放至目标尺寸
    zoom_factor = (target_size / freq_bins, target_size / time_frames)
    stft_scaled = zoom(stft_mag, zoom_factor, order=1)  # 线性插值
    
    return stft_scaled

def cwt_transform(data: np.ndarray) -> np.ndarray:
    """连续小波变换(CWT)"""
    data = data.cpu().numpy()
    target_size = data.shape[0]
    wavelet = 'morl'  # 选择小波函数'morl', 'cmor', 'gaus', 'mexh'
    
    min_scale = 1            # 更合理的尺度范围设置，避免尺度范围过大导致计算复杂和特征稀疏
    max_scale = len(data)    # 限制最大尺度，避免过大
    scales = np.arange(min_scale, max_scale)
    
    # 使用pywt进行CWT计算。注意：pywt.cwt返回的是(coeffs, frequencies)
    coeffs, _ = pywt.cwt(data, scales=scales, wavelet=wavelet, sampling_period=1.0)
    cwt_mag = np.abs(coeffs) # 取绝对值获取幅度谱
    # 计算缩放因子，保持宽高比或拉伸到目标尺寸，使用双线性插值(order=1)保持平滑度
    original_height, original_width = cwt_mag.shape
    zoom_factor = (target_size / original_height, target_size / original_width)
    cwt_scaled = zoom(cwt_mag, zoom_factor, order=1)    # 应用缩放
    
    return cwt_scaled

def hht_transform(data: np.ndarray) -> np.ndarray:
    """希尔伯特-黄变换(HHT)实现：融合瞬时幅值和频率的特征"""
    # 确保输入为NumPy数组并转换为浮点型
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()  # 处理PyTorch Tensor输入
    data = np.asarray(data, dtype=np.float64)  # 统一转换为float64
    data_len = len(data)
    
    # 1. 经验模态分解(EMD)获取本征模态函数(IMF)
    emd = EMD()
    try:
        imfs = emd.emd(data)  # 输出形状: (n_imfs, data_len)
    except Exception as e:
        print(f"EMD分解失败: {e}")
        return np.outer(data, data)  # 回退方案: 原始数据外积图
    
    # 处理分解结果异常的情况
    n_imfs = imfs.shape[0] if imfs.ndim == 2 else 0
    if n_imfs == 0 or imfs.shape[1] != data_len:
        return np.outer(data, data)
    
    # 2. 对每个IMF提取瞬时幅值和频率并融合
    fused_features = []
    for i in range(n_imfs):
        imf = imfs[i].squeeze()  # 确保IMF为一维数组
        if imf.ndim != 1 or len(imf) != data_len:
            continue  # 跳过无效IMF
        
        # 希尔伯特变换提取瞬时特征
        analytic_signal = hilbert(imf)
        amplitude_envelope = np.abs(analytic_signal)  # 瞬时幅值: [data_len]
        
        # 计算瞬时频率（相位的导数）
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))  # 去卷绕相位
        instantaneous_freq = np.diff(instantaneous_phase) / (2 * np.pi)  # 瞬时频率
        instantaneous_freq = np.pad(instantaneous_freq, (0, 1), mode='edge')  # 补齐长度: [data_len]
        
        # 融合幅值和频率（按行堆叠，形成2行data_len列的特征）
        fused = np.stack([amplitude_envelope, instantaneous_freq], axis=0)  # [2, data_len]
        fused_features.append(fused)
    
    # 处理无有效特征的情况
    if not fused_features:
        return np.outer(data, data)
    
    # 3. 组合所有IMF的融合特征
    hht_combined = np.concatenate(fused_features, axis=0)  # 形状: (2*n_valid_imfs, data_len)
    
    # 4. 缩放至目标尺寸 (data_len, data_len)
    try:
        # 计算缩放因子（确保第一维缩放到data_len，第二维保持不变）
        scale_rows = data_len / hht_combined.shape[0]
        scale_cols = 1.0  # 列维度已为data_len，无需缩放
        hht_scaled = zoom(
            hht_combined,
            zoom=(scale_rows, scale_cols),
            order=1,      # 双线性插值
            mode='nearest'# 边缘处理
        )
    except Exception as e:
        print(f"特征缩放失败: {e}")
        return np.outer(data, data)
    
    # 5. 最终维度校验与调整
    if hht_scaled.ndim != 2:
        hht_scaled = hht_scaled.reshape(data_len, data_len)
    if hht_scaled.shape != (data_len, data_len):
        hht_scaled = zoom(hht_scaled, (data_len/hht_scaled.shape[0], data_len/hht_scaled.shape[1]), order=1)
    
    return hht_scaled

def recurrence_plot(data, eps=1, steps=10):
    """递归图(RP)计算"""
    d = pairwise_distances(data)
    d = d / eps
    d[d > steps] = steps
    return d

def get_mtf(x, size=5):
    """马尔可夫转换场(MTF)计算"""
    # 使用数据的实际最小最大值计算分位数
    x = x.cpu().numpy().flatten()
    min_val = np.min(x)
    max_val = np.max(x)
    
    # 生成分位数
    quantiles = np.linspace(min_val, max_val, size + 1)[1:]  # 创建size个分位数
    
    # 将值映射到分位数索引
    def value_to_quantile(val):
        for i, q in enumerate(quantiles):
            if val <= q:
                return i
        return size - 1  # 如果值大于所有分位数，返回最后一个索引
    
    # 向量化映射函数
    q_func = np.vectorize(value_to_quantile)
    q = q_func(x)
    
    # 构建转移矩阵
    transition_matrix = np.zeros((size, size))
    for i in range(len(x) - 1):
        transition_matrix[q[i], q[i + 1]] += 1
    
    # 归一化转移矩阵（行归一化）
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # 避免除以零
    transition_matrix = transition_matrix / row_sums
    
    # 构建MTF矩阵
    n = len(x)
    mtf_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mtf_matrix[i, j] = transition_matrix[q[i], q[j]]
    
    return mtf_matrix

def visualize_multichannel(features, channel_names):
    """可视化多通道特征图"""
    features = features[:3]  # 取前三个样本
    num_samples = features.shape[0]
    num_channels = features.shape[1]
    
    # 通道名称映射
    name_map = {
        0: "GAF", 1: "STFT", 2: "CWT", 
        3: "HHT", 4: "MTF", 5: "RP"
    }
    
    # 设置画布和网格
    plt.figure(figsize=(5 * num_channels, 5 * num_samples))
    gs = GridSpec(num_samples, num_channels, hspace=0.3, wspace=0.3)
    
    # 遍历每个样本和每个通道
    for sample_idx in range(num_samples):
        for channel_idx in range(num_channels):
            ax = plt.subplot(gs[sample_idx, channel_idx])
            channel_num = channel_names[channel_idx]
            channel_name = name_map.get(channel_num, f"Channel {channel_num}")
            ax.set_title(f"Sample {sample_idx+1} - {channel_name}", fontsize=10)
            
            # 显示特征图
            im = ax.imshow(features[sample_idx, channel_idx], cmap='viridis', aspect='equal')
            ax.axis('off')
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
    
    plt.savefig('visualize-results/多通道特征图可视化.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()