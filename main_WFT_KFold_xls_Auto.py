# 用于自动完成15个不同window长度时，不同overlap但是数据集相同长度的训练
import time, os, torch
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from collections import Counter, defaultdict
from models import *
from utils.dataAugmenter import DataAugmenter
from utils.to2D import wavetoimagetransf
from utils.tSNE import visualize_wheel_force_features
from utils.sample_loss_analyzer import SampleLossAnalyzer
from utils.visualize import *

# 随机种子必须设置，否则每次运行都不同，设置后，每次运行结果都是一样的
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(2025)

# 如果不存在指定目录，则创建
def create_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"目录 '{dir_path}' 创建成功")
    else:
        print(f"目录 '{dir_path}' 已存在")

def print_and_log(hintstr):
    print(hintstr)
    fp_log.write(hintstr)
    fp_log.write('\n')
    fp_log.flush()

def save_to_csv(hintstr):
    fp_result.write(hintstr)
    fp_result.write('\n')
    fp_result.flush()

def save_to_csv_confuse(hintstr):
    fp_result_confuse.write(hintstr)
    fp_result_confuse.write('\n')
    fp_result_confuse.flush()

# 绘制精度和损失曲线
def display_figure_lossaccuray(fig_idx, num_epochs, results_path, train_accuracy_list, train_loss_list, valid_accuracy_list, valid_loss_list):
    plt.close(fig_idx)
    epochs = range(1, num_epochs + 1)
    fig = plt.figure(num=fig_idx, figsize=(18, 6), dpi=100)

    train_accuracy_list_cpu = train_accuracy_list
    valid_accuracy_list_cpu = valid_accuracy_list
    train_accuracy_list_cpu = torch.tensor(train_accuracy_list_cpu).detach().cpu().numpy()
    valid_accuracy_list_cpu = torch.tensor(valid_accuracy_list_cpu).detach().cpu().numpy()

    # 绘制损失图像
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, label='Train Loss')
    plt.plot(epochs, valid_loss_list, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Epochs')

    # 绘制准确率图像
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy_list_cpu, label='Train Accuracy')
    plt.plot(epochs, valid_accuracy_list_cpu, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs. Epochs')

    image_pathname = results_path + 'Fold_' + str(k+1) + '.jpg'
    plt.savefig(image_pathname, dpi=600, bbox_inches='tight', pad_inches=0)

def get_params_without_bn(model):
    params = []
    for name, param in model.named_parameters():
        if 'bn' in name or 'bias' in name:
            # BN参数和偏置不应用权重衰减
            params.append({'params': param, 'weight_decay': 0.0})
        else:
            params.append({'params': param, 'weight_decay': 1e-3})  
    return params

def visualize_relation_distribution(relation_records, save_dir, epoch=None):
    """可视化relation值的分布情况"""
    if epoch is None:
        # 生成所有epoch的relation分布趋势图
        plt.figure(figsize=(12, 6))
        epochs = sorted([int(k.split('_')[1]) for k in relation_records.keys()])
        means = []
        stds = []
        medians = []
        
        for ep in epochs:
            rels = np.array(relation_records[f'epoch_{ep}'])
            means.append(np.mean(rels))
            stds.append(np.std(rels))
            medians.append(np.median(rels))
        
        plt.plot(epochs, means, 'o-', label='Mean')
        plt.fill_between(epochs, np.array(means) - np.array(stds), 
                        np.array(means) + np.array(stds), alpha=0.2, label='±1 Std')
        plt.plot(epochs, medians, 's-', label='Median')
        plt.ylim(0, 1)  # relation值范围在0-1之间
        plt.xlabel('Epoch')
        plt.ylabel('Relation Value')
        plt.title('Relation Value Trend During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'relation_trend.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存数值数据
        trend_data = {
            'epochs': epochs,
            'means': means,
            'stds': stds,
            'medians': medians
        }
        with open(os.path.join(save_dir, 'relation_trend_data.json'), 'w') as f:
            json.dump(trend_data, f, indent=2)
    else:
        # 生成特定epoch的relation分布图
        if f'epoch_{epoch}' not in relation_records:
            return
        
        rels = np.array(relation_records[f'epoch_{epoch}'])
        plt.figure(figsize=(10, 6))
        plt.hist(rels, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(np.mean(rels), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(rels):.4f}')
        plt.axvline(np.median(rels), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(rels):.4f}')
        plt.xlim(0, 1)
        plt.xlabel('Relation Value')
        plt.ylabel('Frequency')
        plt.title(f'Relation Value Distribution at Epoch {epoch}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'relation_distribution_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

Epoch =700
learn_rate = 0.001            # 0.001
batch_size = 256              # 批量大小
label_count = 7               # 分类数量
force_type = 'Fx'             # 选择力类型，Fx/Fy/Fz/My
Selected_channel = 5          # 定义要使用的二维化通道（0-5对应GAF/STFT/CWT/HHT/MTF/RP）
# 可选值：'CNN_1D'/'CNN_2D'/'ResNet_2D'/'CNN_SMNK1D'/'CNN_PMNK1D'/'CNN_SPMNK1D'/'CNN_SBPMNK1D'
model_name = 'CNN_1D+CNN_2D'       
Augmentation = True           # 是否使用数据增强
Use_meta = False              # 是否使用元学习集成
Classweights = True           # 是否使用类别权重
tSNE = True                   # 是否使用tSNE可视化
model_similarity = False      # 是否计算模型相似度
Modelsave = False             # 是否保存模型
hidden_dim = 128              # 隐藏层维度        

labels_Cn = ['大碎石路面', '左右起伏路', '普通平整水泥路', '普通砖石路', '条纹型路面', '窨井盖路', '馒头圆饼路']
labels_En = ['Large gravel road', 'Undulating road', 'Ordinary concrete road', 'Masonry road', 'Striped road', 'Manhole cover road', 'Round cake road']
window_size = [64]

# 每次训练数据都放在 result_file 中
formatted_time = time.strftime("%Y-%m-%d %H-%M", time.localtime())
result_file = 'Result//不同窗口尺寸和重叠率//' + force_type + '_' + model_name + '_' + formatted_time + '.csv'
result_file_confuse = 'Result//不同窗口尺寸和重叠率//' + force_type + '_' + model_name + '_confuse_' + formatted_time + '.csv'
fp_result = open(result_file, "w")    # 不存在则创建；存在则只追加内容
fp_result_confuse = open(result_file_confuse, "w")

save_to_csv('window size,overlap,验证精度,平均最高验证精度,最高验证精度,F1 Score,大碎石路面,左右起伏路,普通平整水泥路,普通砖石路,条纹型路面,窨井盖路,馒头圆饼路')
save_to_csv_confuse('window size,overlap,Fold,大碎石路面,左右起伏路,普通平整水泥路,普通砖石路,条纹型路面,窨井盖路,馒头圆饼路')

# 创建一个空的字典，用于存储每个window size的全局权重记录
global_weight_records = defaultdict(list)

# 遍历所有window size
for array_idx, array_len in enumerate(window_size):
    print_and_log('window size = ', array_len)
    if array_len==64:
        overlap_array = [16]
    elif array_len==128:
        overlap_array = [0,	32,	64,	96]   # 128
    elif array_len==192:
        overlap_array = [0,	32,	64, 96, 128, 160]   # 192
    elif array_len==256:
        overlap_array = [0,	32,	64, 96, 128, 160, 192, 224]   # 256
    elif array_len==320:
        overlap_array = [0, 40, 80, 120, 160, 200, 240, 256, 280]   # 320
    elif array_len==384:
        overlap_array = [0, 48, 96, 144, 192, 240, 288, 320, 336]   # 384
    elif array_len==448:
        overlap_array = [64, 192, 256, 320, 352, 384, 400, 416]   # 448
    elif array_len==512:
        overlap_array = [64, 256, 352, 384, 416, 448, 464, 480, 488, 496]   # 512

    for overlap_idx, overlap_len in enumerate(overlap_array):
        # 以下代码用于训练模型并保存最优模型和结果
        result_data_dir = 'ori-' + str(array_len) + '-' + str(overlap_len)
        results_dir = 'Result//不同窗口尺寸和重叠率//' + model_name + '//' + result_data_dir + '//' + force_type + '//'
        logfile_pathname = results_dir + force_type + '_' + result_data_dir + '_' + formatted_time + '.txt'
        data_dirname = 'data//不同窗口尺寸和重叠率的车轮力数据//' + result_data_dir + '//'
        create_directory(results_dir)

        # 生成数据文件名
        data_src_filename = data_dirname + 'data_' + force_type + '.mat'
        label_src_filename = data_dirname + 'label_' + force_type + '.mat'

        # 从文件中读取数据
        data_src_mat = loadmat(data_src_filename)
        label_src_mat = loadmat(label_src_filename)
        if force_type == 'Fx':
            data_src = data_src_mat['data_Fx']
            label_src = label_src_mat['label_Fx']
        elif force_type == 'Fy':
            data_src = data_src_mat['data_Fy']
            label_src = label_src_mat['label_Fy']
        elif force_type == 'Fz':
            data_src = data_src_mat['data_Fz']
            label_src = label_src_mat['label_Fz']
        elif force_type == 'My':
            data_src = data_src_mat['data_My']
            label_src = label_src_mat['label_My']

        flat_label_ori = label_src.flatten()    # 将源标签数据展平为一维数组
        counter_ori = Counter(flat_label_ori)   # 统计展平后标签数据中每个元素的出现次数               

        # 处理数据，以满足格式要求
        data_src = torch.from_numpy(data_src).float()
        label_src = torch.from_numpy(label_src)
        data_src = torch.reshape(data_src, (data_src.shape[0], data_src.shape[1], 1))
        # 转置维度以匹配Conv1D层输入要求 [样本数, L, 1] -> [样本数, 1, L]
        data_src = data_src.permute(0, 2, 1)
        label_src = label_src.view(-1)      # 将二维(4096,1)转为（4096,）
        
        sequence_length = data_src.shape[1] # 定义序列长度为数据源的第二个维度
        data_number = data_src.shape[0]     # 定义数据数量为数据源的第一个维度
        
        avg_fold_train_lost = []        # 初始化存储每个fold训练损失的列表
        avg_fold_valid_lost = []        # 初始化存储每个fold验证损失的列表        
        avg_fold_train_accuracy = []    # 初始化存储每个fold训练准确率的列表        
        avg_fold_valid_accuracy = []    # 初始化存储每个fold验证准确率的列表        
        max_fold_train_accuracy = []    # 初始化存储每个fold最大训练准确率的列表       
        max_fold_valid_accuracy = []    # 初始化存储每个fold最大验证准确率的列表        
        fold_accuracyscore_array = []   # 初始化存储每个fold准确率得分的列表        
        fold_precisionscore_array = []  # 初始化存储每个fold精确率得分的列表        
        fold_recallscore_array = []     # 初始化存储每个fold召回率得分的列表        
        fold_f1score_array = []         # 初始化存储每个foldF1得分的列表       
        confusion_matrix_handle1 = []   # 初始化存储每个fold混淆矩阵的第一个列表        
        confusion_matrix_handle2 = []   # 初始化存储每个fold混淆矩阵的第二个列表

        # 每次值都发生变化，所以每次都要重新加载数据
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # 创建过程记录文件
        fp_log = open(logfile_pathname, "w")# 打开日志文件以写入模式，准备记录程序运行日志

        print_and_log('Epoch = ' + str(Epoch))
        print_and_log('learn_rate = ' + str(learn_rate))
        print_and_log('batch_size = ' + str(batch_size))
        print_and_log('label_count = ' + str(label_count))
        print_and_log('force_type = ' + str(force_type))
        print_and_log('Selected_channel = ' + str(Selected_channel))
        print_and_log('model_name = ' + str(model_name))
        print_and_log('Augmentation = ' + str(Augmentation))
        print_and_log('Use_meta = ' + str(Use_meta))
        print_and_log('Classweights = ' + str(Classweights))
        print_and_log('tSNE = ' + str(tSNE))
        print_and_log('meta_hidden_dim = ' + str(hidden_dim))
        print_and_log('Modelsave = ' + str(Modelsave))
        print_and_log("___________________________________________________")
        print_and_log(f"当前 GPU 名称: {torch.cuda.get_device_name()}")

        if model_similarity:
            analyzer = RoadFeatureHeatmapAnalyzer()
            response_1d, response_2d = analyzer.run_analysis(force_type='Fx')# 分析Fx数据

        for k, (train_idx, valid_idx) in enumerate(skf.split(data_src, label_src)):# 对数据集进行k折交叉验证，遍历每一折的数据索引

            # 构造日志字符串，包含当前折号、训练集和验证集的索引形状、批次大小以及数据集的总数量
            log_str = "Fold：{}, {}, {}, batch_size = {}, 数据集数量 = {}".format(k+1, train_idx.shape, valid_idx.shape, batch_size, data_src.shape[0])

            print_and_log(log_str)# 打印日志字符串并将其写入日志文件
            print('train_idx: ', train_idx[:10], '        ', train_idx[-10:])
            print('valod_idx: ', valid_idx[:10], '        ', valid_idx[-10:])

            # 使用训练集的统计量进行归一化(注意：并不是所有数据统计量来归一化)
            train_vectors = data_src[train_idx]
            train_mean = train_vectors.mean(dim=0, keepdim=True)# 表示沿着样本数量的维度（即第0维）计算训练数据子集 train_vectors 每个特征的均值。
            train_std = train_vectors.std(dim=0, keepdim=True)  # 计算训练数据的标准差
            data_src_norm = (data_src - train_mean) / (train_std+ 1e-8) # 在数据归一化部分添加epsilon防止除零

            # 使用相同的索引分割数据集
            train_set, valid_set = data_src_norm[train_idx], data_src_norm[valid_idx]
            train_label, valid_label = label_src[train_idx], label_src[valid_idx]

            # 为一维CNN准备数据
            if Augmentation:
                augmenter = DataAugmenter()
                train_set_1d, train_label = augmenter(train_set, train_label, 
                    methods=['time_warp'],      # 选择增强方法['amplitude', 'noise', 'time_warp', 'reverse', 'shift']
                    num_augmented=2,            # 每个样本增强2次
                    return_merged=True,         # 返回原始和增强数据的合并，false仅返回增强后的数据
                    visualize=True
                )
            else: train_set_1d, train_label = (train_set, train_label) #形状：[N, 1, L]
            valid_set_1d, valid_label = (valid_set, valid_label)
            print_and_log(f"训练集1D形状：{train_set_1d.shape}，验证集1D形状：{valid_set_1d.shape}")
         
            train_set_2d = torch.FloatTensor(wavetoimagetransf(train_set_1d, Selected_channel, save=True))# 形状：[N, 1, L, L]
            valid_set_2d = torch.FloatTensor(wavetoimagetransf(valid_set, Selected_channel, save=False))
            print_and_log(f"训练集2D形状：{train_set_2d.shape}，验证集2D形状：{valid_set_2d.shape}")

            # 类别权重logarithmic
            train_label_np = train_label.cpu().numpy()
            class_counts = Counter(train_label_np)
            num_classes = label_count
            total_samples = len(train_label_np)

            class_weights = []
            for c in range(num_classes):
                count = class_counts.get(c, 0)
                weight = np.log((total_samples + num_classes) / (count + 1))  # +1避免log(0)
                class_weights.append(weight)
            class_weights = torch.FloatTensor(class_weights).cuda()

            # 创建训练集和测试集（同时包含一维数据、二维数据和标签）
            train_dataset = TensorDataset(train_set_1d, train_set_2d, train_label)
            valid_dataset = TensorDataset(valid_set_1d, valid_set_2d, valid_label)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

            train_data_number = train_set_1d.shape[0]
            valid_data_number = valid_set.shape[0]
            print_and_log("训练集: 数据量 = {}, 批数 = {}.".format(train_data_number, len(train_loader)))
            print_and_log("测试集: 数据量 = {}, 批数 = {}.".format(valid_data_number, len(valid_loader)))

            # 创建样本损失分析器
            fold_save_dir = os.path.join(results_dir, 'loss_visualizations', f'fold_{k+1}')
            loss_analyzer = SampleLossAnalyzer(save_dir=fold_save_dir)

            # 根据model_name初始化对应模型
            if 'CNN_1D' in model_name: MAMIL1D = CNN_1D(in_channels = 1) .cuda()
            elif 'CNN_SMNK1D' in model_name: MAMIL1D = CNN_SMNK1D(in_channels = 1) .cuda()
            elif 'CNN_PMNK1D' in model_name: MAMIL1D = CNN_PMNK1D(in_channels = 1) .cuda()
            elif 'CNN_SPMNK1D' in model_name: MAMIL1D = CNN_SPMNK1D(in_channels = 1) .cuda()
            elif 'CNN_SBPMNK1D' in model_name: MAMIL1D = CNN_SBPMNK1D(in_channels = 1) .cuda()
            else: MAMIL1D = None

            if 'CNN_2D' in model_name: MAMIL2D = CNN_2D(in_channels = 1).cuda()
            elif 'ResNet_2D' in model_name: MAMIL2D = ResNet_2D(in_channels = 1).cuda()
            elif 'HMCNN2D' in model_name: MAMIL2D = HMCNN_2D(in_channels = 1).cuda()
            else: MAMIL2D = None

            attention = CrossModalAttention(dim1 = 64, dim2 = 64, hidden_dim = hidden_dim).cuda() if '+' in model_name else None 
            classifier = Classifier(in_features = hidden_dim if '+' in model_name else 64, num_classes = label_count).cuda()
            # 损失函数将LogSoftMax和NLLLoss集成到一个类中，不使用类别权重，类别权重将和损失函数中的权重合并
            loss_ce = nn.CrossEntropyLoss(weight=class_weights, reduction='none') if Classweights else nn.CrossEntropyLoss(reduction='none')
            
            #修改优化器配置
            optimizers = []
            schedulers = []
            
            if MAMIL1D:
                cnn1d_optimizer = torch.optim.Adam(MAMIL1D.parameters(), lr=learn_rate, weight_decay=1e-4)
                optimizers.append(cnn1d_optimizer)
                scheduler_1D = torch.optim.lr_scheduler.ReduceLROnPlateau(cnn1d_optimizer, mode='min', factor=0.5, patience=10,min_lr=1e-6)
                schedulers.append(scheduler_1D)

            if MAMIL2D:
                cnn2d_optimizer = torch.optim.Adam(get_params_without_bn(MAMIL2D), lr=learn_rate * 0.5)
                optimizers.append(cnn2d_optimizer)
                scheduler_2D = torch.optim.lr_scheduler.ReduceLROnPlateau(cnn2d_optimizer, mode='min',factor=0.5,patience=10,min_lr=1e-6)
                schedulers.append(scheduler_2D)

            if attention:
                attention_optimizer = torch.optim.Adam(attention.parameters(), lr=learn_rate*0.3, weight_decay=1e-4)
                optimizers.append(attention_optimizer)
                scheduler_AT = torch.optim.lr_scheduler.ReduceLROnPlateau(attention_optimizer, mode='min',factor=0.5,patience=10,min_lr=1e-6)            
                schedulers.append(scheduler_AT)
                
            classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=learn_rate * 0.3, weight_decay=1e-4)
            optimizers.append(classifier_optimizer)      
            scheduler_CL = torch.optim.lr_scheduler.ReduceLROnPlateau(classifier_optimizer, mode='min',factor=0.5,patience=10,min_lr=1e-6)
            schedulers.append(scheduler_CL)

            outputs_valid = []
            targets_valid = []
            list_train_loss = []
            list_train_accuracy = []
            list_valid_loss = []
            list_valid_accuracy = []
            list_train_lastepoch_loss = []
            list_train_lastepoch_accuracy = []
            list_valid1_loss = []
            list_valid1_accuracy = []
            time_start = time.time()

            # 在训练循环之前初始化记录器
            relation_records = defaultdict(list)  # 用于记录relation值

            for i in range(Epoch):
                if i % 100 == 0:
                    print_and_log("___________________________________________________")
                    print_and_log("window长度={}, overlap = {}, 第{}折, 第{}轮训练".format(array_len, overlap_len, k+1, i + 1))

                if MAMIL1D:
                    MAMIL1D.train()  # 将模型设置为训练模式
                if MAMIL2D:
                    MAMIL2D.train()
                if attention:
                    attention.train()
                classifier.train()

                total_train_step = 0
                loss_last_train = 0
                accuracy_last_train = 0
                
                for data_1D, data_2D, target in train_loader: 
                    if MAMIL1D:data_1D = data_1D.cuda()
                    if MAMIL2D:data_2D = data_2D.cuda() 
                    target = target.cuda()
                    batch_size_new = data_1D.shape[0]

                    # 前向传播（适配单/双模型）
                    if '+' in model_name:  
                        features_1d = MAMIL1D(data_1D) 
                        features_2d = MAMIL2D(data_2D) 
                        fused_features, relation = attention(features_1d, features_2d)# 双模型融合
                        # 收集当前batch的relation值
                        relation_values = relation.detach().cpu().numpy().flatten()
                        relation_records[f'epoch_{i+1}'].extend(relation_values.tolist())
                        outputs = classifier(fused_features)
                    else:  # 单模型直接输出
                        if MAMIL1D: features = MAMIL1D(data_1D)
                        else:features = MAMIL2D(data_2D)
                        outputs = classifier(features)

                    # 总损失计算（基础损失 + L1正则 + 集成损失）
                    loss = loss_ce(outputs, target).mean()                   

                    # 同时清空优化器的梯度
                    for optimizer in optimizers:
                        optimizer.zero_grad()

                    # 优化器梯度
                    loss.backward()

                    # 合并梯度裁剪
                    params_to_clip = []
                    if MAMIL1D: params_to_clip.extend(MAMIL1D.parameters())
                    if MAMIL2D: params_to_clip.extend(MAMIL2D.parameters())
                    if attention: params_to_clip.extend(attention.parameters())
                    params_to_clip.extend(classifier.parameters())
                    torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=3.0)

                    # 参数更新
                    for optimizer in optimizers:
                        optimizer.step()

                    # 计算训练指标
                    total_train_step += 1
                    accuracy_last_train = accuracy_last_train + (outputs.argmax(1) == target).sum() / batch_size_new
                    loss_last_train = loss_last_train + loss.item()

                    # 记录样本损失（每10个epoch记录一次）
                    if i % 10 == 0:  # 每10个epoch记录一次，避免数据过多
                        # 计算每个样本的损失
                        loss_per_sample = loss_ce(outputs, target)
                        # 创建样本索引
                        sample_indices = torch.arange(total_train_step * batch_size_new - batch_size_new, 
                                                    total_train_step * batch_size_new)
                        loss_analyzer.record_batch(loss_per_sample, outputs, target, sample_indices)

                # 将每一个epoch获得训练集精度和损失保存在列表中
                list_train_loss.append((loss_last_train / len(train_loader)))
                list_train_accuracy.append((accuracy_last_train / len(train_loader)).data.cpu())

                # 进行模型的验证
                total_valid_step = 0
                loss_last_valid = 0
                accuracy_last_valid = 0
                ensemble_accuracy = 0

                if MAMIL1D: MAMIL1D.eval()
                if MAMIL2D: MAMIL2D.eval()
                if attention: attention.eval()
                classifier.eval()

                outputs_valid = [] if (i + 1 == Epoch) else outputs_valid
                targets_valid = [] if (i + 1 == Epoch) else targets_valid

                with torch.no_grad():
                    for data_1D, data_2D, target in valid_loader:
                        if MAMIL1D:data_1D = data_1D.cuda()
                        if MAMIL2D:data_2D = data_2D.cuda()
                        target = target.cuda()
                        batch_size_new = data_1D.shape[0]

                        if '+' in model_name:  # 双模型融合
                            features_1d = MAMIL1D(data_1D) 
                            features_2d = MAMIL2D(data_2D) 
                            fused_features, relation = attention(features_1d, features_2d)# 双模型融合
                            outputs = classifier(fused_features)
                        else:  # 单模型直接输出
                            if MAMIL1D:features = MAMIL1D(data_1D)
                            else:features = MAMIL2D(data_2D)
                            outputs = classifier(features)
                        
                        total_valid_step += 1
                        batch_loss = loss_ce(outputs, target).mean()
                        loss_last_valid += batch_loss.item()
                        accuracy_last_valid += (outputs.argmax(1) == target).sum() / batch_size_new
                        valid_loss = loss_last_valid / len(valid_loader)

                        if i + 1 == Epoch:
                            output_result = Tensor.cpu(outputs.argmax(1))
                            target_result = Tensor.cpu(target)
                            outputs_valid = np.concatenate((outputs_valid, output_result))
                            targets_valid = np.concatenate((targets_valid, target_result))
                # 记录验证损失和精度
                list_valid_loss.append(loss_last_valid / len(valid_loader))
                list_valid_accuracy.append((accuracy_last_valid / len(valid_loader)).data.cpu())

                # 学习率调度
                for scheduler in schedulers:
                    scheduler.step(valid_loss)

                if i % 100 == 0:
                    print_and_log("训练集第 {} 轮正确率:{:.4f}".format((i + 1), accuracy_last_train / total_train_step))
                    print_and_log("训练集第 {} 轮损失值:{:.4f}".format((i + 1), loss_last_train / total_train_step))
                    print_and_log("验证集第 {} 轮正确率:{:.4f}".format((i + 1), accuracy_last_valid / total_valid_step))
                    print_and_log("验证集第 {} 轮损失值:{:.4f}".format((i + 1), loss_last_valid / total_valid_step))

                if i + 1 == Epoch:
                    fold_accuracy_score = accuracy_score(targets_valid, outputs_valid)
                    fold_precision_score = precision_score(targets_valid, outputs_valid, average="macro") # 0.33333
                    fold_recall_score = recall_score(targets_valid, outputs_valid, average="macro")
                    fold_f1_score = f1_score(targets_valid, outputs_valid, average="macro")

                    fold_accuracyscore_array.append(fold_accuracy_score)
                    fold_precisionscore_array.append(fold_precision_score)
                    fold_recallscore_array.append(fold_recall_score)
                    fold_f1score_array.append(fold_f1_score)

                    # 查看验证集中每个类
                    flat_label_ori = targets_valid.flatten()
                    counter_ori_valid = Counter(flat_label_ori)
                    print(counter_ori_valid)

                    # 计算混淆矩阵并保存
                    cm1 = confusion_matrix(targets_valid, outputs_valid)
                    cm2 = confusion_matrix(targets_valid, outputs_valid, normalize='true')
                    confusion_matrix_handle1.append(cm1)
                    confusion_matrix_handle2.append(cm2)

                    print_and_log("第 {} 折: 准确率 = {:.4f}, 精确率 = {:.4f}, 召回率 = {:.4f}, F1 = {:.4f}".format(k+1, fold_accuracy_score, fold_precision_score, fold_recall_score, fold_f1_score))

            # 对每一折过程进行统计
            avg_train_lost = (sum(list_train_loss) / len(list_train_loss))
            avg_valid_lost = (sum(list_valid_loss) / len(list_valid_loss))
            avg_train_accuracy = (sum(list_train_accuracy) / len(list_train_accuracy))
            avg_valid_accuracy = (sum(list_valid_accuracy) / len(list_valid_accuracy))
            max_train_accuracy = max(list_train_accuracy)
            max_valid_accuracy = max(list_valid_accuracy)

            train_avg_lost_str = "第 {} 折: 训练集平均损失: {:.4f}".format(k+1, avg_train_lost)
            valid_avg_lost_str = "第 {} 折: 验证集平均损失: {:.4f}".format(k+1, avg_valid_lost)
            train_avg_accuracy_str = "第 {} 折: 训练集平均精度: {:.4f}".format(k+1, avg_train_accuracy)
            valid_avg_accuracy_str = "第 {} 折: 验证集平均精度: {:.4f}".format(k+1, avg_valid_accuracy)
            train_max_accuracy_str = "第 {} 折: 训练集最高精度: {:.4f}".format(k+1, max_train_accuracy)
            valid_max_accuracy_str = "第 {} 折: 验证集最高精度: {:.4f}".format(k+1, max_valid_accuracy)
            print_and_log(train_avg_lost_str)
            print_and_log(valid_avg_lost_str)
            print_and_log(train_avg_accuracy_str)
            print_and_log(valid_avg_accuracy_str)
            print_and_log(train_max_accuracy_str)
            print_and_log(valid_max_accuracy_str)

            # 将每一个fold的平均训练精度和验证精度保存在列表中
            avg_fold_train_lost.append(avg_train_lost)
            avg_fold_valid_lost.append(avg_valid_lost)
            avg_fold_train_accuracy.append(avg_train_accuracy)
            avg_fold_valid_accuracy.append(avg_valid_accuracy)
            max_fold_train_accuracy.append(max_train_accuracy)
            max_fold_valid_accuracy.append(max_valid_accuracy)

            time_end = time.time()
            time_c = time_end - time_start
            print_and_log("第 {} 折: time cost = {}min{}s".format(k+1, int(time_c / 60), int(time_c % 60)))

            # 绘制曲线
            display_figure_lossaccuray(overlap_idx*len(overlap_array)+k, Epoch, results_dir, list_train_accuracy, list_train_loss, list_valid_accuracy, list_valid_loss)

            # 生成损失分析报告
            print_and_log(f'生成第{k+1}折的损失分析报告...')
            loss_analyzer.generate_summary(class_names=labels_Cn, top_k=20)
            print_and_log(f'损失分析报告已生成并保存至: {fold_save_dir}')

            if tSNE:
                # 添加t-SNE可视化
                print_and_log('开始t-SNE特征可视化...')
                # 准备t-SNE所需的验证集数据加载器
                tsne_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                # 创建t-SNE结果保存目录
                tsne_save_dir = os.path.join(results_dir, 'tsne_visualizations')
                os.makedirs(tsne_save_dir, exist_ok=True)
                # 根据模型类型选择要可视化的模型
                if '+' in model_name:
                    # 对于双模型，使用融合后的特征
                    visualize_model =  nn.Sequential( MAMIL1D, MAMIL2D, attention)
                    model_type = 'fusionmodel'
                else:
                    # 对于单模型，使用特征提取部分
                    visualize_model = MAMIL1D if MAMIL1D is not None else MAMIL2D
                    model_type = 'MAMIL1D' if MAMIL1D is not None else 'MAMIL2D'
                # 执行t-SNE可视化
                visualize_wheel_force_features(
                    model=visualize_model,
                    data_loader=tsne_loader,
                    device=torch.device('cuda'),
                    output_dir=tsne_save_dir,
                    n_components=2,
                    model_type=model_type,
                    class_names=labels_Cn,
                    k=k,
                    )
                print_and_log(f't-SNE可视化结果已保存至: {tsne_save_dir}')

            # 在每个k折的epoch循环结束后，保存训练好的模型，方便复现最佳结果或后续分析
            if Modelsave:
                model_saved_path = os.path.join(results_dir, 'model_saved', f'fold_{k+1}.pth')
                os.makedirs(os.path.dirname(model_saved_path), exist_ok=True)
                torch.save({
                    'MAMIL1D': MAMIL1D.state_dict() if MAMIL1D else None,
                    'MAMIL2D': MAMIL2D.state_dict() if MAMIL2D else None,
                    'attention': attention.state_dict() if attention else None,
                    'classifier': classifier.state_dict()
                }, model_saved_path)
                print_and_log(f"第{k+1}折模型已保存至: {model_saved_path}")

        # 绘制混淆矩阵
        # Y坐标是真实标签，对应一行数字的和表示该标签的实际数量，所以一行的比例相加=1
        print_and_log('-----confuse matrix-----')
        for idx, cm in enumerate(confusion_matrix_handle1):
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_Cn)
            fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
            disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=20)
            plt.tight_layout()
            image_pathname = results_dir + 'CM_数量_' + str(idx+1) + '.jpg'
            plt.savefig(image_pathname, dpi=300, bbox_inches='tight', pad_inches=0.2)
            print_and_log('----- 数量matrix {}-----'.format(idx))
            for label_idx in range(label_count):
                print_and_log("   {} {} {} {} {} {} {}".format(cm[label_idx][0], cm[label_idx][1], cm[label_idx][2], cm[label_idx][3], cm[label_idx][4], cm[label_idx][5], cm[label_idx][6]))

        for idx, cm in enumerate(confusion_matrix_handle2):
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_Cn,)
            fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
            disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=20)
            plt.tight_layout()
            image_pathname = results_dir + 'CM_比例_' + str(idx+1) + '.jpg'
            plt.savefig(image_pathname, dpi=300, bbox_inches='tight', pad_inches=0.2)

            print_and_log('----- 比例matrix {}-----'.format(idx+1))
            for label_idx in range(label_count):
                print_and_log("   {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}".format(cm[label_idx][0], cm[label_idx][1], cm[label_idx][2], cm[label_idx][3], cm[label_idx][4], cm[label_idx][5], cm[label_idx][6]))
            
            # 保存混淆矩阵到csv文件
            log_str = " {},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(array_len, overlap_len, idx+1, cm[0][0], cm[1][1], cm[2][2], cm[3][3], cm[4][4], cm[5][5], cm[6][6])       
            save_to_csv_confuse(log_str)

        # 输出信息
        print_and_log('*********************************************')
        formatted_list = [f"{num:.4f}" for num in avg_fold_train_lost]
        print_and_log("avg_fold_train_lost: {}".format(formatted_list))
        formatted_list = [f"{num:.4f}" for num in avg_fold_train_accuracy]
        print_and_log("avg_fold_train_accuracy: {}".format(formatted_list))
        formatted_list = [f"{num:.4f}" for num in max_fold_train_accuracy]
        print_and_log("max_fold_train_accuracy: {}".format(formatted_list))

        print_and_log('---------------------------------------------------------------')
        formatted_list = [f"{num:.4f}" for num in avg_fold_valid_lost]
        print_and_log("avg_fold_valid_lost: {}".format(formatted_list))
        formatted_list = [f"{num:.4f}" for num in avg_fold_valid_accuracy]
        print_and_log("avg_fold_valid_accuracy: {}".format(formatted_list))
        formatted_list = [f"{num:.4f}" for num in max_fold_valid_accuracy]
        print_and_log("max_fold_valid_accuracy: {}".format(formatted_list))

        print_and_log('---------------------------------------------------------------')
        print_and_log("K-Fold 平均训练损失: {}".format(sum(avg_fold_train_lost) / len(avg_fold_train_lost)))
        print_and_log("K-Fold 平均训练精度: {}".format(sum(avg_fold_train_accuracy) / len(avg_fold_train_accuracy)))
        print_and_log("K-Fold 最高训练精度: {}".format(sum(max_fold_train_accuracy) / len(max_fold_train_accuracy)))
        print_and_log("K-Fold 平均验证损失: {}".format(sum(avg_fold_valid_lost) / len(avg_fold_valid_lost)))
        print_and_log("K-Fold 平均验证精度: {}".format(sum(avg_fold_valid_accuracy) / len(avg_fold_valid_accuracy)))
        print_and_log("K-Fold 平均最高验证精度: {}".format(sum(max_fold_valid_accuracy) / len(max_fold_valid_accuracy)))
        print_and_log("K-Fold 最高验证精度: {}".format(max(max_fold_valid_accuracy)))

        print_and_log('---------------------------------------------------------------')
        formatted_list = [f"{num:.4f}" for num in fold_accuracyscore_array]
        print_and_log("fold_accuracyscore_array: {}".format(formatted_list))
        formatted_list = [f"{num:.4f}" for num in fold_precisionscore_array]
        print_and_log("fold_precisionscore_array: {}".format(formatted_list))
        formatted_list = [f"{num:.4f}" for num in fold_recallscore_array]
        print_and_log("fold_recallscore_array: {}".format(formatted_list))
        formatted_list = [f"{num:.4f}" for num in fold_f1score_array]
        print_and_log("fold_f1score_array: {}".format(formatted_list))

        print_and_log("K-Fold 平均验证准确率: {}".format(sum(fold_accuracyscore_array) / len(fold_accuracyscore_array)))
        print_and_log("K-Fold 平均验证精确率: {}".format(sum(fold_precisionscore_array) / len(fold_precisionscore_array)))
        print_and_log("K-Fold 平均验证召回率: {}".format(sum(fold_recallscore_array) / len(fold_recallscore_array)))
        print_and_log("K-Fold 平均验证F1-Score: {}".format(sum(fold_f1score_array) / len(fold_f1score_array)))

        # 保存结果到csv文件
        log_str = " {},{},{:.6f},{:.6f},{:.6f},{:.6f}".format(array_len, overlap_len, sum(avg_fold_valid_accuracy) / len(avg_fold_valid_accuracy), \
                   sum(max_fold_valid_accuracy) / len(max_fold_valid_accuracy), max(max_fold_valid_accuracy), sum(fold_f1score_array) / len(fold_f1score_array))
        save_to_csv(log_str)
        print_and_log('*********************************************')

# 使用visualize_modality_comparison对比不同模态的特征分布
if 'CNN_1D' in model_name and 'CNN_2D' in model_name and len(fold_f1score_array) > 0 and model_similarity:
    # 确保有验证数据集可用
    if 'valid_dataset' in locals() or 'valid_dataset' in globals():
        visualize_modality_comparison(
            fold_f1score_array=fold_f1score_array,
            results_dir=results_dir,
            model_name=model_name,
            valid_dataset=valid_dataset,
            labels_Cn=labels_Cn
        )
    else:
        print("警告：未找到验证数据集，无法进行模态特征对比可视化")

print_and_log("训练完成.")
fp_log.close()
fp_result.close()
fp_result_confuse.close()