import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict

class SampleLossAnalyzer:
    """单个样本损失分析器：记录和可视化每个样本的损失值，帮助分析模型表现"""
    def __init__(self, save_dir='./loss_visualizations'):
        self.save_dir = save_dir
        self.sample_losses = defaultdict(list)  # 存储每个样本的损失历史
        self.sample_predictions = []  # 存储预测结果
        self.sample_labels = []  # 存储真实标签
        self.sample_indices = []  # 存储样本索引
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
    
    def record_batch(self, batch_losses, predictions, labels, indices=None):
        """记录一个批次的样本损失
        
        Args:
            batch_losses: 形状为[B]的损失张量
            predictions: 模型预测结果，形状为[B, num_classes]
            labels: 真实标签，形状为[B]
            indices: 样本索引，如果为None则自动生成"""
        # 转换为numpy数组
        losses_np = batch_losses.cpu().detach().numpy()
        preds_np = predictions.cpu().detach().numpy()
        labels_np = labels.cpu().detach().numpy()
        
        # 如果没有提供索引，使用递增的数字
        if indices is None:
            indices = range(len(labels_np))
        else:
            indices = indices.cpu().detach().numpy()
        
        # 记录损失
        for i, idx in enumerate(indices):
            self.sample_losses[idx].append(losses_np[i])
            self.sample_predictions.append((idx, preds_np[i]))
            self.sample_labels.append((idx, labels_np[i]))
    
    def get_hardest_samples(self, top_k=20):
        """获取损失最大的样本
        
        Args:
            top_k: 返回前k个最难的样本
            
        Returns:
            包含(索引, 平均损失, 预测, 标签)的列表"""
        # 计算每个样本的平均损失
        avg_losses = {}
        for idx, losses in self.sample_losses.items():
            avg_losses[idx] = np.mean(losses)
        
        # 按平均损失降序排序
        sorted_samples = sorted(avg_losses.items(), key=lambda x: x[1], reverse=True)
        
        # 获取最难的样本及其信息
        hardest_samples = []
        for idx, loss in sorted_samples[:top_k]:
            # 找到该样本的预测和标签
            pred = None
            label = None
            for i, (sample_idx, p) in enumerate(self.sample_predictions):
                if sample_idx == idx:
                    pred = p
                    label = self.sample_labels[i][1]
                    break
            hardest_samples.append((idx, loss, pred, label))
        
        return hardest_samples
    
    def plot_loss_distribution(self, title='样本损失分布', filename='loss_distribution.png'):
        """绘制样本损失分布直方图"""
        '''整体分布分析：查看直方图的形状，了解大部分样本的损失值集中在什么范围
        高损失样本识别：右侧尾部（损失值较大的区域）如果有明显的样本分布，表明存在一些模型难以处理的样本
        异常检测：如果在极高损失值区域有孤立的样本，这些可能是异常值或标注错误的数据
        模型性能评估：理想情况下，直方图应该左偏，大部分样本集中在较低损失区域'''
        # 计算每个样本的平均损失
        avg_losses = [np.mean(losses) for losses in self.sample_losses.values()]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(avg_losses, bins=50, kde=True)
        plt.title(title)
        plt.xlabel('平均损失值')
        plt.ylabel('样本数量')
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"损失分布图已保存至: {save_path}")
    
    def plot_loss_by_class(self, class_names=None, filename='loss_by_class.png'):
        """按类别绘制平均损失"""
        '''误差线：表示每个类别的损失值波动范围
        数值标签：直接显示每个类别的平均损失值

        实际意义：
        找出平均损失最高的类别，这些是模型最难识别的路面类型
        比较不同类别的损失差异，如果差异很大，可能存在类别不平衡问题
        结合类别数量分析，如果某个类别的样本数量少且损失高，可能需要数据增强'''
        # 按类别分组损失
        class_losses = defaultdict(list)
        
        # 获取每个样本的标签和平均损失
        for idx, losses in self.sample_losses.items():
            avg_loss = np.mean(losses)
            # 查找该样本的标签
            for sample_idx, label in self.sample_labels:
                if sample_idx == idx:
                    class_losses[label].append(avg_loss)
                    break
        
        # 准备绘图数据
        labels = sorted(class_losses.keys())
        mean_losses = [np.mean(class_losses[label]) for label in labels]
        std_losses = [np.std(class_losses[label]) for label in labels]
        
        # 使用类别名称
        if class_names and len(class_names) >= len(labels):
            x_labels = [class_names[label] for label in labels]
        else:
            x_labels = [str(label) for label in labels]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(x_labels, mean_losses, yerr=std_losses, capsize=5)
        
        # 为条形图添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.title('各类别平均损失')
        plt.xlabel('类别')
        plt.ylabel('平均损失值')
        plt.grid(True, axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"类别损失图已保存至: {save_path}")
    
    def plot_loss_vs_confidence(self, filename='loss_vs_confidence.png'):
        """绘制损失值与预测置信度的散点图"""
        '''理想情况：大部分绿色点集中在右下角（高置信度、低损失）

        需要关注：
        右上角的红色点：模型高置信度但预测错误的样本（可能是模型过拟合或数据异常）
        左上角的绿色点：模型低置信度但预测正确的样本（模型不确定但碰巧正确）
        左下角的红色点：模型低置信度且预测错误（预期内的困难样本）'''
        losses = []
        confidences = []
        corrects = []
        
        # 收集数据
        for idx, losses_list in self.sample_losses.items():
            avg_loss = np.mean(losses_list)
            # 查找该样本的预测和标签
            for i, (sample_idx, pred) in enumerate(self.sample_predictions):
                if sample_idx == idx:
                    label = self.sample_labels[i][1]
                    max_conf = np.max(pred)
                    is_correct = np.argmax(pred) == label
                    
                    losses.append(avg_loss)
                    confidences.append(max_conf)
                    corrects.append(is_correct)
                    break
        
        # 转换为numpy数组
        losses = np.array(losses)
        confidences = np.array(confidences)
        corrects = np.array(corrects)
        
        plt.figure(figsize=(10, 6))
        
        # 绘制正确预测的点
        plt.scatter(confidences[corrects], losses[corrects], 
                   alpha=0.6, color='green', label='正确预测', s=30)
        
        # 绘制错误预测的点
        plt.scatter(confidences[~corrects], losses[~corrects], 
                   alpha=0.6, color='red', label='错误预测', s=30)
        
        plt.title('损失值与预测置信度关系')
        plt.xlabel('预测置信度')
        plt.ylabel('平均损失值')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 保存图像
        save_path = os.path.join(self.save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"损失与置信度关系图已保存至: {save_path}")
    
    def generate_summary(self, class_names=None, top_k=20):
        """生成损失分析摘要报告"""
        print("=" * 60)
        print("样本损失分析报告")
        print("=" * 60)
        
        # 基本统计
        avg_losses = [np.mean(losses) for losses in self.sample_losses.values()]
        print(f"总样本数: {len(self.sample_losses)}")
        print(f"平均损失: {np.mean(avg_losses):.4f}")
        print(f"损失标准差: {np.std(avg_losses):.4f}")
        print(f"最小损失: {np.min(avg_losses):.4f}")
        print(f"最大损失: {np.max(avg_losses):.4f}")
        print()
        
        # 获取最难的样本
        hardest_samples = self.get_hardest_samples(top_k)
        print(f"损失最大的{top_k}个样本:")
        print("-" * 60)
        print(f"{'索引':<10}{'平均损失':<15}{'预测类别':<15}{'真实类别':<15}")
        print("-" * 60)
        
        for idx, loss, pred, label in hardest_samples:
            pred_class = np.argmax(pred)
            if class_names:
                pred_name = class_names[pred_class] if pred_class < len(class_names) else str(pred_class)
                label_name = class_names[label] if label < len(class_names) else str(label)
            else:
                pred_name = str(pred_class)
                label_name = str(label)
            
            print(f"{idx:<10}{loss:<15.4f}{pred_name:<15}{label_name:<15}")
        
        print("=" * 60)
        
        # 生成所有可视化
        self.plot_loss_distribution()
        self.plot_loss_by_class(class_names)
        self.plot_loss_vs_confidence()
