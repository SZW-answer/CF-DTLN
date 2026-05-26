"""
RRP (Regression Relevance Propagation) 模块
包含因果解释相关的类和函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# =============================================================================
# 辅助函数
# =============================================================================

def safe_divide(a, b):
    """安全除法，避免除零错误"""
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, input, output):
    """前向传播钩子函数，保存输入和输出"""
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True
    self.Y = output


# =============================================================================
# RelProp 基类和相关层
# =============================================================================

class RelProp(nn.Module):
    """Relevance Propagation 基类"""
    def __init__(self):
        super(RelProp, self).__init__()
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):
        """梯度传播"""
        return torch.autograd.grad(Z, X, S, retain_graph=True)

    def relprop(self, R):
        """相关性反向传播（子类需要重写）"""
        return R


class LeakyReLU(nn.LeakyReLU, RelProp):
    """支持 RRP 的 LeakyReLU"""
    pass


class GELU(nn.GELU, RelProp):
    """支持 RRP 的 GELU"""
    pass


class Softmax(nn.Softmax, RelProp):
    """支持 RRP 的 Softmax"""
    pass


class LayerNorm(nn.LayerNorm, RelProp):
    """支持 RRP 的 LayerNorm"""
    pass


class Dropout(nn.Dropout, RelProp):
    """支持 RRP 的 Dropout"""
    pass


class Clone(RelProp):
    """克隆层 - 将输入复制多份"""
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)
        return outputs

    def relprop(self, R):
        """反向传播相关性"""
        Z = []
        for _ in range(self.num):
            Z.append(self.X)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.X, S)[0]
        R = self.X * C
        return R


class einsum(RelProp):
    """支持 RRP 的 einsum 操作"""
    def __init__(self, equation):
        super().__init__()
        self.equation = equation

    def forward(self, *operands):
        return torch.einsum(self.equation, *operands)

    def relprop(self, R):
        """反向传播相关性"""
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)
        if torch.is_tensor(self.X) == False:
            outputs = [self.X[0] * C[0], self.X[1] * C[1]]
        else:
            outputs = self.X * C[0]
        return outputs


class RegRelProp(RelProp):
    """正则化相关性传播"""
    def relprop(self, R):
        Z = F.linear(self.X, self.weight)
        S = safe_divide(R, Z)
        R = self.X * torch.autograd.grad(Z, self.X, S)[0]
        return R


class Linear(nn.Linear, RegRelProp):
    """支持 RRP 的线性层"""
    pass


# =============================================================================
# 因果解释器
# =============================================================================

class CausalExplainer:
    """
    因果解释器 - 使用 RRP 方法生成因果分数
    参考: code/CausalFormer/explainer/explainer.py
    论文: https://arxiv.org/html/2406.16708v1
    """
    def __init__(self, model, device='cpu'):
        """
        初始化因果解释器
        
        Args:
            model: 训练好的 PredictModel
            device: 计算设备
        """
        self.model = model
        self.model.eval()
        self.device = device

    def generate_causal_scores(self, input_data, target_series_idx, batch_size=1):
        """
        生成因果分数
        
        Args:
            input_data: 输入数据 [N, time_step, series_num, feature_dim]
            target_series_idx: 目标时间序列的索引
            batch_size: 批处理大小
        
        Returns:
            relA: 注意力因果分数 (series_num, series_num)
            relK: 卷积核因果分数
        """
        inputs = torch.split(input_data, batch_size)
        relAs, relKs = [], []

        for data in inputs:
            relA, relK = self._generate_RRP(data, target_series_idx)
            relAs.append(relA)
            relKs.append(relK)

        # 对所有 batch 取平均 (参考 explainer.py 第17-18行)
        relA = torch.stack(relAs).mean(0)
        relK = torch.stack(relKs).mean(0)

        return relA, relK

    def _generate_RRP(self, input_data, target_series_idx):
        """
        内部方法：使用 RRP 生成因果分数
        完全参考 code/CausalFormer/explainer/explainer.py 第21-65行
        
        Args:
            input_data: 输入数据 [batch, time_step, series_num, feature_dim]
            target_series_idx: 目标时间序列的索引
        
        Returns:
            relA: 注意力矩阵因果分数
            relK: 卷积核因果分数
        """
        # 前向传播 (参考 explainer.py 第33行)
        output = self.model(input_data)

        # 创建 one-hot 张量用于目标序列 (参考 explainer.py 第35-36行)
        one_hot = torch.zeros_like(output, dtype=torch.float).to(output.device)
        one_hot[:, :, target_series_idx, :] = 1

        # 克隆 one-hot 并设置 requires_grad (参考 explainer.py 第38-39行)
        one_hot_vector = one_hot.clone()
        one_hot.requires_grad_(True)

        # 计算目标输出 (参考 explainer.py 第41行)
        one_hot_sum = torch.sum(one_hot * output)

        # 反向传播 (参考 explainer.py 第43-44行)
        self.model.zero_grad()
        one_hot_sum.backward(retain_graph=True)

        # 应用 RRP 计算相关性分数 (参考 explainer.py 第46行)
        self.model.relprop(one_hot_vector)

        relAs, relKs = [], []

        # 从每个 encoder 层收集因果分数 (参考 explainer.py 第50-62行)
        for layer in self.model.encoder.layers:
            # 梯度调制 (Gradient Modulation) (参考 explainer.py 第52-53行)
            relA = layer.attention.attention.get_rel() * torch.abs(
                layer.attention.attention.get_grad()
            )
            relK = layer.attention.Wv.get_rel() * torch.abs(
                layer.attention.Wv.get_grad()
            )

            # 只保留正的因果分数 (参考 explainer.py 第59-60行)
            relA = relA.clamp(min=0)
            relK = relK.clamp(min=0)

            # 对样本和头取平均 (参考 explainer.py 第61-62行)
            relAs.append(relA.mean((0, 1)))  # (series_num, series_num)
            relKs.append(relK.mean(0))  # mean for head

        # 沿编码器层维度相乘（累积因果效应）(参考 explainer.py 第63-64行)
        relA = torch.stack(relAs).prod(0)
        relK = torch.stack(relKs).prod(0)

        return relA, relK

    def interpret_causal_graph(self, input_data, target_series_idx, var_names,
                                batch_size=32, threshold=0.1):
        """
        解释因果图，输出因果关系
        
        Args:
            input_data: 输入数据
            target_series_idx: 目标序列索引
            var_names: 变量名称列表
            batch_size: 批处理大小
            threshold: 因果关系阈值
        
        Returns:
            causal_results: 因果关系结果字典
        """
        relA, relK = self.generate_causal_scores(input_data, target_series_idx, batch_size)

        # relA 形状: (series_num, series_num)
        relA_np = relA.detach().cpu().numpy()
        series_num = relA_np.shape[0]

        # relK 形状处理 - 需要转换为 (series_num, series_num)
        relK_np = relK.detach().cpu().numpy()

        # 根据 relK 的维度进行不同处理
        if relK_np.ndim == 5:
            # (n_head, series_num, series_num, time_step, time_step)
            causal_kernel = relK_np.mean(axis=(0, 3, 4))
        elif relK_np.ndim == 4:
            # (series_num, series_num, time_step, time_step)
            causal_kernel = relK_np.mean(axis=(2, 3))
        elif relK_np.ndim == 3:
            # (series_num, series_num, time_step)
            causal_kernel = relK_np.mean(axis=2)
        elif relK_np.ndim == 2:
            # 已经是 (series_num, series_num)
            causal_kernel = relK_np
        else:
            print(f"  警告: relK 维度异常 ({relK_np.ndim}), 仅使用 relA")
            causal_kernel = np.zeros_like(relA_np)

        # 确保形状匹配
        if causal_kernel.shape != relA_np.shape:
            print(f"  警告: causal_kernel shape {causal_kernel.shape} != relA shape {relA_np.shape}")
            print(f"  仅使用 relA 作为因果分数")
            combined_scores = relA_np
        else:
            # 结合注意力和卷积因果分数
            combined_scores = (relA_np + causal_kernel) / 2

        # 提取因果关系
        # 注意: relA[i][j] 表示 j → i 的因果分数 (变量 j 对变量 i 的影响)
        causal_relations = []

        for i in range(series_num):
            for j in range(series_num):
                score = combined_scores[i, j]  # j → i 的因果分数
                if score > threshold:
                    causal_relations.append({
                        'cause': var_names[j] if j < len(var_names) else f'Var_{j}',
                        'effect': var_names[i] if i < len(var_names) else f'Var_{i}',
                        'score': float(score)
                    })

        # 按分数排序
        causal_relations.sort(key=lambda x: x['score'], reverse=True)

        return {
            'relA': relA_np,
            'relK': relK_np,
            'combined_scores': combined_scores,
            'causal_relations': causal_relations
        }


# =============================================================================
# 因果分析辅助函数
# =============================================================================

def normalize_causal_scores(causal_edges):
    """
    将因果分数归一化到 0-1 范围
    
    Args:
        causal_edges: 因果边列表
    
    Returns:
        causal_edges: 归一化后的因果边列表
    """
    if len(causal_edges) == 0:
        return causal_edges

    scores = [edge['score'] for edge in causal_edges]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        # 所有分数相同，设为 1.0
        for edge in causal_edges:
            edge['score'] = 1.0
            edge['score_normalized'] = 1.0
    else:
        for edge in causal_edges:
            normalized = (edge['score'] - min_score) / (max_score - min_score)
            edge['score_original'] = edge['score']  # 保留原始分数
            edge['score'] = round(normalized, 4)  # 归一化分数

    return causal_edges


def compute_lag(relK, j, time_step):
    """
    计算时间滞后
    
    Args:
        relK: (series_num, time_step) 数组
        j: 原因变量索引
        time_step: 时间步数
    
    Returns:
        lag: 时间滞后
    """
    try:
        if isinstance(relK, np.ndarray) and relK.ndim == 2 and j < relK.shape[0]:
            relK_j = relK[j]
            if len(relK_j) > 0 and np.sum(relK_j) > 0:
                indices = np.argsort(-1 * relK_j)
                return max(0, time_step - 1 - indices[0])
    except:
        pass
    return 0
