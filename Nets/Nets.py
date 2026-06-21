"""
神经网络架构模块
包含 CausalFormer 模型的所有网络层和组件
"""
import torch
import torch.nn as nn
import math
import os
from abc import abstractmethod

def custom_repr(self):
    return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr


# 从 RRP 模块导入必要的组件
from RRP.RRP import (
    RelProp, Linear, LayerNorm, Dropout, LeakyReLU,
    Softmax, Clone, einsum, GELU
)


def prepare_device(n_gpu_use):
    """准备设备 (GPU 或 CPU)"""
    if str(os.environ.get("DTLN_FORCE_CPU", "")).lower() in {"1", "true", "yes", "y", "on"}:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    if n_gpu_use == 0:
        return torch.device('cuda:0')
    elif n_gpu_use == 1:
        return torch.device('cuda:1')
    return torch.device("cpu")
 


class BaseModel(nn.Module):
    """所有模型的基类"""
    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError


class GeoPositionalEncoding(BaseModel):
    """
    地理位置编码层
    将经纬度坐标编码为可学习的嵌入向量，融合到模型中
    
    支持两种编码方式：
    1. 正弦位置编码 (类似 Transformer)
    2. 可学习嵌入 (MLP)
    """
    def __init__(self, d_model, max_lat=90, max_lon=180, encoding_type='learnable'):
        super().__init__()
        self.d_model = d_model
        self.max_lat = max_lat
        self.max_lon = max_lon
        self.encoding_type = encoding_type
        
        if encoding_type == 'learnable':
            # 可学习的位置编码 MLP
            self.coord_encoder = nn.Sequential(
                nn.Linear(2, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model)
            )
        else:
            # 正弦位置编码
            self.register_buffer('div_term', 
                torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)))
    
    def forward(self, lat, lon):
        """
        生成位置编码
        
        Args:
            lat: 纬度 (batch,) 或标量
            lon: 经度 (batch,) 或标量
        
        Returns:
            pos_encoding: (batch, d_model) 位置编码向量
        """
        # 归一化到 [-1, 1]
        lat_norm = lat / self.max_lat
        lon_norm = lon / self.max_lon
        
        if self.encoding_type == 'learnable':
            # 拼接归一化坐标
            if lat_norm.dim() == 0:
                coords = torch.stack([lat_norm, lon_norm]).unsqueeze(0)  # (1, 2)
            else:
                coords = torch.stack([lat_norm, lon_norm], dim=-1)  # (batch, 2)
            return self.coord_encoder(coords)
        else:
            # 正弦位置编码
            batch_size = lat.shape[0] if lat.dim() > 0 else 1
            pe = torch.zeros(batch_size, self.d_model, device=lat.device)
            
            # 纬度编码 (偶数维度)
            pe[:, 0::4] = torch.sin(lat_norm.unsqueeze(-1) * self.div_term[::2])
            pe[:, 1::4] = torch.cos(lat_norm.unsqueeze(-1) * self.div_term[::2])
            # 经度编码 (奇数维度)
            pe[:, 2::4] = torch.sin(lon_norm.unsqueeze(-1) * self.div_term[1::2])
            pe[:, 3::4] = torch.cos(lon_norm.unsqueeze(-1) * self.div_term[1::2])
            
            return pe


# =============================================================================
# 辅助变量融合模块 (参考  n)
# 参考论文:
# - FiLM: Visual Reasoning with a General Conditioning Layer (AAAI 2018)
# - Conditional Neural Processes (ICML 2018)
# - CLIMAX: A Foundation Model for Weather and Climate (ICML 2023)
# =============================================================================

class FiLM(BaseModel):
    """
    FiLM (Feature-wise Linear Modulation) 层
    
    通过辅助信息生成 gamma 和 beta 来调制主特征
    公式: y = gamma * x + beta
    
    参考: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018
    
    注意：使用 RRP 兼容的层 (Linear, GELU) 以确保因果分析正确性
    """
    def __init__(self, cond_dim, feature_dim):
        """
        Args:
            cond_dim: 条件信息的维度
            feature_dim: 被调制特征的维度
        """
        super().__init__()
        # 使用 RRP 兼容的层，使用 GELU 激活函数
        self.gamma_fc = nn.Sequential(
             Linear(cond_dim, 256),
             GELU(),
             LayerNorm(256),    
             Linear(256, 256),
             GELU(),
             LayerNorm(256),
             Linear(256, feature_dim),
        )
        
        self.beta_fc = nn.Sequential(
             Linear(cond_dim, 256),
             GELU(),
             LayerNorm(256),    
             Linear(256, 256),
             GELU(),
             LayerNorm(256),
             Linear(256, feature_dim),
        )
        
        # 初始化为恒等变换 (gamma=1, beta=0)
        # 只初始化最后一层，使初始输出接近恒等映射
        nn.init.ones_(self.gamma_fc[-1].weight)
        nn.init.zeros_(self.gamma_fc[-1].bias)
        nn.init.zeros_(self.beta_fc[-1].weight)
        nn.init.zeros_(self.beta_fc[-1].bias)
    
    def forward(self, x, cond):
        """
        Args:
            x: 主特征 (..., feature_dim)
            cond: 条件信息 (batch, cond_dim) 或 (batch, time_step, cond_dim)
        
        Returns:
            调制后的特征 (..., feature_dim)
        """
        gamma = self.gamma_fc(cond)  # (batch, ..., feature_dim)
        beta = self.beta_fc(cond)    # (batch, ..., feature_dim)
        
        # 确保 gamma, beta 与 x 的维度匹配
        while gamma.dim() < x.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        
        return gamma * x + beta


class StaticEncoder(BaseModel):
    """
    静态变量编码器
    
    将静态变量（如 DEM）编码为固定维度的嵌入向量
    这些变量不随时间变化，作为全局条件信息
    """
    def __init__(self, static_dim, d_model, hidden_dim=None):
        """
        Args:
            static_dim: 静态变量的数量
            d_model: 输出嵌入维度
            hidden_dim: 隐藏层维度 (默认为 d_model // 2)
        """
        super().__init__()
        hidden_dim = hidden_dim or d_model // 2
        
        self.encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, static_vars):
        """
        Args:
            static_vars: 静态变量 (batch, static_dim)
        
        Returns:
            static_embedding: (batch, d_model)
        """
        return self.encoder(static_vars)


class AuxiliaryTemporalEncoder(BaseModel):
    """
    辅助时间序列变量编码器
    
    将辅助时间序列变量（如 temperature_2m, LST）编码为上下文表示
    这些变量提供额外信息但不参与因果关系的分析
    
    参考: 
    - Perceiver: General Perception with Iterative Attention (ICML 2021)
    - Temporal Fusion Transformers (IJF 2021)
    """
    def __init__(self, aux_series_num, input_window, feature_dim, d_model, drop_prob=0.1):
        """
        Args:
            aux_series_num: 辅助变量数量
            input_window: 时间窗口长度
            feature_dim: 特征维度
            d_model: 输出嵌入维度
            drop_prob: Dropout 概率
        """
        super().__init__()
        self.aux_series_num = aux_series_num
        self.input_window = input_window
        self.feature_dim = feature_dim
        self.d_model = d_model
        
        # 时间序列编码: 将每个辅助变量编码为 d_model 维向量
        self.temporal_encoder = nn.Sequential(
            nn.Linear(input_window * feature_dim, d_model),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 跨变量注意力池化: 将多个辅助变量聚合为单个上下文向量
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.Tanh(),
            nn.Linear(d_model // 4, 1)
        )
        
        self.output_norm = nn.LayerNorm(d_model)
    
    def forward(self, aux_data):
        """
        Args:
            aux_data: 辅助时间序列数据 (batch, time_step, aux_series_num, feature_dim)
        
        Returns:
            aux_context: 辅助变量上下文 (batch, d_model)
        """
        batch_size = aux_data.shape[0]
        
        # 重排为 (batch, aux_series_num, time_step * feature_dim)
        aux_data = aux_data.permute(0, 2, 1, 3)  # (batch, aux_series_num, time_step, feature_dim)
        aux_data = aux_data.reshape(batch_size, self.aux_series_num, -1)
        
        # 编码每个辅助变量
        aux_encoded = self.temporal_encoder(aux_data)  # (batch, aux_series_num, d_model)
        
        # 注意力池化
        attn_weights = self.attention_pool(aux_encoded)  # (batch, aux_series_num, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # 加权求和
        aux_context = torch.sum(aux_encoded * attn_weights, dim=1)  # (batch, d_model)
        
        return self.output_norm(aux_context)


class AuxiliaryFusionModule(BaseModel):
    """
    辅助变量融合模块
    
    将静态变量和辅助时间序列变量融合到主模型中
    使用 FiLM 机制进行特征调制，不影响因果关系的分析
    
    设计原则:
    1. 辅助变量只影响特征表示，不参与因果注意力计算
    2. 通过 FiLM 提供"约束/边界条件"的作用
    3. 因果分析时只考虑主要预测变量 (PREDICTORS → TARGET)
    """
    def __init__(self, d_model, aux_series_num=0, static_dim=0, input_window=24, 
                 feature_dim=1, drop_prob=0.1):
        """
        Args:
            d_model: 模型维度
            aux_series_num: 辅助时间序列变量数量
            static_dim: 静态变量数量
            input_window: 时间窗口长度
            feature_dim: 特征维度
            drop_prob: Dropout 概率
        """
        super().__init__()
        self.d_model = d_model
        self.aux_series_num = aux_series_num
        self.static_dim = static_dim
        self.has_aux = aux_series_num > 0
        self.has_static = static_dim > 0
        
        # 辅助时间序列编码器
        if self.has_aux:
            self.aux_encoder = AuxiliaryTemporalEncoder(
                aux_series_num, input_window, feature_dim, d_model, drop_prob
            )
        
        # 静态变量编码器
        if self.has_static:
            self.static_encoder = StaticEncoder(static_dim, d_model)
        
        # 条件信息融合
        cond_dim = d_model * (int(self.has_aux) + int(self.has_static))
        if cond_dim > 0:
            # 压缩条件信息
            self.cond_fusion = nn.Sequential(
                nn.Linear(cond_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model)
            )
            
            # FiLM 调制层 (用于调制嵌入层输出)
            self.film_embed = FiLM(d_model, d_model)
            
            # FiLM 调制层 (用于调制 Encoder 层输出)
            self.film_encoder = FiLM(d_model, feature_dim)
    
    def encode_conditions(self, aux_data=None, static_vars=None):
        """
        编码辅助条件信息
        
        Args:
            aux_data: 辅助时间序列数据 (batch, time_step, aux_series_num, feature_dim)
            static_vars: 静态变量 (batch, static_dim)
        
        Returns:
            cond: 融合后的条件信息 (batch, d_model)
        """
        cond_parts = []
        
        if self.has_aux and aux_data is not None:
            aux_context = self.aux_encoder(aux_data)  # (batch, d_model)
            cond_parts.append(aux_context)
        
        if self.has_static and static_vars is not None:
            static_context = self.static_encoder(static_vars)  # (batch, d_model)
            cond_parts.append(static_context)
        
        if len(cond_parts) == 0:
            return None
        
        # 拼接并融合
        cond = torch.cat(cond_parts, dim=-1)  # (batch, cond_dim)
        cond = self.cond_fusion(cond)  # (batch, d_model)
        
        return cond
    
    def modulate_embedding(self, embedding, cond):
        """
        使用 FiLM 调制嵌入层输出
        
        Args:
            embedding: 嵌入表示 (batch, series_num, d_model)
            cond: 条件信息 (batch, d_model)
        
        Returns:
            调制后的嵌入 (batch, series_num, d_model)
        """
        if cond is None:
            return embedding
        
        # 扩展条件到每个序列变量
        cond_expanded = cond.unsqueeze(1)  # (batch, 1, d_model)
        return self.film_embed(embedding, cond_expanded)
    
    def modulate_encoder_output(self, encoder_out, cond):
        """
        使用 FiLM 调制 Encoder 层输出
        
        Args:
            encoder_out: Encoder 输出 (batch, series_num, time_step, feature_dim)
            cond: 条件信息 (batch, d_model)
        
        Returns:
            调制后的输出 (batch, series_num, time_step, feature_dim)
        """
        if cond is None:
            return encoder_out
        
        # 扩展条件到每个序列和时间步
        cond_expanded = cond.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, d_model)
        return self.film_encoder(encoder_out, cond_expanded)


class Embedding(BaseModel):
    """
    特征嵌入层 (支持地理位置编码)
    
    将时序特征嵌入并可选地融合地理位置信息
    """
    def __init__(self, series_num, input_window, feature_dim, d_model, drop_prob, device, 
                 use_geo_encoding=False):
        super().__init__()
        self.series_num = series_num
        self.input_window = input_window
        self.feature_dim = feature_dim
        self.use_geo_encoding = use_geo_encoding
        self.d_model = d_model
        
        self.feature_emb = Linear(
            in_features=self.input_window * self.feature_dim,
            out_features=d_model,
            bias=True
        )
        self.feature_emb.weight.data.normal_(
            0, math.sqrt(2.0 / (self.input_window * self.feature_dim + d_model))
        )
        self.norm = LayerNorm(d_model)
        self.drop_out = Dropout(drop_prob)
        
        # 地理位置编码器
        if use_geo_encoding:
            self.geo_encoder = GeoPositionalEncoding(d_model, encoding_type='learnable')
            # 融合层：将特征嵌入与地理编码融合
            self.fusion_layer = Linear(d_model * 2, d_model, bias=True)
            self.fusion_norm = LayerNorm(d_model)

    def forward(self, x, lat=None, lon=None):
        """
        前向传播
        
        Args:
            x: 输入特征 (batch, series_num, input_window, feature_dim)
            lat: 纬度 (batch,) - 可选
            lon: 经度 (batch,) - 可选
        
        Returns:
            embedding: (batch, series_num, d_model)
        """
        x = x.reshape(-1, self.series_num, self.input_window * self.feature_dim)
        embedding = self.feature_emb(x)  # (batch, series_num, d_model)
        
        # 融合地理位置编码
        if self.use_geo_encoding and lat is not None and lon is not None:
            geo_encoding = self.geo_encoder(lat, lon)  # (batch, d_model)
            # 扩展到每个序列变量
            geo_encoding = geo_encoding.unsqueeze(1).expand(-1, self.series_num, -1)
            # 拼接并融合
            combined = torch.cat([embedding, geo_encoding], dim=-1)  # (batch, series_num, d_model*2)
            embedding = self.fusion_layer(combined)  # (batch, series_num, d_model)
            embedding = self.fusion_norm(embedding)
        
        return self.drop_out(self.norm(embedding))


class CausalConv(BaseModel):
    """因果卷积层 - 用于捕获时间滞后关系"""
    def __init__(self, series_num, input_window, n_head, device):
        super().__init__()
        self.series_num = series_num
        self.input_window = input_window
        self.n_head = n_head
        self.device = device
        
        # 可学习的卷积核参数
        self.K = nn.Parameter(
            torch.ones((n_head, series_num, series_num, input_window), dtype=torch.float)
        )
        self.mul = einsum('hxyji,bxif->bhxyjf')
        self.base = torch.tensor([i for i in range(1, input_window + 1)]).reshape(
            1, 1, 1, 1, -1, 1
        ).to(device)
        
        # 用于保存权重、梯度和相关性
        self.wgt = None
        self.grad = None
        self.rel = None

    def save_wgt(self, wgt):
        self.wgt = wgt

    def save_grad(self, grad):
        self.grad = grad

    def save_rel(self, rel):
        self.rel = rel

    def get_wgt(self):
        return self.wgt

    def get_grad(self):
        return self.grad

    def get_rel(self):
        return self.rel

    def forward(self, x):
        # 构造因果卷积核
        kernel = []
        for i in range(self.input_window):
            shifted = torch.roll(self.K, i + 1, dims=3)
            kernel.append(shifted)
        kernel = torch.stack(kernel).permute(1, 2, 3, 0, 4)
        kernel = torch.tril(kernel, diagonal=0)  # 确保因果性
        
        kernel.requires_grad_()
        self.save_wgt(kernel)
        kernel.register_hook(self.save_grad)
        
        # 应用卷积
        x = self.mul([kernel, x]) / self.base
        
        # 处理对角线元素（自身影响）
        for i in range(self.series_num):
            x[:, :, i, i, :, :] = x[:, :, i, i, :, :].roll(1, dims=2)
            x[:, :, i, i, 0, :] *= 0
        
        return x

    def relprop(self, rel):
        """反向传播相关性"""
        for i in range(self.series_num):
            rel[:, :, i, i, :, :] = rel[:, :, i, i, :, :].roll(-1, dims=2)
        rel = rel * self.base
        rel_k, rel_x = self.mul.relprop(rel)
        self.save_rel(rel_k)
        return rel_x

    def regularization(self):
        """正则化损失"""
        return torch.mean(torch.norm(self.K, dim=-1, p=1))


class MultiVariateCausalAttention(BaseModel):
    """多变量因果注意力机制"""
    def __init__(self, series_num, input_window, feature_dim, d_model, n_head, tau, device):
        super().__init__()
        self.series_num = series_num
        self.input_window = input_window
        self.d_tensor = d_model // n_head
        self.n_head = n_head
        self.tau = tau
        
        self.qk_mul = einsum('bhid,bhdj->bhij')
        self.softmax = Softmax(dim=-1)
        
        # 可学习的掩码矩阵 - 用于建模变量间因果关系
        self.mask = nn.Parameter(
            torch.ones((n_head, series_num, series_num), dtype=torch.float)
        )
        self.hardmard_product = einsum('hij,bhij->bhij')
        self.mul = einsum('bhij,bhjitf->bhitf')
        
        self.wgt = None
        self.grad = None
        self.rel = None

    def save_wgt(self, satt):
        self.wgt = satt

    def save_grad(self, grad):
        self.grad = grad

    def save_rel(self, rel):
        self.rel = rel

    def get_wgt(self):
        return self.wgt

    def get_grad(self):
        return self.grad

    def get_rel(self):
        return self.rel

    def forward(self, q, k, v):
        # 计算注意力分数
        score = self.qk_mul([q, k.transpose(2, 3)]) / math.sqrt(
            self.input_window * self.d_tensor
        )
        
        # 应用可学习掩码
        A = self.hardmard_product([self.mask, score])
        A = self.softmax(A / self.tau)
        
        A.requires_grad_()
        self.save_wgt(A)
        A.register_hook(self.save_grad)
        
        return self.mul([A, v])

    def relprop(self, rel):
        """反向传播相关性"""
        rel_A, rel_v = self.mul.relprop(rel)
        self.save_rel(rel_A)
        rel_score = self.softmax.relprop(rel_A)
        rel_mask, rel_score = self.hardmard_product.relprop(rel_score)
        rel_score *= math.sqrt(self.input_window * self.d_tensor)
        rel_q, rel_k = self.qk_mul.relprop(rel_score)
        return rel_q, rel_k.transpose(2, 3), rel_v

    def regularization(self):
        """正则化损失"""
        return torch.mean(torch.norm(self.mask, dim=-1, p=1))


class MultiHeadAttention(BaseModel):
    """多头注意力层"""
    def __init__(self, series_num, input_window, feature_dim, d_model, n_head, tau, device):
        super().__init__()
        self.n_head = n_head
        self.series_num = series_num
        self.input_window = input_window
        self.feature_dim = feature_dim
        
        self.attention = MultiVariateCausalAttention(
            series_num, input_window, feature_dim, d_model, n_head, tau, device
        )
        self.Wq = Linear(d_model, d_model, bias=True)
        self.Wk = Linear(d_model, d_model, bias=True)
        self.Wv = CausalConv(series_num, input_window, n_head, device)
        self.w_concat = Linear(n_head * feature_dim, feature_dim, bias=False)

    def forward(self, q, k, v):
        q, k, v = self.Wq(q), self.Wk(k), self.Wv(v)
        q, k = self.split(q), self.split(k)
        out = self.attention(q, k, v)
        out = self.concat(
            out.reshape(-1, self.n_head, self.series_num * self.input_window, self.feature_dim)
        )
        return self.w_concat(
            out.reshape(-1, self.series_num, self.input_window, self.n_head * self.feature_dim)
        )

    def split(self, tensor):
        """分割张量到多个头"""
        b, l, d = tensor.size()
        return tensor.view(b, l, self.n_head, d // self.n_head).transpose(1, 2)

    def concat(self, tensor):
        """合并多个头的输出"""
        b, h, l, d = tensor.size()
        return tensor.permute(0, 2, 1, 3).contiguous().view(b, l, h * d)

    def regularization(self):
        return self.attention.regularization() + self.Wv.regularization()

    def relprop(self, rel):
        """反向传播相关性"""
        rel = self.w_concat.relprop(rel)
        rel = self.split(
            rel.reshape(-1, self.series_num * self.input_window, self.n_head * self.feature_dim)
        )
        rel = rel.reshape(-1, self.n_head, self.series_num, self.input_window, self.feature_dim)
        rel_q, rel_k, rel_v = self.attention.relprop(rel)
        rel_q, rel_k = self.concat(rel_q), self.concat(rel_k)
        return self.Wq.relprop(rel_q), self.Wk.relprop(rel_k), self.Wv.relprop(rel_v)


class PositionwiseFeedForward(BaseModel):
    """位置前馈神经网络"""
    def __init__(self, dim, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = Linear(dim, hidden, bias=True)
        self.linear2 = Linear(hidden, dim, bias=True)
        self.activation = LeakyReLU()
        self.dropout = Dropout(drop_prob)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

    def relprop(self, rel):
        """反向传播相关性"""
        return self.linear1.relprop(
            self.activation.relprop(self.dropout.relprop(self.linear2.relprop(rel)))
        )


class EncoderLayer(BaseModel):
    """编码器层"""
    def __init__(self, series_num, input_window, feature_dim, d_model, n_head, 
                 ffn_hidden, drop_prob, tau, device):
        super().__init__()
        self.qk = Clone()
        self.attention = MultiHeadAttention(
            series_num, input_window, feature_dim, d_model, n_head, tau, device
        )
        self.norm1 = LayerNorm([input_window, feature_dim])
        self.dropout1 = Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(feature_dim, ffn_hidden, drop_prob)
        self.norm2 = LayerNorm([input_window, feature_dim])
        self.dropout2 = Dropout(drop_prob)

    def forward(self, x_embedding, x):
        q, k = self.qk(x_embedding, 2)
        x = self.dropout1(self.norm1(self.attention(q=q, k=k, v=x)))
        x = self.dropout2(self.norm2(self.ffn(x)))
        return x

    def regularization(self):
        return self.attention.regularization()

    def relprop(self, rel):
        """反向传播相关性"""
        rel = self.norm2.relprop(self.dropout2.relprop(rel))
        rel = self.ffn.relprop(rel)
        rel = self.norm1.relprop(self.dropout1.relprop(rel))
        rel_q, rel_k, rel_v = self.attention.relprop(rel)
        return self.qk.relprop((rel_q, rel_k)), rel_v


class Encoder(BaseModel):
    """
    编码器 - 堆叠多个编码器层
    支持地理位置编码融合
    支持辅助变量 (AUX_PREDICTORS 和 STATIC_VARS) 的 FiLM 调制
    """
    def __init__(self, series_num, input_window, feature_dim, d_model, n_head, 
                 n_layers, ffn_hidden, drop_prob, tau, device, use_geo_encoding=False,
                 aux_series_num=0, static_dim=0):
        super().__init__()
        self.use_geo_encoding = use_geo_encoding
        self.has_aux = aux_series_num > 0 or static_dim > 0
        
        self.emb = Embedding(series_num, input_window, feature_dim, d_model, drop_prob, device,
                            use_geo_encoding=use_geo_encoding)
        self.layers = nn.ModuleList([
            EncoderLayer(
                series_num, input_window, feature_dim, d_model, n_head,
                ffn_hidden, drop_prob, tau, device
            )
            for _ in range(n_layers)
        ])
        
        # 辅助变量融合模块
        if self.has_aux:
            self.aux_fusion = AuxiliaryFusionModule(
                d_model=d_model,
                aux_series_num=aux_series_num,
                static_dim=static_dim,
                input_window=input_window,
                feature_dim=feature_dim,
                drop_prob=drop_prob
            )

    def forward(self, x, lat=None, lon=None, aux_data=None, static_vars=None):
        """
        前向传播
        
        Args:
            x: 输入张量 (batch, series_num, time_step, feature_dim)
            lat: 纬度 (batch,) - 可选
            lon: 经度 (batch,) - 可选
            aux_data: 辅助时间序列变量 (batch, time_step, aux_series_num, feature_dim) - 可选
            static_vars: 静态变量 (batch, static_dim) - 可选
        """
        embedding = self.emb(x, lat, lon)
        
        # 如果有辅助变量，编码条件信息并调制嵌入
        cond = None
        if self.has_aux:
            cond = self.aux_fusion.encode_conditions(aux_data, static_vars)
            embedding = self.aux_fusion.modulate_embedding(embedding, cond)
        self.last_cond = cond
        
        for layer in self.layers:
            x = layer(embedding, x)
        
        # 如果有辅助变量，调制 Encoder 输出
        if self.has_aux and cond is not None:
            x = self.aux_fusion.modulate_encoder_output(x, cond)
        
        return x

    def regularization(self):
        return sum([layer.regularization() for layer in self.layers]) / len(self.layers)

    def relprop(self, rel):
        """
        反向传播相关性
        
        注意: RRP 只对主要变量 (PREDICTORS) 进行因果分析
        辅助变量通过 FiLM 条件化 forward 激活，但不作为 RRP 因果图节点。
        这里仅在核心 encoder layers 内部进行 RRP，并按网络反向拓扑逆序传播。
        """
        for layer in reversed(self.layers):
            emb_rel, rel = layer.relprop(rel)
        return rel


class PredictModel(BaseModel):
    """
    CausalFormer 预测模型 (支持地理位置编码和辅助变量融合)
    用于多变量时间序列预测的因果感知 Transformer 模型
    
    功能特性：
    1. 支持经纬度位置编码融合，使模型学习地理空间信息
    2. 支持辅助时间序列变量 (AUX_PREDICTORS): 如 temperature_2m, LST
    3. 支持静态变量 (STATIC_VARS): 如 DEM
    4. 辅助变量通过 FiLM 机制融合，不影响 PREDICTORS → TARGET 的因果分析
    
    设计原则：
    - PREDICTORS: 主要预测变量，参与因果发现
    - TARGET: 目标变量
    - AUX_PREDICTORS: 辅助时间序列变量，提供上下文信息，不参与因果分析
    - STATIC_VARS: 静态变量，作为全局条件，不参与因果分析
    """
    def __init__(self, config, d_model, n_head, n_layers, ffn_hidden, drop_prob, tau, 
                 use_geo_encoding=False, aux_series_num=0, static_dim=0):
        """
        Args:
            config: 配置字典
            d_model: 模型维度
            n_head: 注意力头数
            n_layers: 编码器层数
            ffn_hidden: FFN 隐藏层维度
            drop_prob: Dropout 概率
            tau: 温度参数
            use_geo_encoding: 是否使用地理位置编码
            aux_series_num: 辅助时间序列变量数量 (AUX_PREDICTORS)
            static_dim: 静态变量数量 (STATIC_VARS)
        """
        super().__init__()
        self.args = config['data_loader']['args']
        self.input_window = self.args['time_step']
        self.output_window = self.args['output_window']
        self.series_num = self.args['series_num']
        self.feature_dim = self.args['feature_dim']
        self.device = prepare_device(config['n_gpu'])
        self.use_geo_encoding = use_geo_encoding
        self.aux_series_num = aux_series_num
        self.static_dim = static_dim
        
        self.encoder = Encoder(
            self.series_num, self.input_window, self.feature_dim,
            d_model, n_head, n_layers, ffn_hidden, drop_prob, tau, self.device,
            use_geo_encoding=use_geo_encoding,
            aux_series_num=aux_series_num,
            static_dim=static_dim
        )
        self.fc = Linear(self.feature_dim, self.args['output_dim'], bias=True)

    def forward(self, x, lat=None, lon=None, aux_data=None, static_vars=None):
        """
        前向传播
        
        Args:
            x: 主输入张量 [batch, time_step, series_num, feature_dim]
               包含 PREDICTORS 和 TARGET 变量
            lat: 纬度 (batch,) - 可选，当 use_geo_encoding=True 时使用
            lon: 经度 (batch,) - 可选，当 use_geo_encoding=True 时使用
            aux_data: 辅助时间序列数据 [batch, time_step, aux_series_num, feature_dim] - 可选
                     包含 AUX_PREDICTORS 变量，如 temperature_2m, LST, total_evaporation_sum
            static_vars: 静态变量 [batch, static_dim] - 可选
                        包含 STATIC_VARS 变量，如 DEM
        
        Returns:
            output: 预测输出 [batch, output_window, series_num, output_dim]
        """
        x = x.permute(0, 2, 1, 3)  # [batch, series_num, time_step, feature_dim]
        out = self.encoder(x, lat, lon, aux_data, static_vars)
        out = self.fc(out)
        return out.permute(0, 2, 1, 3)[:, -self.output_window:, ...]

    def regularization(self):
        """计算正则化损失"""
        return self.encoder.regularization()

    def relprop(self, rel):
        """
        反向传播相关性 (RRP)
        
        重要说明:
        - RRP 只对主要变量 (PREDICTORS) 进行因果分析
        - 辅助变量 (AUX_PREDICTORS, STATIC_VARS) 不参与 RRP 计算
        - 这确保了因果发现的纯净性，只关注 PREDICTORS → TARGET 的关系
        
        Args:
            rel: 相关性张量 [batch, output_window, series_num, output_dim]
        
        Returns:
            rel: 输入相关性 [batch, time_step, series_num, feature_dim]
        """
        pad = torch.zeros(
            (rel.shape[0], self.input_window - self.output_window, rel.shape[2], rel.shape[3])
        ).to(self.device)
        rel = torch.cat((pad, rel), 1).permute(0, 2, 1, 3)
        rel = self.encoder.relprop(self.fc.relprop(rel))
        return rel.permute(0, 2, 1, 3)
