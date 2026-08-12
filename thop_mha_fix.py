# -*- coding: utf-8 -*-
"""
thop 0.1.1 MHA 计数修复（Bug fix）
================================
thop 0.1.1 没有为 nn.MultiheadAttention 注册计数规则，profile() 会把整块标准
注意力记成 0 FLOPs / 0 Params（in_proj 是裸 Parameter，out_proj 虽是 nn.Linear
但 dfs_count 在 MHA 处不递归到子模块的 param 统计）。结果：凡用标准 MHA 的
模型（如 Light-TFT、Wave-Light-TFT）Params/FLOPs 都被严重低估。

本模块提供：
- count_multihead_attention：补全 MHA 的 MACs 钩子
- profile_fixed：FLOPs 含 MHA、Params 取 sum(p.numel()) 真实值的便捷函数

用法：
    from thop_mha_fix import profile_fixed
    flops_str, params_str, flops, params = profile_fixed(model, inputs)
"""
import torch
import torch.nn as nn
from thop import profile, clever_format


def count_multihead_attention(m, x, y):
    """nn.MultiheadAttention 的 MACs 钩子（自注意力，q=k=v）。

    标准自注意力一次前向：
        in_proj(q,k,v)  3 个 E×E 投影
        QK^T            B*N*N*E
        attn·V          B*N*N*E
        out_proj        1 个 E×E 投影
    合计 = 4*B*N*E*E + 2*B*N*N*E
    （num_heads 不改变参数量/MACs，只改变 head_dim，故公式与 heads 无关）
    """
    q = x[0]
    E = m.embed_dim
    if getattr(m, "batch_first", False):
        B, N = q.shape[0], q.shape[1]
    else:
        N, B = q.shape[0], q.shape[1]
    m.total_ops += torch.DoubleTensor([int(4 * B * N * E * E + 2 * B * N * N * E)])


# 供 profile(custom_ops=...) 直接使用
MHA_CUSTOM_OPS = {nn.MultiheadAttention: count_multihead_attention}


def profile_fixed(model, inputs, fmt="%.3f"):
    """修复后的 thop profile。

    - FLOPs：传入 custom_ops 补上 nn.MultiheadAttention；
    - Params：用 sum(p.numel()) 取真实值（thop 会漏算 MHA.in_proj 等裸 Parameter，
      以及 DynamicWeightModule.w_win 等无 hook 模块的参数）。

    Returns:
        (flops_str, params_str, flops, params)
    """
    flops, _ = profile(model, inputs=inputs, verbose=False, custom_ops=MHA_CUSTOM_OPS)
    params = sum(p.numel() for p in model.parameters())
    flops_str, params_str = clever_format([flops, params], fmt)
    return flops_str, params_str, flops, params
