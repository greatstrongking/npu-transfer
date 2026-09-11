#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from ml_dtypes import bfloat16

def impl(x, target, weight=None, reduction="mean", ignore_index=-100):
    orig_dtype = x.dtype
    x_f32 = x.astype(np.float32)
    target_int = target.astype(np.int64)
    num_class = x_f32.shape[-1]
    N = x_f32.shape[0]

    # 1. 初始化类别权重
    if weight is None:
        w_cls = np.ones(num_class, dtype=np.float32)
    else:
        w_cls = weight.astype(np.float32)

    # 2. 取出每个样本对应类别的log_prob x_{n,y_n}
    target_expand = target_int[..., np.newaxis]
    x_select = np.take_along_axis(x_f32, target_expand, axis=-1).squeeze(-1)

    # 3. 每个样本对应权重 w_{y_n}
    w_sample = np.take(w_cls, target_int)

    # 4. ignore_index掩码：ignore位置权重置0
    valid_mask = target_int != ignore_index
    w_sample[~valid_mask] = 0.0

    # 5. 逐元素loss l_n = -w_{y_n} * x_{n,y_n}
    loss_elem = -1.0 * w_sample * x_select

    # 6. 计算totalWeight = sum(w_sample)
    total_weight_val = np.sum(w_sample)

    # 7. 按reduction分支计算输出loss
    if reduction == "none":
        loss_out = loss_elem
    elif reduction == "sum":
        loss_out = np.array([np.sum(loss_elem)], dtype=np.float32)
    elif reduction == "mean":
        if total_weight_val == 0.0:
            loss_out = np.array([0.0], dtype=np.float32)
        else:
            loss_out = np.array([np.sum(loss_elem) / total_weight_val], dtype=np.float32)
    else:
        raise ValueError(f"unsupported reduction: {reduction}, only none/sum/mean")

    total_weight_out = np.array([total_weight_val], dtype=np.float32)

    # 8. 还原原始输入精度（保持原有逻辑）
    if orig_dtype == np.float16:
        loss_out = loss_out.astype(np.float16)
        total_weight_out = total_weight_out.astype(np.float16)
    elif orig_dtype == bfloat16:
        loss_out = loss_out.astype(bfloat16)
        total_weight_out = total_weight_out.astype(bfloat16)

    return loss_out, total_weight_out


if __name__ == "__main__":
    # 清理bin文件
    for f in glob.glob("*.bin"):
        os.remove(f)
    
    # 从 JSON 第一个 case 获取参数
    d_type = "float16"
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": bfloat16,
        "float64": np.float64,
        "int8": np.int8,
        "int16": np.int16,
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "uint16": np.uint16,
        "uint32": np.uint32,
        "uint64": np.uint64,
        "bool": np.bool_,
        "fp8_e4m3fn": np.uint8,
        "fp8_e5m2": np.uint8,
    }
    np_type = d_type_dict[d_type]
    
    # 生成输入数据
    input_x = np.ones((3, 5)).astype(d_type_dict["float16"])
    input_target = np.ones((3)).astype(d_type_dict["int32"])
    input_weight = np.ones((5)).astype(d_type_dict["float16"])
    attr_reduction = "mean"
    attr_ignore_index = -100
    
    # 计算 golden 数据
    golden = impl(input_x, input_target, input_weight, attr_reduction, attr_ignore_index)
    
    # 保存数据到文件
    input_x.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_nll_loss_x.bin")
    input_target.astype(d_type_dict["int32"]).tofile(f"{d_type}_input_nll_loss_target.bin")
    input_weight.astype(d_type_dict["float16"]).tofile(f"{d_type}_input_nll_loss_weight.bin")
    if golden is not None:
        if isinstance(golden, (list, tuple)):
            _out_dtypes = ["float16", "float16"]
            for _gi, _g in enumerate(golden):
                _dt = _out_dtypes[_gi] if _gi < len(_out_dtypes) else _out_dtypes[-1]
                _g.astype(d_type_dict[_dt]).tofile(f"{_dt}_golden_nll_loss_{_gi}.bin")
        else:
            golden.astype(d_type_dict["float16"]).tofile("float16_golden_nll_loss_0.bin")
    
    print(f"生成完成: dtype={d_type}")
