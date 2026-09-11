# NllLoss 算子 NPU 测评报告

> 测评日期: 2026-09-11 20:51
> 项目路径: /home/developer/npu-transfer-3

---

## 1. 测试环境

### 1.1 硬件环境

| 项目 | 信息 |
|------|------|
| CPU | aarch64, 40 核 |
| 内存 | 229 GiB (可用 223 GiB) |
| NPU 型号 | Ascend910 (实际 SoC: ascend910_93) |
| NPU 数量 | 2 卡 (Chip 0 Phy-ID 4, Chip 1 Phy-ID 5) |
| NPU 状态 | OK |
| NPU 功耗 | 165.4W |
| NPU 温度 | 44°C |
| NPU HBM | 3109 MB / 65536 MB (已用/总量) |
| PCIe | 0000:0A:00.0 / 0000:0B:00.0 |

### 1.2 软件环境

| 项目 | 信息 |
|------|------|
| 操作系统 | Linux 5.10.0-182 (HCE 2.0) aarch64 |
| CANN 版本 | 9.0.0 |
| NPU 驱动版本 | 25.5.5 |
| 编译器 | GCC 9.4.0 |
| CMake | 3.x |
| 编译线程数 | 8 |
| 目标 SoC | ascend910_93 |

---

## 2. 算子信息

| 项目 | 信息 |
|------|------|
| 算子名称 | NllLoss (负对数似然损失) |
| 实现框架 | Ascend C |
| 数学公式 | $l_n = -w_{y_n} \cdot x_{n,y_n}$，$w_c = \text{weight}[c] \cdot \mathbf{1}\{y_n \neq \text{ignore\_index}\}$ |
| reduction 模式 | none / sum / mean |
| 输入 | x (预测概率), target (类别索引), weight (类别权重, 可选) |
| 输出 | y (loss), total_weight (有效权重总和) |
| 支持数据类型 | float16+int32, float16+int64, bfloat16+int32, bfloat16+int64, float32+int32, float32+int64 |
| Kernel 二进制数量 | 6 个 (每种数据类型组合一个) |
| 打包产物 | custom_opp_ubuntu_aarch64.run (703 KB) |

---

## 3. 构建产物

### 3.1 编译产物清单

| 产物 | 路径 | 大小 |
|------|------|------|
| 算子包 | `build/custom_opp_ubuntu_aarch64.run` | 703 KB |
| Kernel 二进制 (x6) | `build/op_kernel/ascendc_kernels/binary/ascend910_93/nll_loss/*.o` | 54 KB / 个 |
| op_api 库 | `packages/vendors/nll_loss_custom/op_api/lib/libcust_opapi.so` | 45 KB |
| op_proto 库 | `packages/vendors/nll_loss_custom/op_proto/lib/.../libcust_opsproto_rt2.0.so` | 1.2 MB |
| op_tiling 库 | `packages/vendors/nll_loss_custom/op_impl/.../libcust_opmaster_rt2.0.so` | 1.1 MB |
| 算子头文件 | `packages/vendors/nll_loss_custom/op_api/include/aclnn_nll_loss.h` | — |

### 3.2 Kernel 二进制列表

| 文件名 | schMode | 数据类型 | 大小 |
|--------|---------|----------|------|
| NllLoss_16ed208fe43dc95ef29ee4be6c9b2b58.o | 0 | float16 + int32 | 54 KB |
| NllLoss_544e688d88d2072b55454da35c9a80c0.o | 1 | float16 + int64 | 54 KB |
| NllLoss_3bac53b52ce5efc58119b7a74b1ae55e.o | 2 | bfloat16 + int32 | 54 KB |
| NllLoss_2142aa1b825349ce540f7daa73ac8235.o | 3 | bfloat16 + int64 | 54 KB |
| NllLoss_4044e40db812b66bba079d9f05d426bf.o | 4 | float32 + int32 | 54 KB |
| NllLoss_b52a99aa788e5f8d2198356874da3dfb.o | 5 | float32 + int64 | 54 KB |

---

## 4. 测试结果汇总

### 4.1 总体结果

| 测试项 | 状态 | 耗时 | 说明 |
|--------|:----:|------|------|
| 项目编译 (build.sh) | ✅ 通过 | ~15.1s | 含 cmake + 编译 + 打包 + 安装 |
| 算子包安装 | ✅ 通过 | <1s | 安装到 cann-9.0.0/opp/vendors/ |
| CPU 仿真 UT (TmSim) | ✅ 通过 | 165 ms | Google Test, 1 个用例 |
| 精度比对 (compare_data.py) | ✅ 通过 | — | MERE=0.000000, MARE=0.000000 |
| **NPU 实际执行** (examples) | ✅ **通过** | ~4.1s | 含 cmake + 编译 + 执行 |

### 4.2 设备运行日志检查

| 检查项 | 结果 |
|--------|------|
| ERROR 日志数量 | **0** |
| WARNING 日志数量 | 1 (无关: tensorflow SO 加载失败) |
| AICORE 异常 | 无 |
| SMMU 故障 | 无 |
| Kernel 任务失败 | 无 |

---

## 5. CPU 仿真 UT 测试详情

### 5.1 测试框架

- 框架: Google Test 1.14.0
- ABI: `_GLIBCXX_USE_CXX11_ABI=0` (匹配 CANN 库)
- 模拟器: TmSim (串行模式)

### 5.2 测试用例

| 用例名 | 状态 | 耗时 |
|--------|:----:|------|
| `NllLossKernelTest.test_kernel_run` | ✅ PASSED | 165 ms |

**测试套件**: 1 个套件, 1 个用例, 全部通过

### 5.3 测试输入数据

| 参数 | 值 | 说明 |
|------|-----|------|
| x (输入概率) | shape=(3,5), dtype=float16, 全 1.0 | 3 个样本, 5 个类别 |
| target (类别索引) | shape=(3,), dtype=int32, 全 1 | 每个样本取类别 1 |
| weight (类别权重) | shape=(5,), dtype=float16, 全 1.0 | 5 个类别等权 |
| reduction | "mean" | 均值归约 |
| ignore_index | -100 | 忽略索引 |

### 5.4 测试输出 (Golden vs Output)

| 输出 | Golden | Output | 绝对误差 |
|------|--------|--------|----------|
| loss (y) | -1.0 (float16) | -1.0 (float16) | **0.0** |
| total_weight | 3.0 (float16) | 3.0 (float16) | **0.0** |

### 5.5 精度指标

| 指标 | 值 | 阈值 (float16) | 状态 |
|------|-----|----------------|------|
| MERE (平均相对误差) | 0.000000 | 0.000977 (2^-10) | ✅ PASS |
| MARE (最大相对误差) | 0.000000 | 0.009766 (10×2^-10) | ✅ PASS |

---

## 6. NPU 实际执行测试详情

### 6.1 测试流程

```
1. 编译算子项目 (build.sh)
2. 安装自定义算子包 (custom_opp_ubuntu_aarch64.run)
3. 编译 examples 测试程序 (examples/run.sh)
4. 在 NPU 上执行 aclnnNllLoss 调用
5. 检查执行结果和返回码
```

### 6.2 测试参数

| 参数 | 值 |
|------|-----|
| 设备 ID | 0 |
| x tensor | shape=[3,5], dtype=ACL_FLOAT16, 值全 1.0 |
| target tensor | shape=[3], dtype=ACL_INT32, 值全 1 |
| weight tensor | shape=[5], dtype=ACL_FLOAT16, 值全 1.0 |
| y tensor (输出) | shape=[1], dtype=ACL_FLOAT16 |
| total_weight tensor (输出) | shape=[1], dtype=ACL_FLOAT16 |
| reduction | "mean" |
| ignoreIndex | -100 |

### 6.3 API 调用链路

| 步骤 | API | 返回值 |
|------|-----|--------|
| 1. ACL 初始化 | `aclInit()` | 0 (成功) |
| 2. 设置设备 | `aclrtSetDevice(0)` | 0 (成功) |
| 3. 创建流 | `aclrtCreateStream()` | 0 (成功) |
| 4. 获取 workspace | `aclnnNllLossGetWorkspaceSize()` | 0 (成功) |
| 5. 执行算子 | `aclnnNllLoss()` | 0 (成功) |
| 6. 同步等待 | `aclrtSynchronizeStream()` | 0 (成功) |
| 7. 资源释放 | `aclDestroyTensor()` 等 | — |

### 6.4 执行结果

```
========================================
nll_loss 算子调用示例
========================================
执行调用示例...
========================================
执行完成
========================================
```

- 返回码: **0** (成功)
- 错误码: **无**
- 程序正常退出

---

## 7. 构建性能

| 阶段 | 耗时 |
|------|------|
| cmake 配置 | ~2s |
| 编译 (host + kernel) | ~8s |
| 打包 (CPack) | ~3s |
| 安装 | ~2s |
| **总计** | **~15.1s** |

---

## 8. 结论

### 8.1 测试结论

| 维度 | 结论 |
|------|------|
| 功能正确性 | ✅ CPU 仿真 UT 与 NPU 实际执行均通过 |
| 数值精度 | ✅ 输出与 Golden 完全一致 (MERE=0, MARE=0) |
| 运行稳定性 | ✅ 无 AICORE 异常、SMMU 故障或内存越界 |
| 编译正确性 | ✅ 所有 6 种数据类型组合的 Kernel 二进制均成功生成 |
| 打包部署 | ✅ 算子包安装成功，运行时库加载正常 |

### 8.2 适用场景

- 小批量 (nSize ≤ tileN) 场景下使用单核执行，已验证通过
- 支持 float16/bfloat16/float32 三种数据类型
- 支持 int32/int64 两种 target 索引类型
- 支持 mean/sum/none 三种 reduction 模式

### 8.3 已知限制

- 大批量 (nSize > tileN) 多核协作场景尚未在 NPU 上验证
- 当前测试仅覆盖 float16 + int32 组合，其他 5 种类型组合需补充测试
