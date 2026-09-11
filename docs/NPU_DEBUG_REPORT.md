# NllLoss 算子 NPU 执行故障分析与修复方向

> 文档用途: AI Agent 排查用。记录当前故障现象、已排除的因素、待验证假设、修复方案和验证方法。
> 最后更新: 2026-09-11

---

## 1. 基本信息

```yaml
算子名: NllLoss (负对数似然损失)
实现: Ascend C, 目标芯片 ascend910_93
CANN: 9.0.0
项目路径: /home/developer/npu-transfer-3
关键源文件:
  kernel 入口:   op_kernel/nll_loss.cpp
  kernel 实现:   op_kernel/nll_loss.h           # 所有计算逻辑在此
  tiling 数据:   op_kernel/nll_loss_tiling_data.h
  tiling key:    op_kernel/nll_loss_tiling_key.h
  host tiling:   op_host/nll_loss_tiling.cpp
  host infershape: op_host/nll_loss_infershape.cpp
  example 入口:  examples/test_aclnn_nll_loss.cpp
  编译脚本:      build.sh
```

---

## 2. 测试结果矩阵

| 测试阶段 | 状态 | 说明 |
|----------|:----:|------|
| 编译 build.sh | PASS | 生成 custom_opp_ubuntu_aarch64.run |
| 算子包安装 | PASS | 安装到 cann-9.0.0/opp/vendors/nll_loss_custom/ |
| CPU 仿真 UT (TmSim) | PASS | 1 test passed, 172ms |
| 精度比对 compare_data.py | PASS | MERE=0.0, MARE=0.0 |
| **NPU 实际执行** | **FAIL** | AICORE exception, retCode=0x26 |

**结论: 算法逻辑正确，问题出在 NPU 硬件执行阶段。**

---

## 3. 已排除的故障

### 3.1 SoC 类型不匹配 (已修复)

- `build.sh` 原默认 `COMPUTE_UNIT="ascend910b"`
- 当前 NPU 实际 SoC 为 `ascend910_93`
- 症状: `binary_info_config.json of socVersion [ascend910_93] does not support opType [NllLoss]`
- 修复: 已改为 `COMPUTE_UNIT="ascend910_93"`，重新编译安装后算子可被识别

---

## 4. 当前故障详情

### 4.1 错误码链路

```
aclnnNllLossGetWorkspaceSize → 成功 (ret=0)
aclnnNllLoss → 成功 (ret=0)
aclrtSynchronizeStream → 失败 (ret=507015)
  内部错误码: 0x7150026
  Kernel 返回码: 0x26 (aicore exception)
```

### 4.2 故障 Kernel

```
NllLoss_3bac53b52ce5efc58119b7a74b1ae55e_0
对应 schMode=0: half (float16) + int32
```

### 4.3 Kernel 参数快照 (从 Runtime 日志提取)

```
args[0]  x            : 0x3fffffb9000     (global memory)
args[1]  target       : 0x12c0c0015000    (global memory)
args[2]  weight       : 0x12c0c0016000    (global memory)
args[3]  y            : 0x12c0c0017000    (global memory)
args[4]  total_weight : 0x12c0c0018000    (global memory)
args[5]  workspace    : 0x12c0c0019000    (global memory)
args[6]  tiling       : 0x12c0c001a000    (global memory)
args[7]  tiling_data  : 0x12c1000000c8
--- tiling 字段 ---
args[8]  (tiling 值)  : 0x12c0c0013000
args[9]  tilingMode   : 1 (NORMAL)
args[10] needCoreNum  : 3
args[11] nSize        : 3
args[12] cSize        : 5
args[13] perCoreSize  : 1
args[14] lastCoreSize : 1
args[15] perCoreLoop  : 0
args[16] perCoreLeft  : 1
args[17] lastCoreLoop : 1
args[18] lastCoreLeft : 0
args[19] xUbElems     : 16 (0x10)
args[20] targetUbElems: 8  (0x8)
args[21] weightUbElems: 16 (0x10)
args[22] ignoreIndex  : -100 (0xffffffffffffff9c)
args[23] reduction    : 1 (MEAN)
args[24] hasWeight    : 1
args[25] tileN        : 1
args[26] xDimNum      : 2
tilingKey: 0, blockDim: 3
```

### 4.4 测试输入数据

```
x:          shape=[3,5], dtype=float16, 所有元素=1.0
target:     shape=[3],   dtype=int32,   所有元素=1
weight:     shape=[5],   dtype=float16, 所有元素=1.0
reduction:  "mean"
ignoreIndex: -100
```

**预期输出**: y=scalar(-1.0), total_weight=scalar(3.0)

---

## 5. 故障根因分析

### 5.1 假设 A: ReduceSum count 过小 (置信度: 高)

**位置**: `op_kernel/nll_loss.h:319, 324`
```cpp
ReduceSum(tmp, fpX, work, curN);   // curN=1
ReduceSum(tmp, fpW, work, curN);   // curN=1
```

**分析**:
- 当前 tileN=1, perCoreSize=1, 所以 curN=1
- Ascend C `ReduceSum(dst, src, sharedTmpBuffer, count)` 对 count 有最小元素数要求
- 在 ascend910_93 上，float 类型 ReduceSum 要求 count >= 8 (即 32 字节对齐)
- count=1 可能触发非法指令或未定义行为

**修复方向**: 将 count 对齐到 8 的倍数 (padding 元素已在 Duplicate 时清零)

### 5.2 假设 B: workBuf 空间不足 (置信度: 高)

**位置**: `op_kernel/nll_loss.h:186`
```cpp
pipe.InitBuffer(workBuf, fpBytes);  // fpBytes = targetUbElems(8) * sizeof(float)(4) = 32
```

**分析**:
- ReduceSum 的 `sharedTmpBuffer` 参数需要足够的临时空间
- 根据 Ascend C API，ReduceSum 内部需要的临时 buffer 大小取决于输入数据量和归约算法
- 32 字节 (8 个 float) 可能不够 ReduceSum 内部使用
- 尤其在 count 对齐后 (假设 A 修复后 count=8)，内部可能需要更多空间

**修复方向**: 增大 workBuf 到至少 256 字节 (64 个 float)

### 5.3 假设 C: Duplicate 元素数不对齐 (置信度: 中)

**位置**: `op_kernel/nll_loss.h:236-237` (GatherNormal)
```cpp
Duplicate(vx, static_cast<T>(0), alignN);  // alignN = AlignElems(curN=1, 16) = 16
Duplicate(vw, static_cast<T>(0), alignN);  // alignN = 16
```

**分析**:
- `alignN=16` 意味着 Duplicate 操作处理 16 个 half 元素 = 32 字节, 这是对齐的
- 但 `vx` 和 `vw` 来自 `validXBuf`/`validWBuf`, 大小为 `targetUbElems * sizeof(T) = 8 * 2 = 16` 字节
- **16 字节 buffer 上执行 32 字节 Duplicate → 越界写入!**

**这是最可能的根因。**

**修复方向**: 确保 validXBuf/validWBuf 尺寸 >= alignN * sizeof(T)

### 5.4 假设 D: InitBuffers 中 vBytes 过小 (置信度: 高, 与 C 关联)

**位置**: `op_kernel/nll_loss.h:161`
```cpp
int64_t vBytes = tiling_->targetUbElems * static_cast<int64_t>(sizeof(T));
// = 8 * 2 = 16 字节
```

**但 GatherNormal 中**:
```cpp
int64_t alignN = AlignElems(curN, static_cast<int64_t>(sizeof(T)));
// curN=1, sizeof(half)=2, BLOCK_BYTES=32, blockElems=32/2=16
// AlignElems(1, 16) = 16
// Duplicate 操作处理 16 * 2 = 32 字节
// 但 vBytes = 16 字节 → 越界!
```

**修复方向**: vBytes 应取 `max(targetUbElems * sizeof(T), alignN * sizeof(T))`

### 5.5 假设 E: SyncAll 时序 (置信度: 低)

**位置**: `op_kernel/nll_loss.h:452`
```cpp
WritePartialToWorkspace();
SyncAll();
ReduceAndWriteOutputs();
```

**分析**: 3 个 core 各自写完 workspace 后 SyncAll, 再由 core 0 归约。时序看起来正确, 但如果 WritePartialToWorkspace 内的 DataCopy 还没真正完成就 SyncAll, 可能导致 core 0 读到未写入的数据。

**修复方向**: 在 WritePartialToWorkspace 末尾添加 WaitSToMte3()

---

## 6. 推荐修复方案 (按优先级排列)

### 修复 1 (最高优先级): 修正 InitBuffers 中 buffer 尺寸计算

**文件**: `op_kernel/nll_loss.h`
**函数**: `InitBuffers()` (行 156-187)

**当前代码问题**:
```cpp
// 行 161
int64_t vBytes = tiling_->targetUbElems * static_cast<int64_t>(sizeof(T));
// targetUbElems=8, sizeof(half)=2 → vBytes=16
// 但 GatherNormal 中 AlignElems(1, 16)=16 → Duplicate 操作 16*2=32 字节 → 越界
```

**修复方案**:
```cpp
template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::InitBuffers()
{
    int64_t xBytes = tiling_->xUbElems * static_cast<int64_t>(sizeof(T));
    int64_t tBytes = tiling_->targetUbElems * static_cast<int64_t>(sizeof(TargetT));
    int64_t wBytes = tiling_->weightUbElems * static_cast<int64_t>(sizeof(T));
    int64_t vBytes = tiling_->targetUbElems * static_cast<int64_t>(sizeof(T));
    int64_t fpBytes = tiling_->targetUbElems * static_cast<int64_t>(sizeof(float));

    // 确保所有 buffer 至少 BLOCK_BYTES 大小
    xBytes = std::max(xBytes, BLOCK_BYTES);
    tBytes = std::max(tBytes, BLOCK_BYTES);
    wBytes = std::max(wBytes, BLOCK_BYTES);
    vBytes = std::max(vBytes, BLOCK_BYTES);
    fpBytes = std::max(fpBytes, BLOCK_BYTES);

    // vBytes 还需要容纳 AlignElems 后的最大尺寸
    // 在 GatherNormal 中 alignN = AlignElems(curN, BLOCK_BYTES/sizeof(T))
    // 最大 alignN = AlignElems(tileN, BLOCK_BYTES/sizeof(T))
    int64_t maxAlignN = AlignElems(tiling_->tileN, BLOCK_BYTES / static_cast<int64_t>(sizeof(T)));
    int64_t requiredVBytes = maxAlignN * static_cast<int64_t>(sizeof(T));
    vBytes = std::max(vBytes, requiredVBytes);

    // fpBytes 需要容纳 ReduceSum 对齐后的 count
    // ReduceSum count 对齐到 8 的倍数
    int64_t alignedTileN = ((tiling_->tileN + 7) / 8) * 8;
    int64_t requiredFpBytes = alignedTileN * static_cast<int64_t>(sizeof(float));
    fpBytes = std::max(fpBytes, requiredFpBytes);

    // workBuf 需要足够 ReduceSum 内部使用
    int64_t workBytes = std::max(fpBytes, static_cast<int64_t>(256));

    pipe.InitBuffer(xBuf, xBytes);
    pipe.InitBuffer(targetBuf, tBytes);
    pipe.InitBuffer(weightBuf, wBytes);
    pipe.InitBuffer(validXBuf, vBytes);
    pipe.InitBuffer(validWBuf, vBytes);
    pipe.InitBuffer(fpXBuf, fpBytes);
    pipe.InitBuffer(fpWBuf, fpBytes);
    pipe.InitBuffer(tmpBuf, fpBytes);
    pipe.InitBuffer(workBuf, workBytes);
}
```

### 修复 2 (高优先级): ReduceSum count 对齐

**文件**: `op_kernel/nll_loss.h`
**函数**: `ComputeTile()` (行 296-340)

**当前代码** (行 319, 324):
```cpp
ReduceSum(tmp, fpX, work, curN);   // curN 可能为 1
ReduceSum(tmp, fpW, work, curN);
```

**修复方案**:
```cpp
// 对齐到 float 最小对齐要求
int64_t reduceN = ((curN + 7) / 8) * 8;
if (reduceN < 8) reduceN = 8;

ReduceSum(tmp, fpX, work, reduceN);
WaitVToS();
localLoss_ += tmp.GetValue(0);

WaitSToV();
ReduceSum(tmp, fpW, work, reduceN);
WaitVToS();
localWeight_ += tmp.GetValue(0);
```

**注意**: padding 元素已由 Duplicate(fpX, 0.0f, calcN) 初始化为 0, 不影响求和结果。

### 修复 3 (中优先级): WritePartialToWorkspace 添加同步

**文件**: `op_kernel/nll_loss.h`
**函数**: `WritePartialToWorkspace()` (行 359-375)

**在每次 DataCopyPad 后确保写入完成**:
```cpp
template <typename T, typename TargetT>
__aicore__ inline void NllLoss<T, TargetT>::WritePartialToWorkspace()
{
    LocalTensor<float> tmp = tmpBuf.Get<float>();
    Duplicate(tmp, 0.0f, WS_FLOATS_PER_SLOT);
    WaitVToS();
    tmp.SetValue(0, localLoss_);
    WaitSToMte3();
    DataCopyExtParams params{1, static_cast<uint32_t>(BLOCK_BYTES), 0, 0, 0};
    DataCopyPad(wsGM[blockIdx_ * WS_FLOATS_PER_SLOT], tmp, params);
    WaitVToMte3();  // 新增: 确保 loss 写入完成

    Duplicate(tmp, 0.0f, WS_FLOATS_PER_SLOT);
    WaitVToS();
    tmp.SetValue(0, localWeight_);
    WaitSToMte3();
    int64_t wBase = tiling_->needCoreNum * WS_FLOATS_PER_SLOT;
    DataCopyPad(wsGM[wBase + blockIdx_ * WS_FLOATS_PER_SLOT], tmp, params);
    WaitVToMte3();  // 新增: 确保 weight 写入完成
}
```

### 修复 4 (可选): 简化 Tiling, 限制 perCoreSize 最小值

**文件**: `op_host/nll_loss_tiling.cpp`
**函数**: `NllLossTilingFunc()` (行 111-249)

**思路**: 如果 perCoreSize 过小 (如 1), 增大 tileN 以覆盖更多元素, 减少边界条件触发:

```cpp
// 在行 206 之后添加
// 确保 tileN 不小于最小对齐要求
constexpr int64_t MIN_REDUCE_ALIGN = 8;
if (tileN < MIN_REDUCE_ALIGN) {
    tileN = MIN_REDUCE_ALIGN;
}
```

---

## 7. 推荐排查步骤

```
步骤 1: 应用修复 1 + 修复 2 (buffer 尺寸 + ReduceSum 对齐)
  ↓ 编译: bash build.sh && bash build/custom_opp_ubuntu_aarch64.run
  ↓ 测试: rm -rf examples/build && cd examples && bash run.sh
  ↓ 通过 → 问题解决
  ↓ 失败 → 继续步骤 2

步骤 2: 追加修复 3 (WritePartialToWorkspace 同步)
  ↓ 编译测试同上
  ↓ 通过 → 问题解决
  ↓ 失败 → 继续步骤 3

步骤 3: 添加 Kernel Printf 定位崩溃点
  在 Process() 函数中关键位置添加 printf
  ↓ 编译测试, 观察输出
  ↓ 根据崩溃阶段细化修复

步骤 4: 如仍失败, 使用 msprof 获取硬件级异常
  export ASCEND_GLOBAL_LOG_LEVEL=0
  msprof --application-profiling=on --output=./prof ./examples/build/bin/test_aclnn_nll_loss

步骤 5: 简化测试用例
  修改 examples: xShape={1,2}, targetShape={1}
  排除多 core 协作导致的问题
```

---

## 8. 关键代码位置

### op_kernel/nll_loss.h 函数与行号

```
函数名                     行号        风险等级   说明
─────────────────────────────────────────────────────
AlignElems()              78-85       -         对齐计算工具
WaitMte2ToS()             88-93       -         同步: MTE2→Scalar
WaitSToV()                96-101      -         同步: Scalar→Vector
WaitVToS()                104-109     -         同步: Vector→Scalar
WaitVToMte3()             112-117     -         同步: Vector→MTE3
WaitSToMte3()             120-125     -         同步: Scalar→MTE3
Init()                    128-145     -         初始化, 设置 GM/Tiling
ParseCoreRange()          148-153     -         计算当前 core 范围
InitBuffers()             156-187     ★★★       [假设B/D] buffer 尺寸
CopyInX()                 190-196     ★         拷贝 X 到 UB
CopyInTarget()            199-205     ★         拷贝 target 到 UB
CopyOutY()                208-213     -         拷贝 UB 到 Y
LoadWeight()              216-225     -         加载 weight
GatherNormal()            228-255     ★★        [假设C] Duplicate 越界
GatherLarge()             258-293     ★         大 tensor 模式
ComputeTile()             296-340     ★★★       [假设A] ReduceSum 对齐
ProcessOneTile()          343-356     -         单 tile 处理调度
WritePartialToWorkspace() 359-375     ★★        [假设E] 同步时序
WriteScalarOut()          378-395     -         标量输出
ReduceAndWriteOutputs()   398-427     ★         多 core 归约
Process()                 430-454     -         主流程入口
```

### op_host/nll_loss_tiling.cpp 关键行号

```
函数名                     行号        说明
──────────────────────────────────────────────
NllLossTilingFunc()       111-249     Tiling 主函数
  GetPlatformInfo()       99-109      获取 UB/CoreNum
  ParseReduction()        49-61       解析 reduction 字符串
  GetSchMode()            63-73       根据 dtype 选 schMode
  SplitLoops()            75-96       计算 tile 循环参数
TilingParseForNllLoss()   252-255     Parse 函数 (空)
```

---

## 9. 验证命令速查

```bash
# 编译
bash build.sh

# 安装算子包
bash build/custom_opp_ubuntu_aarch64.run

# 运行 NPU examples
rm -rf examples/build
source /home/developer/Ascend/cann-9.0.0/set_env.sh
export LD_LIBRARY_PATH=/home/developer/Ascend/cann-9.0.0/opp/vendors/nll_loss_custom/op_api/lib/:${LD_LIBRARY_PATH}
cd examples && bash run.sh

# 运行 CPU 仿真 UT
bash build.sh -u

# 查看最新 plog 错误日志
ls -lt ~/ascend/log/debug/plog/ | head -1
cat ~/ascend/log/debug/plog/plog-<latest>.log

# 查看 NPU 状态
npu-smi info
```

---

## 10. 总结

| 假设 | 根因 | 置信度 | 修复难度 | 建议 |
|------|------|:------:|:--------:|------|
| D | validXBuf/validWBuf 尺寸 16B, Duplicate 写 32B → **越界** | 高 | 低 | 优先修复 |
| B | workBuf 32B 不够 ReduceSum 内部使用 | 高 | 低 | 与 D 一起修 |
| A | ReduceSum count=1 不满足对齐要求 | 高 | 低 | 对齐到 8 |
| C | GatherNormal 中 alignN > buffer 容量 | 中 | 低 | 已被 D 覆盖 |
| E | WritePartialToWorkspace 写入未完成就 SyncAll | 低 | 低 | 可选 |

**建议优先级**: 修复 1 (buffer) > 修复 2 (ReduceSum 对齐) > 修复 3 (同步) > 修复 4 (Tiling)

---

## 11. 相较于测试前的代码改动

> 基准: `git` 仓库 HEAD 提交 (测试前状态)

### 11.1 已修改的跟踪文件

#### 改动 1: `build.sh` (内容修改)

**文件**: `build.sh` 第 50 行
**性质**: 功能性修改, 修复 SoC 类型不匹配问题

```diff
-COMPUTE_UNIT="ascend910b"
+COMPUTE_UNIT="ascend910_93"
```

**原因**: `build.sh` 默认编译目标为 `ascend910b`, 但当前环境 NPU 实际 SoC 为 `ascend910_93`。不修改会导致算子包安装后 NPU 无法识别 (报 `binary_info_config.json of socVersion [ascend910_93] does not support opType`)。

**回退方法**: `git checkout -- build.sh`

#### 改动 2: `examples/run.sh` (权限修改)

**文件**: `examples/run.sh`
**性质**: 权限变更, 无内容修改

```diff
-old mode 100644 (rw-r--r--)
+new mode 100755 (rwxr-xr-x)
```

**原因**: 原始文件缺少执行权限, `build.sh -e` 内部通过 `./run.sh` 调用时报 `Permission denied`。通过 `chmod +x` 添加执行权限。

**回退方法**: `git checkout -- examples/run.sh` 或 `chmod 644 examples/run.sh`

#### 改动 3: `tests/ut/run.sh` (权限修改)

**文件**: `tests/ut/run.sh`
**性质**: 权限变更, 无内容修改

```diff
-old mode 100644 (rw-r--r--)
+new mode 100755 (rwxr-xr-x)
```

**原因**: 与 examples/run.sh 同理, `build.sh -u` 内部通过 `./run.sh` 调用时需要执行权限。

**回退方法**: `git checkout -- tests/ut/run.sh` 或 `chmod 644 tests/ut/run.sh`

### 11.2 已删除的跟踪文件

#### 删除: 无关 .doc 文件

```
删除: 实验四数字电路实验报告王宇松2025681093.doc (4.3MB, 二进制)
```

**说明**: 该文件为与项目无关的 .doc 文档, 在测试过程中被删除。与算子功能无关。

**恢复方法**: `git checkout -- "实验四数字电路实验报告王宇松2025681093.doc"`

### 11.3 未跟踪的新增文件 (测试产物)

以下文件为测试过程中生成, 不在 git 跟踪范围内:

| 文件/目录 | 来源 | 说明 |
|-----------|------|------|
| `docs/NPU_DEBUG_REPORT.md` | 本次新建 | 本调试报告文档 |
| `docs/AI_FIX_GUIDE.md` | 如存在 | AI 修复指南文档 |
| `build/` | 编译产物 | build.sh 生成的编译中间文件和输出包 |
| `build_out/` | 编译产物 | build.sh 的额外输出目录 |
| `examples/build/` | 编译产物 | examples 的 CMake 构建目录 |
| `tests/ut/build/` | 编译产物 | UT 测试的 CMake 构建目录 |
| `tests/ut/op_kernel/nll_loss_data/*.bin` | UT 生成 | gen_data.py 生成的测试数据和 nll_loss_op_kernel_ut 输出的结果 bin |
| `.opencode/` | IDE 配置 | opencode 工具配置目录 |
| `.vscode/` | IDE 配置 | VS Code 配置目录 |

**清理方法**:
```bash
# 清理编译产物
bash build.sh --make_clean
rm -rf examples/build tests/ut/build

# 清理 UT 测试数据
rm -f tests/ut/op_kernel/nll_loss_data/*.bin
```

### 11.4 算子核心源码改动情况

| 文件 | 是否修改 |
|------|:--------:|
| `op_kernel/nll_loss.cpp` | ❌ 未改动 |
| `op_kernel/nll_loss.h` | ❌ 未改动 |
| `op_kernel/nll_loss_tiling_data.h` | ❌ 未改动 |
| `op_kernel/nll_loss_tiling_key.h` | ❌ 未改动 |
| `op_host/nll_loss_def.cpp` | ❌ 未改动 |
| `op_host/nll_loss_infershape.cpp` | ❌ 未改动 |
| `op_host/nll_loss_tiling.cpp` | ❌ 未改动 |
| `op_host/CMakeLists.txt` | ❌ 未改动 |
| `op_kernel/CMakeLists.txt` | ❌ 未改动 |
| `CMakeLists.txt` | ❌ 未改动 |
| `examples/test_aclnn_nll_loss.cpp` | ❌ 未改动 |
| `examples/CMakeLists.txt` | ❌ 未改动 |
| `tests/ut/op_kernel/test_nll_loss.cpp` | ❌ 未改动 |
| `tests/ut/op_kernel/nll_loss_tiling.h` | ❌ 未改动 |
| `tests/ut/op_kernel/nll_loss_data/gen_data.py` | ❌ 未改动 |
| `tests/ut/op_kernel/nll_loss_data/compare_data.py` | ❌ 未改动 |

**结论**: 算子核心源码 (op_kernel/, op_host/, examples/test_aclnn_nll_loss.cpp, tests/) 均未做任何修改。所有改动仅限于构建脚本 (`build.sh`) 和脚本执行权限 (`run.sh`)。

### 11.5 一键回退所有改动

```bash
# 回退跟踪文件的内容和权限改动
git checkout -- build.sh examples/run.sh tests/ut/run.sh

# 恢复已删除的 .doc 文件 (可选)
git checkout -- "实验四数字电路实验报告王宇松2025681093.doc"

# 清理所有测试产物
bash build.sh --make_clean
rm -rf examples/build tests/ut/build docs/
rm -f tests/ut/op_kernel/nll_loss_data/*.bin
```
