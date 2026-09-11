# NllLoss

Ascend C 实现的负对数似然损失。目标芯片 Ascend910B。规格与实现说明以 [docs/AI_PROJECT.md](../docs/AI_PROJECT.md) 为准。

## 功能说明

$$
l_n = -w_{y_n}\, x_{n,y_n},\quad
w_c=\mathrm{weight}[c]\cdot 1\{y_n \neq \mathrm{ignore\_index}\}
$$

`reduction` 为 `none` 时输出逐样本 loss；`sum` 为求和；`mean` 为按有效权重归一化。同时输出 `total_weight`。

## 产品支持

| 产品 | 是否支持 |
|------|:--------:|
| Atlas A2（Ascend910B） | √ |
| 其他 | 本模板未注册 |

## 参数说明

见 [docs/aclnnNllLoss.md](../docs/aclnnNllLoss.md)。

## 编译与示例

```bash
cd code
bash build.sh          # 编译 custom opp
bash build.sh -e       # 编译并跑 examples（需要 NPU + 已安装算子包）
```

安装生成的 `custom_opp_*.run` 后，示例通过 `aclnnNllLossGetWorkspaceSize` / `aclnnNllLoss` 调用。

## 目录

| 路径 | 说明 |
|------|------|
| `op_host/` | 定义、InferShape、Tiling |
| `op_kernel/` | Ascend C kernel |
| `examples/` | ACLNN 调用示例 |
| `tests/ut/` | 模板 UT 骨架 |
