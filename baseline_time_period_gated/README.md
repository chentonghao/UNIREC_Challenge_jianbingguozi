# baseline_time_period_gated

这是在 `baseline_time_period/` 基础上复制出来的一版实验目录，用来验证“时间门控融合替代直接相加”是否能改善周期时间特征的注入方式。

## 实验目标

上一版 `baseline_time_period/` 已经加入了两类周期时间信息：

- request-level 周期 dense 特征
  - `hour_sin`
  - `hour_cos`
  - `dow_sin`
  - `dow_cos`
- sequence-level 周期 bucket 特征
  - `hour bucket`
  - `day-of-week bucket`

但上一版在 sequence 侧采用的是最直接的融合方式：

- `token_emb + time_emb`
- `token_emb + hour_emb`
- `token_emb + dow_emb`

这一版不改数据定义，只改“怎么把时间信息融进 sequence token embedding”：

- 默认仍然是 `add`
- 新增可选 `gated_add`

核心思路是：

- 保持默认行为和上一版完全一致
- 当打开 `gated_add` 时，不再把时间 embedding 硬加到 token 上
- 改成先通过 gate 控制每个 token 接收多少时间信号，再把时间 embedding 加回去

## 主要改动

### 1. `model.py`

这一版只在 `PCVRHyFormer` 中增加时间融合控制，不改数据 shape，不改其他路径。

#### 新增模型参数

- `time_fusion_mode: str = "add"`
- `time_gate_init_bias: float = -2.0`

含义：

- `add`
  - 保持上一版的直接相加逻辑
- `gated_add`
  - 使用 gate 控制时间 embedding 的注入强度

#### 新增 gate

当 `num_time_buckets > 0` 时新增：

- `self.time_gate = nn.Linear(d_model, d_model)`

当 `use_time_period_features=True` 时新增：

- `self.hour_gate = nn.Linear(d_model, d_model)`
- `self.dow_gate = nn.Linear(d_model, d_model)`

#### 初始化策略

为了避免一开始又退化成“硬加”，这三个 gate 采用：

- `weight = 0`
- `bias = time_gate_init_bias`

默认 `time_gate_init_bias = -2.0`，因此初始时：

- `sigmoid(-2.0) ≈ 0.119`

也就是说，训练一开始时间信号是“小开口”注入，而不是满量注入。

#### 融合逻辑

在 `_embed_seq_domain()` 中：

- 若 `time_fusion_mode == "add"`
  - 保持原来的直接相加逻辑
- 若 `time_fusion_mode == "gated_add"`
  - 先记录 `base_token = token_emb`
  - 再按下面的方式融合：

```python
token_emb = token_emb + sigmoid(self.time_gate(base_token)) * time_emb
token_emb = token_emb + sigmoid(self.hour_gate(base_token)) * hour_emb
token_emb = token_emb + sigmoid(self.dow_gate(base_token)) * dow_emb
```

这里的 gate 是“每个 token 一个门”，shape 不变，不影响后续 backbone 接口。

### 2. `train.py`

新增两个训练参数：

- `--time_fusion_mode`
  - 可选值：`add` / `gated_add`
  - 默认：`add`
- `--time_gate_init_bias`
  - 默认：`-2.0`

这两个参数会写入 `train_config.json`，用于推理阶段重建同样的模型配置。

此外，这一版继续保留平台平铺目录运行所需的 fallback import。

### 3. `infer.py`

推理脚本同步支持这两个新配置：

- `time_fusion_mode`
- `time_gate_init_bias`

如果 checkpoint 里的 `train_config.json` 存在这两个字段，就按训练配置重建；如果不存在，则回退到：

- `time_fusion_mode = "add"`
- `time_gate_init_bias = -2.0`

### 4. `trainer.py`

训练逻辑本身没有额外改动，只修正为导入当前 `baseline_time_period_gated` 目录下的模块，避免误用上一版模型代码。

### 5. `run.sh`

默认训练命令不变，仍然保持上一版时间特征实验的默认入口。

只增加了一个门控实验示例注释：

```bash
# bash run.sh --use_time_period_features --use_time_buckets \
#   --time_fusion_mode gated_add --time_gate_init_bias -2.0
```

## 设计动机

这版实验想解决的问题不是“要不要时间特征”，而是“时间特征怎么融进去”。

上一版最激进的地方在于：

- request-level 周期特征直接拼到 `user_dense_feats`
- seq 的 `time/hour/dow` embedding 直接加到 `token_emb`

其中 sequence 侧的“三路直接相加”更容易把噪声时间信号硬灌进 backbone。

因此这一版优先验证：

- 时间特征本身也许不是问题
- 更可能的问题是“融合方式太硬”

## 训练方式

默认运行：

```bash
bash run.sh
```

显式运行门控实验：

```bash
bash run.sh \
  --use_time_period_features \
  --use_time_buckets \
  --time_fusion_mode gated_add \
  --time_gate_init_bias -2.0
```

如果想回退到上一版“直接相加”行为：

```bash
bash run.sh \
  --use_time_period_features \
  --use_time_buckets \
  --time_fusion_mode add
```

## 推理方式

使用当前目录下的推理脚本：

```bash
python infer.py
```

它会从 checkpoint 目录读取：

- `model.pt`
- `schema.json`
- `train_config.json`
- 可选 `ns_groups.json`

并按训练时的 `time_fusion_mode` / `time_gate_init_bias` 重建模型。

## 当前状态

这版代码已经做过本地烟测：

- `gated_add` 训练可跑通
- checkpoint 可正常保存
- `infer.py` 可正常恢复 checkpoint 并输出预测

因此这版现在已经具备继续提交平台做正式实验的条件。
