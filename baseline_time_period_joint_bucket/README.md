# baseline_time_period

这是在 `baseline/` 基础上复制出来的一版实验目录，用来验证“周期时间特征”是否能带来收益。

## 实验目标

在原有 `time bucket` 特征之外，额外引入两类周期时间信息：

- request-level 周期 dense 特征
  - `hour_sin`
  - `hour_cos`
  - `dow_sin`
  - `dow_cos`
- sequence-level 周期 bucket 特征
  - `hour bucket`
  - `day-of-week bucket`

设计原则是：

- 默认关闭时，行为尽量与原 `baseline` 保持一致
- 只有打开 `--use_time_period_features` 时，才启用新增周期时间特征

## 主要改动

### 1. `dataset.py`

这一版做了两类新增特征。

#### request-level 周期特征

基于样本当前请求时间 `timestamp` 生成：

- `hour_sin`
- `hour_cos`
- `dow_sin`
- `dow_cos`

并追加到 `user_dense_feats` 末尾。

#### sequence-level 周期 bucket

为每个序列域额外生成：

- `*_hour_bucket`
- `*_dow_bucket`

映射规则：

- `hour: 0..23 -> 1..24`
- `dow: 0..6 -> 1..7`
- padding 保持 `0`

同时保留原来的：

- `*_time_bucket`

#### 开关控制

新增参数：

- `use_time_period_features`

只有该开关为 `True` 时，才会：

- 扩展 `user_dense_feats`
- 生成 `hour/dow bucket`

### 2. `model.py`

#### 扩展 `ModelInput`

新增两个输入字段：

- `seq_hour_buckets`
- `seq_dow_buckets`

#### 新增 embedding

在 sequence token 上增加两套 embedding：

- `nn.Embedding(25, d_model, padding_idx=0)` 对应 hour
- `nn.Embedding(8, d_model, padding_idx=0)` 对应 day-of-week

它们会在 `use_time_period_features=True` 时叠加到 sequence token embedding 上。

### 3. `train.py`

新增训练参数：

- `--use_time_period_features`

同时做了两类兼容处理：

#### 平台平铺目录导入兼容

比赛平台不会保留原来的包目录，而是把文件平铺复制到运行目录。  
所以这里增加了 import fallback：

- 本地开发时优先用包路径导入
- 平台上退回到同目录导入

避免出现：

```text
ModuleNotFoundError: No module named 'UNIREC_Challenge_jianbingguozi'
```

#### `argparse --help` 修复

原 help 文本里有未转义的 `%`，会导致 `--help` 直接报错。  
这里已经修复。

### 4. `trainer.py`

主要做了两件事：

- 把 `seq_hour_buckets` / `seq_dow_buckets` 正确传给 `ModelInput`
- 增加平台平铺目录下的 fallback import

此外，这一版训练仍然会在 checkpoint 目录写出：

- `model.pt`
- `schema.json`
- `train_config.json`
- 可选 `ns_groups.json`

### 5. `infer.py`

这一版单独补了一份 `baseline_time_period/infer.py`。

原因是原来的 `baseline/infer.py` 只会处理：

- `seq_time_buckets`

但不会处理新增的：

- `seq_hour_buckets`
- `seq_dow_buckets`

所以如果直接复用原推理脚本，会导致训练和推理输入不一致。

新的 `infer.py` 具备这些能力：

- 根据 checkpoint 中的
  - `model.pt`
  - `schema.json`
  - `train_config.json`
  - 可选 `ns_groups.json`
  重建模型
- 根据 `train_config.json` 判断是否启用 `use_time_period_features`
- 推理时把 `seq_hour_buckets` / `seq_dow_buckets` 一并送进模型
- 兼容平台平铺目录运行

### 6. `run.sh`

当前默认配置与原 `baseline/run.sh` 尽量保持一致，只额外默认打开：

- `--use_time_period_features`
- `--use_time_buckets`

也就是说，这一版默认就是“时间周期特征实验版”。

## 平台运行注意事项

### 1. 平台是平铺目录，不是原包结构

平台会把这些文件直接复制到一个目录下运行，例如：

- `train.py`
- `trainer.py`
- `dataset.py`
- `model.py`
- `utils.py`
- `infer.py`

不会保留：

```text
UNIREC_Challenge_jianbingguozi/baseline_time_period/
```

所以这版代码已经专门加了 fallback import 兼容这种运行方式。

### 2. `ns_groups.json` 在当前默认 RankMixer 路径下不是必需

当前 `run.sh` 里默认使用：

```bash
--ns_groups_json ""
```

因此在默认 `rankmixer` 配置下：

- 就算没上传 `ns_groups.json`
- 代码也会退回到“每个特征单独一组”的默认逻辑

所以缺少 `ns_groups.json` 不会导致当前默认训练路径直接报错。

### 3. OOM 风险与这版实验无关的部分

平台上如果出现 OOM，通常首先看这些参数：

- `batch_size`
- `seq_max_lens`
- `emb_skip_threshold`
- `seq_encoder_type`

这版额外时间周期特征会增加少量显存压力，但平台上真正导致 OOM 的通常还是原始模型规模和 batch size。

## 训练方式

直接运行：

```bash
bash run.sh
```

推荐显式运行方式：

```bash
bash run.sh \
  --use_time_period_features \
  --use_time_buckets \
  --dropout_rate 0.01 \
  --num_epochs 3
```

## 推理方式

请使用：

- `baseline_time_period/infer.py`

不要直接用：

- `baseline/infer.py`

原因是这版推理必须和训练期一样，把新增的 sequence hour/dow bucket 一起送进模型。

推理脚本依赖的环境变量：

- `MODEL_OUTPUT_PATH`
- `EVAL_DATA_PATH`
- `EVAL_RESULT_PATH`

## 目录内文件

这一版当前包含：

- `dataset.py`
- `model.py`
- `train.py`
- `trainer.py`
- `utils.py`
- `run.sh`
- `infer.py`
- `ns_groups.json`

