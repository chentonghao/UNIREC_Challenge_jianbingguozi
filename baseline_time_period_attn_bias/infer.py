"""PCVRHyFormer inference script for the baseline_time_period_attn_bias variant.

Rebuilds the model from ``schema.json`` + ``ns_groups.json`` +
``train_config.json`` saved next to ``model.pt`` during training.

Environment variables:
    MODEL_OUTPUT_PATH  Checkpoint directory containing ``model.pt`` and sidecars.
    EVAL_DATA_PATH     Test data directory (*.parquet + schema.json).
    EVAL_RESULT_PATH   Output directory for ``predictions.json``.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from UNIREC_Challenge_jianbingguozi.baseline_time_period_attn_bias.dataset import (
        FeatureSchema,
        NUM_TIME_BUCKETS,
        PCVRParquetDataset,
    )
    from UNIREC_Challenge_jianbingguozi.baseline_time_period_attn_bias.model import (
        ModelInput,
        PCVRHyFormer,
    )
except ModuleNotFoundError:
    from dataset import FeatureSchema, NUM_TIME_BUCKETS, PCVRParquetDataset
    from model import ModelInput, PCVRHyFormer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


_FALLBACK_MODEL_CFG = {
    'd_model': 64,
    'emb_dim': 64,
    'num_queries': 1,
    'num_hyformer_blocks': 2,
    'num_heads': 4,
    'seq_encoder_type': 'transformer',
    'hidden_mult': 4,
    'dropout_rate': 0.01,
    'seq_top_k': 50,
    'seq_causal': False,
    'action_num': 1,
    'num_time_buckets': NUM_TIME_BUCKETS,
    'use_time_attn_bias': False,
    'rank_mixer_mode': 'full',
    'use_rope': False,
    'rope_base': 10000.0,
    'emb_skip_threshold': 0,
    'seq_id_threshold': 10000,
    'use_time_period_features': False,
    'ns_tokenizer_type': 'rankmixer',
    'user_ns_tokens': 0,
    'item_ns_tokens': 0,
}

_FALLBACK_SEQ_MAX_LENS = 'seq_a:256,seq_b:256,seq_c:512,seq_d:512'
_FALLBACK_BATCH_SIZE = 256
_FALLBACK_NUM_WORKERS = 16
_MODEL_CFG_KEYS = list(_FALLBACK_MODEL_CFG.keys())


def build_feature_specs(
    schema: FeatureSchema,
    per_position_vocab_sizes: List[int],
) -> List[Tuple[int, int, int]]:
    specs: List[Tuple[int, int, int]] = []
    for fid, offset, length in schema.entries:
        vs = max(per_position_vocab_sizes[offset:offset + length])
        specs.append((vs, offset, length))
    return specs


def _parse_seq_max_lens(sml_str: str) -> Dict[str, int]:
    seq_max_lens: Dict[str, int] = {}
    for pair in sml_str.split(','):
        k, v = pair.split(':')
        seq_max_lens[k.strip()] = int(v.strip())
    return seq_max_lens


def load_train_config(model_dir: str) -> Dict[str, Any]:
    train_config_path = os.path.join(model_dir, 'train_config.json')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            cfg = json.load(f)
        logging.info(f"Loaded train_config from {train_config_path}")
        return cfg
    logging.warning(
        f"train_config.json not found in {model_dir}, falling back to hardcoded defaults."
    )
    return {}


def resolve_model_cfg(train_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    for key in _MODEL_CFG_KEYS:
        if key == 'num_time_buckets':
            if 'num_time_buckets' in train_config:
                cfg[key] = train_config['num_time_buckets']
            elif 'use_time_buckets' in train_config:
                cfg[key] = NUM_TIME_BUCKETS if train_config['use_time_buckets'] else 0
            else:
                cfg[key] = _FALLBACK_MODEL_CFG[key]
            continue

        if key in train_config:
            cfg[key] = train_config[key]
        else:
            cfg[key] = _FALLBACK_MODEL_CFG[key]
            logging.warning(f"train_config missing '{key}', using fallback = {cfg[key]}")
    return cfg


def build_model(
    dataset: PCVRParquetDataset,
    model_cfg: Dict[str, Any],
    ns_groups_json: Optional[str] = None,
    device: str = 'cpu',
) -> PCVRHyFormer:
    if ns_groups_json and os.path.exists(ns_groups_json):
        logging.info(f"Loading NS groups from {ns_groups_json}")
        with open(ns_groups_json, 'r') as f:
            ns_groups_cfg = json.load(f)
        user_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.user_int_schema.entries)
        }
        item_fid_to_idx = {
            fid: i for i, (fid, _, _) in enumerate(dataset.item_int_schema.entries)
        }
        user_ns_groups = [
            [user_fid_to_idx[f] for f in fids]
            for fids in ns_groups_cfg['user_ns_groups'].values()
        ]
        item_ns_groups = [
            [item_fid_to_idx[f] for f in fids]
            for fids in ns_groups_cfg['item_ns_groups'].values()
        ]
    else:
        logging.info("No NS groups JSON found, using default: each feature as one group")
        user_ns_groups = [[i] for i in range(len(dataset.user_int_schema.entries))]
        item_ns_groups = [[i] for i in range(len(dataset.item_int_schema.entries))]

    user_int_feature_specs = build_feature_specs(
        dataset.user_int_schema, dataset.user_int_vocab_sizes)
    item_int_feature_specs = build_feature_specs(
        dataset.item_int_schema, dataset.item_int_vocab_sizes)

    logging.info(f"Building PCVRHyFormer with cfg: {model_cfg}")
    model = PCVRHyFormer(
        user_int_feature_specs=user_int_feature_specs,
        item_int_feature_specs=item_int_feature_specs,
        user_dense_dim=dataset.user_dense_schema.total_dim,
        item_dense_dim=dataset.item_dense_schema.total_dim,
        seq_vocab_sizes=dataset.seq_domain_vocab_sizes,
        user_ns_groups=user_ns_groups,
        item_ns_groups=item_ns_groups,
        **model_cfg,
    ).to(device)
    return model


def load_model_state_strict(model: nn.Module, ckpt_path: str, device: str) -> None:
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)


def get_ckpt_path(model_dir: str) -> Optional[str]:
    if not model_dir or not os.path.isdir(model_dir):
        return None
    for item in os.listdir(model_dir):
        if item.endswith('.pt'):
            return os.path.join(model_dir, item)
    return None


def _batch_to_model_input(batch: Dict[str, Any], device: str) -> ModelInput:
    device_batch: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            device_batch[k] = v.to(device, non_blocking=True)
        else:
            device_batch[k] = v

    seq_domains = device_batch['_seq_domains']
    seq_data: Dict[str, torch.Tensor] = {}
    seq_lens: Dict[str, torch.Tensor] = {}
    seq_time_buckets: Dict[str, torch.Tensor] = {}
    seq_hour_buckets: Optional[Dict[str, torch.Tensor]] = None
    seq_dow_buckets: Optional[Dict[str, torch.Tensor]] = None
    has_hour_buckets = any(f'{domain}_hour_bucket' in device_batch for domain in seq_domains)
    has_dow_buckets = any(f'{domain}_dow_bucket' in device_batch for domain in seq_domains)
    if has_hour_buckets:
        seq_hour_buckets = {}
    if has_dow_buckets:
        seq_dow_buckets = {}
    for domain in seq_domains:
        seq_data[domain] = device_batch[domain]
        seq_lens[domain] = device_batch[f'{domain}_len']
        B, _, L = device_batch[domain].shape
        seq_time_buckets[domain] = device_batch.get(
            f'{domain}_time_bucket',
            torch.zeros(B, L, dtype=torch.long, device=device),
        )
        if seq_hour_buckets is not None and f'{domain}_hour_bucket' in device_batch:
            seq_hour_buckets[domain] = device_batch[f'{domain}_hour_bucket']
        if seq_dow_buckets is not None and f'{domain}_dow_bucket' in device_batch:
            seq_dow_buckets[domain] = device_batch[f'{domain}_dow_bucket']

    return ModelInput(
        user_int_feats=device_batch['user_int_feats'],
        item_int_feats=device_batch['item_int_feats'],
        user_dense_feats=device_batch['user_dense_feats'],
        item_dense_feats=device_batch['item_dense_feats'],
        seq_data=seq_data,
        seq_lens=seq_lens,
        seq_time_buckets=seq_time_buckets,
        seq_hour_buckets=seq_hour_buckets,
        seq_dow_buckets=seq_dow_buckets,
    )


def main() -> None:
    model_dir = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')

    if not model_dir or not data_dir or not result_dir:
        raise ValueError(
            "MODEL_OUTPUT_PATH, EVAL_DATA_PATH, and EVAL_RESULT_PATH must all be set."
        )
    os.makedirs(result_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    schema_path = os.path.join(model_dir, 'schema.json')
    if not os.path.exists(schema_path):
        schema_path = os.path.join(data_dir, 'schema.json')
    logging.info(f"Using schema: {schema_path}")

    train_config = load_train_config(model_dir)
    seq_max_lens = _parse_seq_max_lens(
        train_config.get('seq_max_lens', _FALLBACK_SEQ_MAX_LENS)
    )
    logging.info(f"seq_max_lens: {seq_max_lens}")

    batch_size = int(train_config.get('batch_size', _FALLBACK_BATCH_SIZE))
    num_workers = int(train_config.get('num_workers', _FALLBACK_NUM_WORKERS))
    use_time_period_features = bool(train_config.get('use_time_period_features', False))
    if 'time_period_mode' in train_config:
        time_period_mode = str(train_config['time_period_mode'])
    else:
        time_period_mode = 'full' if use_time_period_features else 'off'

    test_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
        use_time_period_features=use_time_period_features,
        time_period_mode=time_period_mode,
    )
    logging.info(f"Total test samples: {test_dataset.num_rows}")

    model_cfg = resolve_model_cfg(train_config)
    ns_groups_json = train_config.get('ns_groups_json', None)
    if ns_groups_json:
        local_candidate = os.path.join(model_dir, os.path.basename(ns_groups_json))
        if os.path.exists(local_candidate):
            ns_groups_json = local_candidate

    model = build_model(
        test_dataset,
        model_cfg=model_cfg,
        ns_groups_json=ns_groups_json,
        device=device,
    )

    ckpt_path = get_ckpt_path(model_dir)
    if ckpt_path is None:
        raise FileNotFoundError(f"No *.pt file found under MODEL_OUTPUT_PATH={model_dir!r}")
    logging.info(f"Loading checkpoint from {ckpt_path}")
    load_model_state_strict(model, ckpt_path, device)
    model.eval()
    logging.info("Model loaded successfully")

    loader_kw: Dict[str, Any] = {
        'batch_size': None,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kw['prefetch_factor'] = 2
    test_loader = DataLoader(test_dataset, **loader_kw)

    all_probs = []
    all_user_ids = []
    logging.info("Starting inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            model_input = _batch_to_model_input(batch, device)
            user_ids = batch.get('user_id', [])
            logits, _ = model.predict(model_input)
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_user_ids.extend(user_ids)

            if (batch_idx + 1) % 100 == 0:
                logging.info(f"Processed {(batch_idx + 1) * batch_size} samples")

    predictions = {
        'predictions': dict(zip(all_user_ids, all_probs)),
    }
    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    logging.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == "__main__":
    main()
