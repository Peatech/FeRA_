# Artifact Traceability — FeRA

Maps every figure and table in the paper to its reproduction command.
All runs use `seed=42` and `training_mode=sequential` unless noted.
Attack runs load a pretrained checkpoint and run for 100 poisoning rounds.

**Checkpoint (CIFAR-10, ResNet-18, round 2000):**
```
checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

---

## Motivation Figures

### `fig:variance_suppression`, `fig:collusion_patterns`, `fig:topk_eigenvalues`, `fig:norm_inflation`, `fig:cka_attack`, `fig:pca_scatter`

_Observation figures: representation variance, inter-client similarity, top-k eigenvalues, spectral-ratio amplification, CKA matrix, 2D PCA of deltas._

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  aggregator_config.fera_visualize.log_ranked_metric_tables=true \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

Metrics written to `outputs/.../fera_visualize/all_rounds_metrics.csv`.

---

## Main Tables

### `tab:iid_main` — IID CIFAR-10: 9 attacks × defences

```bash
python main.py --config-name cifar10 \
  aggregator=<DEFENCE> \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=<TRIGGER> \
  atk_config.model_poison_method=<ATTACK> \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=0.9 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

`<DEFENCE>`: `fera_visualize`, `unweighted_fedavg`, `multi_krum`, `foolsgold`, `flame`, `fltrust`, `robustlr`, `deepsight`, `fldetector`  
`<ATTACK>` / `<TRIGGER>`: `base/badnets`, `neurotoxin/pattern`, `a3fl/pattern`, `anticipate/pattern`, `chameleon/chameleon`, `dba/distributed`, `iba/iba`

---

### `tab:other_datasets` — CIFAR-100 and Tiny-ImageNet

```bash
# CIFAR-100
python main.py --config-name cifar100 \
  aggregator=<DEFENCE> \
  atk_config=cifar100_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=1001 atk_config.poison_end_round=1101 \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR100_unweighted_fedavg/resnet18_round_1000_dir_0.4.pth

# Tiny-ImageNet
python main.py --config-name tinyimagenet \
  aggregator=<DEFENCE> \
  atk_config=tinyimagenet_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=501 atk_config.poison_end_round=601 \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/TINYIMAGENET_unweighted_fedavg/resnet18_round_500_dir_0.4.pth
```

---

### `tab:non_iid` — Non-IID severity (Neurotoxin, CIFAR-10)

```bash
python main.py --config-name cifar10 \
  aggregator=<DEFENCE> \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=<0.5|0.3|0.1> num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

---

### `tab:iid_sensitivity` — FeRA across α values and 7 attacks

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=<TRIGGER> \
  atk_config.model_poison_method=<ATTACK> \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=<0.1|0.2|0.5|0.7|0.9> num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

---

### `tab:cross_device` — Cross-silo FL (10 clients, all participate)

```bash
python main.py --config-name cifar10 \
  aggregator=<DEFENCE> \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  num_clients=10 num_clients_per_round=10 \
  alpha=<0.9|0.5> num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

---

### `tab:a3fl_cifar10` — A3FL on CIFAR-10

```bash
python main.py --config-name cifar10 \
  aggregator=<DEFENCE> \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=a3fl \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

### `fig:a3fl_iid_performance` — A3FL across datasets and α values

Same command with `alpha=<0.1|0.2|0.5|0.7|0.9>`, repeated for `--config-name cifar100` and `tinyimagenet`.

---

### `tab:adaptive_attacks` — Adaptive BadNet

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=badnets \
  atk_config.model_poison_method=adaptive_badnet \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

---

### `tab:decorr_attack` — Decorrelated backdoor

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=badnets \
  atk_config.model_poison_method=decorrelated \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

---

### `tab:no_attack_convergence` — No-attack FPR and convergence gap

```bash
# FeRA — no attack
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  no_attack=True \
  alpha=<1.0|0.5|0.2> num_rounds=100 seed=42

# FedAvg baseline
python main.py --config-name cifar10 \
  aggregator=unweighted_fedavg \
  no_attack=True \
  alpha=<1.0|0.5|0.2> num_rounds=100 seed=42
```

---

### `tab:poison_percentage` — Varying malicious client fraction

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=badnets \
  atk_config.model_poison_method=base \
  atk_config.fraction_adversaries=<0.1|0.2|0.3> \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

---

### `tab:more_datasets` / `fig:gtsrb` — GTSRB

```bash
python main.py --config-name gtsrb \
  aggregator=<DEFENCE> \
  atk_config=gtsrb_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  alpha=0.5 num_rounds=100 seed=42
```

### `tab:mnist_family_shallow` / `tab:mnist_defended` — EMNIST / FEMNIST / MNIST

```bash
python main.py --config-name <mnist|emnist|femnist> \
  aggregator=<DEFENCE> \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=<ATTACK> \
  num_rounds=100 seed=42
```

---

## Ablation Tables

### `tab:ablation` — Filter component contributions

```bash
# All filters on (default)
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=<badnets|pattern> \
  atk_config.model_poison_method=<base|neurotoxin> \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth

# Consistency filter only
  aggregator_config.fera_visualize.scaled_norm_filter.enabled=false

# Norm-Inflation filter only
  aggregator_config.fera_visualize.default_filter.enabled=false
```

---

### `tab:root_size` — Root dataset size (in-distribution)

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  aggregator_config.fera_visualize.root_size=<16|32|64|128|256> \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

---

### `tab:root_size_ood` — Root dataset size (OOD: CIFAR-100 as root)

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  aggregator_config.fera_visualize.use_ood_root_dataset=true \
  aggregator_config.fera_visualize.ood_root_dataset_name=CIFAR100 \
  aggregator_config.fera_visualize.root_size=<16|32|64|128|256> \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

---

### `tab:architectures` — Model architecture comparison

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=1200 atk_config.poison_end_round=1450 \
  model=<resnet18|resnet34|vgg16> \
  alpha=0.5 num_rounds=250 seed=42
```

---

### `tab:layer_selection` — Representation extraction depth

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  aggregator_config.fera_visualize.extraction_layer=<penultimate|layer4|layer3|layer2> \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

---

### `tab:feature_dimension` — Feature dimension sensitivity

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  aggregator_config.fera_visualize.feature_dim=<64|128|256|512> \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

---

### `tab:weight_ablation` — DAS spectral vs. delta weight

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  aggregator_config.fera_visualize.spectral_weight=<0.3|0.5|0.7|1.0> \
  aggregator_config.fera_visualize.delta_weight=<0.7|0.5|0.3|0.0> \
  aggregator_config.fera_visualize.default_filter.combined_threshold=0.60 \
  aggregator_config.fera_visualize.default_filter.tda_threshold=0.60 \
  aggregator_config.fera_visualize.default_filter.mutual_sim_threshold=0.60 \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

---

### `tab:consistency_ablation` — Consistency filter threshold sensitivity

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  aggregator_config.fera_visualize.default_filter.combined_threshold=<0.40|0.50|0.60> \
  aggregator_config.fera_visualize.default_filter.tda_threshold=<0.40|0.50|0.60> \
  aggregator_config.fera_visualize.default_filter.mutual_sim_threshold=<0.50|0.60|0.70> \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

---

### `tab:fera-norm-ablation` — Norm-Inflation filter MAD threshold

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=badnets \
  atk_config.model_poison_method=anticipate \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  aggregator_config.fera_visualize.default_filter.enabled=false \
  aggregator_config.fera_visualize.collusion_filter.enabled=false \
  aggregator_config.fera_visualize.outlier_filter.enabled=false \
  aggregator_config.fera_visualize.scaled_norm_filter.enabled=true \
  aggregator_config.fera_visualize.scaled_norm_filter.spectral_ratio_threshold=<50|100|200|500> \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

---

### `tab:bsp_vs_discard` — BSP vs. discard aggregation

```bash
# BSP (Benign Subspace Projection) — paper default
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  aggregator_config.fera_visualize.flagged_client_treatment=project \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth

# Discard
  aggregator_config.fera_visualize.flagged_client_treatment=discard
```

---

## Key Source Files

| File | Role |
|------|------|
| `main.py` | Experiment entry point (Hydra) |
| `config/base.yaml` | FeRA hyperparameter defaults |
| `backfed/servers/fera_visualize_server.py` | FeRA detection and aggregation |
| `backfed/servers/defense_categories.py` | FPR / detection metrics |
| `backfed/clients/adaptive_badnet_client.py` | Adaptive BadNet (`tab:adaptive_attacks`) |
| `backfed/clients/decorrelated_malicious_client.py` | Decorrelated attack (`tab:decorr_attack`) |
