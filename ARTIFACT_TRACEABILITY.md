# Artifact Traceability — FeRA

Maps every figure and table in the paper to its reproduction command.
All runs use `seed=42` and `training_mode=sequential` unless noted.

**Checkpoint availability.** Full pretraining checkpoints (CIFAR-100, Tiny-ImageNet,
non-IID CIFAR-10) are held on institutional storage and are omitted here to
preserve anonymity. Due to repository size constraints, only the IID CIFAR-10
checkpoint (α = 0.9, round 2000) is included:

```
checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

Tables that require a different checkpoint can be reproduced by first running
the pretraining command (no attack, `no_attack=True`, appropriate `num_rounds`)
and then resuming with the generated `.pth` file.

**Common FL configuration (unless stated otherwise).** N=100 clients, n=10
sampled per round, 10% malicious (fixed identities), α=0.9 (IID) or α=0.5
(non-IID default), 2 local epochs, SGD η_s=0.5 / η_c=0.1.

---

## Motivation Figures

### Figures: Representation Variance, Collusion Patterns, Top-k Eigenvalues, Spectral Amplification, CKA Matrix, 2D PCA (`fig:variance_suppression`, `fig:collusion_patterns`, `fig:topk_eigenvalues`, `fig:norm_inflation`, `fig:cka_attack`, `fig:pca_scatter`)

All six observation figures share a single run with per-round metrics logging enabled:

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

Per-client metrics (spectral norm, eigenvalues, pairwise similarity, DAS) are
written to `outputs/.../fera_visualize/all_rounds_metrics.csv`.

---

## Main Tables

### Performance Comparison on IID CIFAR-10 (`tab:iid_main`)

One run per cell: 6 attacks × 7 defences + no-defence baseline.

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

| Attack | `model_poison_method` | `data_poison_method` |
|--------|-----------------------|----------------------|
| BadNet | `base` | `badnets` |
| Blended | `base` | `blended` |
| Edge-case | `base` | `edge_case` |
| IBA | `iba` | `iba` |
| Neurotoxin | `neurotoxin` | `pattern` |
| Chameleon | `chameleon` | `chameleon` |

`<DEFENCE>`: `unweighted_fedavg`, `fera_visualize`, `multi_krum`, `foolsgold`,
`flame`, `fltrust`, `robustlr`, `deepsight`

---

### Performance on CIFAR-100 and Tiny-ImageNet (`tab:other_datasets`)

```bash
# CIFAR-100  (pretraining checkpoint: round 1000, α=0.5)
python main.py --config-name cifar100 \
  aggregator=<DEFENCE> \
  atk_config=cifar100_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=1001 atk_config.poison_end_round=1101 \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=<cifar100_checkpoint>

# Tiny-ImageNet  (pretraining checkpoint: round 500, α=0.5)
python main.py --config-name tinyimagenet \
  aggregator=<DEFENCE> \
  atk_config=tinyimagenet_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=501 atk_config.poison_end_round=601 \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=<tinyimagenet_checkpoint>
```

---

### Performance under Non-IID Data Distributions (`tab:non_iid`)

Neurotoxin on CIFAR-10, α ∈ {0.2, 0.5, 0.7}.

```bash
python main.py --config-name cifar10 \
  aggregator=<DEFENCE> \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=<0.2|0.5|0.7> num_rounds=100 seed=42 \
  checkpoint=<cifar10_alpha_matched_checkpoint>
```

---

### Non-IID Severity Sweep across Seven Attacks (`tab:iid_sensitivity`)

FeRA only, CIFAR-10, α ∈ {0.1, 0.2, 0.5, 0.7, 1.0}.

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=<TRIGGER> \
  atk_config.model_poison_method=<ATTACK> \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=<0.1|0.2|0.5|0.7|1.0> num_rounds=100 seed=42 \
  checkpoint=<alpha_matched_checkpoint>
```

---

### Cross-Silo FL — 10 Clients, Full Participation (`tab:cross_device`)

Pixel-pattern attack, IID and non-IID (α=0.5).

```bash
python main.py --config-name cifar10 \
  aggregator=<DEFENCE> \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=base \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  num_clients=10 num_clients_per_round=10 \
  alpha=<0.9|0.5> num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

---

### A3FL on CIFAR-10 (`tab:a3fl_cifar10`)

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

### Backdoor Accuracy under A3FL across Non-IID Levels and Datasets (`fig:a3fl_iid_performance`)

Same command with `alpha=<0.1|0.2|0.5|0.7|1.0>`, repeated for `--config-name cifar100`
and `tinyimagenet` using their respective checkpoints.

---

### Adaptive BadNet Attack (`tab:adaptive_attacks`)

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

### Decorrelated Backdoor Attack (`tab:decorr_attack`)

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

### FeRA under No-Attack Conditions (`tab:no_attack_convergence`)

```bash
# FeRA
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

### Poisoning Percentage Impact (`tab:poison_percentage`)

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.fraction_adversaries=<0.05|0.10|0.20|0.30> \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

---

### EMNIST and FEMNIST Performance (`tab:mnist_family_shallow`)

```bash
python main.py --config-name <emnist|femnist> \
  aggregator=<DEFENCE> \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=<ATTACK> \
  atk_config.poison_start_round=5 atk_config.poison_end_round=100 \
  num_rounds=100 seed=42
```

### Defense Performance on MNIST and F-MNIST under DBA (`tab:more_datasets`)

```bash
python main.py --config-name <mnist|fmnist> \
  aggregator=<DEFENCE> \
  atk_config.data_poison_method=distributed \
  atk_config.model_poison_method=dba \
  atk_config.poison_start_round=5 atk_config.poison_end_round=100 \
  num_rounds=100 seed=42
```

### MNIST Performance (`tab:mnist_defended`)

```bash
python main.py --config-name mnist \
  aggregator=<DEFENCE> \
  atk_config.data_poison_method=<TRIGGER> \
  atk_config.model_poison_method=<ATTACK> \
  atk_config.poison_start_round=5 atk_config.poison_end_round=100 \
  num_rounds=100 seed=42
```

### GTSRB Performance (`fig:gtsrb`)

```bash
python main.py --config-name gtsrb \
  aggregator=<DEFENCE> \
  atk_config=gtsrb_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=<ATTACK> \
  alpha=0.5 num_rounds=100 seed=42
```

---

## Ablation Tables

### Detection Mechanism Ablation (`tab:ablation`)

CIFAR-10, α=0.5, Neurotoxin / BadNet. Toggle individual filters:

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=<badnets|pattern> \
  atk_config.model_poison_method=<base|neurotoxin> \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth

# Consistency filter only (disable Norm-Inflation)
  aggregator_config.fera_visualize.scaled_norm_filter.enabled=false

# Norm-Inflation filter only (disable Consistency)
  aggregator_config.fera_visualize.default_filter.enabled=false
```

---

### Root Dataset Size Sensitivity (`tab:root_size`)

In-distribution root, Neurotoxin, CIFAR-10, α=0.5.

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

### Root Dataset Size Sensitivity — OOD Root (`tab:root_size_ood`)

CIFAR-100 as OOD root dataset.

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

### Performance across Model Architectures (`tab:architectures`)

Pixel-pattern attack, poisoning from round 1200 for 250 rounds.

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=base \
  atk_config.poison_start_round=1200 atk_config.poison_end_round=1450 \
  model=<resnet18|resnet34|vgg16> \
  alpha=0.5 num_rounds=250 seed=42
```

---

### Representation Extraction Depth (`tab:layer_selection`)

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

### Feature Dimension Impact (`tab:feature_dimension`)

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

### DAS Combined Score Weight Ablation (`tab:weight_ablation`)

Neurotoxin, CIFAR-10, α=0.5. Consistency filter active; other filters disabled.

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
  aggregator_config.fera_visualize.collusion_filter.enabled=false \
  aggregator_config.fera_visualize.outlier_filter.enabled=false \
  aggregator_config.fera_visualize.scaled_norm_filter.enabled=false \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

---

### Consistency Filter Threshold Sensitivity (`tab:consistency_ablation`)

Neurotoxin, CIFAR-10, α=0.5, 10% malicious clients.

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  aggregator_config.fera_visualize.default_filter.combined_threshold=<0.40|0.50|0.60> \
  aggregator_config.fera_visualize.default_filter.tda_threshold=<0.40|0.50|0.60> \
  aggregator_config.fera_visualize.default_filter.mutual_sim_threshold=<0.80|0.70|0.60> \
  alpha=0.5 num_rounds=100 seed=42 \
  checkpoint=checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth
```

Default paper values: Combined ≤ 50%, DAS ≤ 50%, MutualSim ≥ 60%.

---

### Norm-Inflation Filter MAD Threshold Ablation (`tab:fera-norm-ablation`)

Anticipate attack, CIFAR-10, α=0.5. Only Norm-Inflation filter active.

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

### BSP versus Discard Aggregation (`tab:bsp_vs_discard`)

```bash
# BSP (default)
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
