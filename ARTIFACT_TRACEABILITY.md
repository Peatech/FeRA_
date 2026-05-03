# Artifact Traceability — FeRA (ACM CCS Artifact Evaluation)

Every figure and table in the submitted paper is listed below with the exact
`main.py` command (or Hydra config path) needed to reproduce it from this
repository.

**General conventions**

- All experiments use `training_mode=sequential` unless the command shows `parallel`.
- `checkpoint=null` starts training from scratch; supply a pretrained `.pth` for
  continued-training experiments.
- Results are logged to `outputs/<run_name>/.../*.csv`.
- Substitute `num_gpus=0.25` / `num_cpus=4` for a single-GPU machine.

---

## Figures

### Figure: `fig:variance_suppression`
_Representation variance distribution (benign vs. backdoored clients)._

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  aggregator_config.fera_visualize.log_ranked_metric_tables=true \
  atk_config.model_poison_method=neurotoxin \
  atk_config.data_poison_method=pattern \
  num_rounds=100 seed=42 alpha=0.5
```
Variance metrics are written to `outputs/.../fera_visualize/all_rounds_metrics.csv`
(columns `spectral_norm`, `trace`, `eigenvalue_1`…`eigenvalue_5`).

---

### Figure: `fig:collusion_patterns`
_Pairwise cosine similarity heatmap (3 malicious, 7 benign)._

Same run as above; pairwise similarity values are in column `mutual_similarity`
of `all_rounds_metrics.csv`.

---

### Figure: `fig:topk_eigenvalues`
_Top-k eigenvalue spectra (malicious vs. benign)._

Same run; columns `eigenvalue_1` … `eigenvalue_5` per client per round.

---

### Figure: `fig:norm_inflation`
_Spectral-ratio amplification under scaling attacks._

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  aggregator_config.fera_visualize.log_ranked_metric_tables=true \
  atk_config.model_poison_method=anticipate \
  atk_config.data_poison_method=pattern \
  num_rounds=100 seed=42 alpha=0.5
```
Column `spectral_norm` (relative to per-round median) in `all_rounds_metrics.csv`.

---

### Figure: `fig:cka_attack` and `fig:pca_scatter`
_CKA similarity matrix / 2D PCA of representation deltas._

These are illustrative observation figures produced by running any attack with
`log_ranked_metric_tables=true` and extracting the representation tensors from
`all_rounds_metrics.csv` using any PCA / CKA script.

---

### Figure: `fig:a3fl_iid_performance`
_BA under A3FL across Dirichlet α levels (CIFAR-10, CIFAR-100, TinyImageNet)._

Produced from Table `tab:a3fl_cifar10` data (see below) and corresponding runs
on CIFAR-100 / TinyImageNet with `--config-name cifar100` / `tinyimagenet`.

---

### Figure: `fig:gtsrb`
_GTSRB performance curves._

```bash
python main.py --config-name gtsrb \
  aggregator=fera_visualize \
  atk_config.model_poison_method=neurotoxin \
  atk_config.data_poison_method=pattern \
  num_rounds=100 seed=42 alpha=0.5
```

---

## Main-paper tables

### Table `tab:iid_main`
_IID CIFAR-10 — 9 attacks × 9 defences + no-defence._

Each cell is one run; the command pattern is:

```bash
python main.py --config-name cifar10 \
  aggregator=<DEFENSE> \
  atk_config.model_poison_method=<ATTACK> \
  atk_config.data_poison_method=<TRIGGER> \
  alpha=0.9 \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2100 \
  num_rounds=100 seed=42
```

**Attacks** (`atk_config.model_poison_method`):  
`base`, `neurotoxin`, `a3fl`, `anticipate`, `chameleon`, `dba`, `iba`  
**Triggers** (`atk_config.data_poison_method`):  
`badnets`, `pattern`, `pixel`, `blended`, `distributed`, `edge_case`, `iba`, `chameleon`, `centralized`  
**Defences** (`aggregator`):  
`unweighted_fedavg`, `fera_visualize`, `multi_krum`, `foolsgold`, `flame`,
`fltrust`, `robustlr`, `deepsight`, `fldetector`

---

### Table `tab:other_datasets`
_CIFAR-100 and Tiny-ImageNet under pixel-pattern attacks._

```bash
# CIFAR-100
python main.py --config-name cifar100 \
  aggregator=<DEFENSE> \
  atk_config.data_poison_method=pattern \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2100 \
  num_rounds=100 seed=42 alpha=0.5

# TinyImageNet
python main.py --config-name tinyimagenet \
  aggregator=<DEFENSE> \
  atk_config.data_poison_method=pattern \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2100 \
  num_rounds=100 seed=42 alpha=0.5
```

---

### Table `tab:non_iid`
_Non-IID severity — Neurotoxin on CIFAR-10, α ∈ {0.5, 0.3, 0.1}._

```bash
python main.py --config-name cifar10 \
  aggregator=<DEFENSE> \
  atk_config.model_poison_method=neurotoxin \
  atk_config.data_poison_method=pattern \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2100 \
  num_rounds=100 seed=42 \
  alpha=<0.5|0.3|0.1>
```

---

### Table `tab:iid_sensitivity`
_FeRA across 7 attacks × Dirichlet α values on CIFAR-10._

Same pattern as `tab:iid_main` with `aggregator=fera_visualize` only, sweeping
`alpha` over `{0.9, 0.7, 0.5, 0.3, 0.1}`.

---
hyyu
### Table `tab:cross_device`
_Cross-silo FL (10 clients, all participate)._

```bash
python main.py --config-name cifar10 \
  aggregator=<DEFENSE> \
  num_clients=10 \
  num_clients_per_round=10 \
  atk_config.data_poison_method=pattern \
  alpha=<0.9|0.5> \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2100 \
  num_rounds=100 seed=42
```

---

### Table `tab:a3fl_cifar10`
_A3FL attack on CIFAR-10._

```bash
python main.py --config-name cifar10 \
  aggregator=<DEFENSE> \
  atk_config.model_poison_method=a3fl \
  atk_config.data_poison_method=pattern \
  alpha=0.5 \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2100 \
  num_rounds=100 seed=42
```

---

### Table `tab:adaptive_attacks`
_Adaptive BadNet attack (`adaptive_badnet` client)._

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config.model_poison_method=adaptive_badnet \
  atk_config.data_poison_method=badnets \
  alpha=0.5 \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2100 \
  num_rounds=100 seed=42
```
Client implementation: `backfed/clients/adaptive_badnet_client.py`

---

### Table `tab:decorr_attack`
_Decorrelated backdoor attack._

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config.model_poison_method=decorrelated \
  atk_config.data_poison_method=badnets \
  alpha=0.5 \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2100 \
  num_rounds=100 seed=42
```
Client implementation: `backfed/clients/decorrelated_malicious_client.py`

---

## Appendix / ablation tables

### Table `tab:ablation`
_Detection mechanism contributions (filter components on/off)._

```bash
# All filters on (default)
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config.model_poison_method=<base|neurotoxin> \
  atk_config.data_poison_method=<badnets|pattern> \
  alpha=0.5 num_rounds=100 seed=42

# Consistency filter only (disable norm-inflation)
  aggregator_config.fera_visualize.scaled_norm_filter.enabled=false

# Norm-inflation filter only (disable consistency)
  aggregator_config.fera_visualize.default_filter.enabled=false
```

---

### Table `tab:root_size`
_Root dataset size sensitivity (in-distribution)._

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config.model_poison_method=neurotoxin \
  atk_config.data_poison_method=pattern \
  alpha=0.5 num_rounds=100 seed=42 \
  aggregator_config.fera_visualize.root_size=<16|32|64|128|256>
```

---

### Table `tab:root_size_ood`
_Root dataset size sensitivity (OOD: CIFAR-100 as root)._

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config.model_poison_method=neurotoxin \
  atk_config.data_poison_method=pattern \
  alpha=0.5 num_rounds=100 seed=42 \
  aggregator_config.fera_visualize.use_ood_root_dataset=true \
  aggregator_config.fera_visualize.ood_root_dataset_name=CIFAR100 \
  aggregator_config.fera_visualize.root_size=<16|32|64|128|256>
```

---

### Table `tab:architectures`
_Architecture comparison (ResNet-18, ResNet-34, VGG-16, etc.)._

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config.data_poison_method=pattern \
  alpha=0.5 seed=42 \
  model=<ResNet18|ResNet34|VGG16> \
  atk_config.poison_start_round=1200 atk_config.poison_end_round=1450 \
  num_rounds=250
```

---

### Table `tab:layer_selection`
_Representation extraction depth (penultimate vs. layer2/3/4)._

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config.model_poison_method=neurotoxin \
  atk_config.data_poison_method=pattern \
  alpha=0.5 num_rounds=100 seed=42 \
  aggregator_config.fera_visualize.extraction_layer=<penultimate|layer4|layer3|layer2>
```

---

### Table `tab:feature_dimension`
_Feature dimension impact._

Controlled via architecture (feature dim is set by the model);
`aggregator_config.fera_visualize.feature_dim` overrides the auto-detected value.

---

### Table `tab:weight_ablation`
_Combined score weight (spectral vs. delta)._

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config.model_poison_method=neurotoxin \
  alpha=0.5 num_rounds=100 seed=42 \
  aggregator_config.fera_visualize.spectral_weight=<0.3|0.5|0.6|0.7|1.0> \
  aggregator_config.fera_visualize.delta_weight=<0.7|0.5|0.4|0.3|0.0>
```

---

### Table `tab:consistency_ablation`
_Threshold sensitivity (Combined / DAS / MutualSim)._

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config.model_poison_method=neurotoxin \
  alpha=0.5 num_rounds=100 seed=42 \
  aggregator_config.fera_visualize.default_filter.combined_threshold=<0.40|0.50|0.60> \
  aggregator_config.fera_visualize.default_filter.tda_threshold=<0.40|0.50|0.60> \
  aggregator_config.fera_visualize.default_filter.mutual_sim_threshold=<0.60|0.70|0.80>
```
Default paper values: **Combined ≤ 50%, DAS ≤ 50%, MutualSim ≥ 60%**.

---

### Table `tab:fera-norm-ablation`
_MAD sensitivity parameter k for norm-inflation filter._

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config.model_poison_method=anticipate \
  alpha=0.5 num_rounds=100 seed=42 \
  aggregator_config.fera_visualize.scaled_norm_filter.enabled=true \
  aggregator_config.fera_visualize.scaled_norm_filter.spectral_ratio_threshold=<50|100|200|500>
```

---

### Table `tab:bsp_vs_discard`
_BSP vs. discard aggregation comparison._

```bash
# BSP (Benign Subspace Projection)
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  aggregator_config.fera_visualize.flagged_client_treatment=project \
  atk_config.model_poison_method=neurotoxin \
  alpha=0.5 num_rounds=100 seed=42

# Discard
  aggregator_config.fera_visualize.flagged_client_treatment=discard
```

---

### Table `tab:no_attack_convergence`
_FeRA under zero-attack conditions (FPR and MA gap vs. FedAvg)._

```bash
# FeRA — no attack, IID (α=1.0)
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  no_attack=True \
  num_rounds=100 seed=42 alpha=1.0 \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2100

# Repeat for alpha=0.5 and alpha=0.2
# FedAvg baseline (replace aggregator=unweighted_fedavg)
```
Convenience Slurm script: `slurm_noattack_cifar10_dirichlet.sbatch`
(set `RUN_ALL_ALPHAS=1` to run all three Dirichlet settings in one job).

**Empirically measured FPR (this artifact, rounds 2001–2100):**

| Setting | Mean FPR_clean |
|---------|---------------|
| IID (α = 1.0) | 0.0576 |
| Non-IID (α = 0.5) | 0.0616 |
| Non-IID (α = 0.2) | 0.0879 |

---

### Table `tab:mnist_family_shallow`
_EMNIST and FEMNIST using MnistNet._

```bash
python main.py --config-name emnist \
  aggregator=fera_visualize \
  atk_config.model_poison_method=<ATTACK> \
  num_rounds=100 seed=42

python main.py --config-name femnist \
  aggregator=fera_visualize \
  atk_config.model_poison_method=<ATTACK> \
  num_rounds=100 seed=42
```

---

### Table `tab:more_datasets`
_Additional dataset results (e.g. GTSRB)._

```bash
python main.py --config-name gtsrb \
  aggregator=<DEFENSE> \
  atk_config.model_poison_method=<ATTACK> \
  num_rounds=100 seed=42 alpha=0.5
```

---

### Table `tab:mnist_defended`
_MNIST defended variants._

```bash
python main.py --config-name mnist \
  aggregator=fera_visualize \
  atk_config.model_poison_method=<ATTACK> \
  num_rounds=100 seed=42
```

---

### Table `tab:poison_percentage`
_Varying poison percentage / malicious client fraction._

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config.model_poison_method=neurotoxin \
  alpha=0.5 num_rounds=100 seed=42 \
  atk_config.malicious_rate=<0.05|0.10|0.20|0.30>
```

---

## Key source files

| File | Role |
|------|------|
| `main.py` | Experiment entry point (Hydra). |
| `config/base.yaml` | All FeRA hyperparameter defaults. |
| `config/cifar10.yaml` | CIFAR-10 dataset / model defaults. |
| `backfed/servers/fera_visualize_server.py` | FeRA detection + aggregation logic. |
| `backfed/servers/defense_categories.py` | FPR / detection metric computation. |
| `backfed/clients/adaptive_badnet_client.py` | Adaptive BadNet (Table `tab:adaptive_attacks`). |
| `backfed/clients/decorrelated_malicious_client.py` | Decorrelated attack (Table `tab:decorr_attack`). |
| `backfed/servers/fera_attention_server.py` | Attention-based FeRA variant. |

---

*This file was generated to satisfy ACM CCS artifact evaluation requirements.*
*Every result in the submitted paper can be reproduced using the commands above*
*without modifying any source file.*
