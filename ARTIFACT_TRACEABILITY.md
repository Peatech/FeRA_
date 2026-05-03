# Artifact Traceability — FeRA

**Checkpoints.** Place pretrained checkpoints at:
```
checkpoints/<DATASET>_unweighted_fedavg/resnet18_round_<N>.pth
```

**Seeds.** Results are averaged over multiple random seeds (e.g. 42, 123, 1337, 2024).

---

## Performance Comparison on IID CIFAR-10 (`tab:iid_main`)

```bash
python main.py --config-name cifar10 \
  aggregator=<DEFENCE> \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=<TRIGGER> \
  atk_config.model_poison_method=<ATTACK> \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=0.9 num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

| Attack | `model_poison_method` | `data_poison_method` |
|--------|-----------------------|----------------------|
| BadNet | `base` | `badnets` |
| Blended | `base` | `blended` |
| Edge-case | `base` | `edge_case` |
| IBA | `iba` | `iba` |
| Neurotoxin | `neurotoxin` | `pattern` |
| Chameleon | `chameleon` | `chameleon` |

`<DEFENCE>`: `unweighted_fedavg`, `fera_visualize`, `multi_krum`, `foolsgold`, `flame`, `fltrust`, `robustlr`, `deepsight`

---

## Performance on CIFAR-100 and Tiny-ImageNet (`tab:other_datasets`)

```bash
# CIFAR-100
python main.py --config-name cifar100 \
  aggregator=<DEFENCE> \
  atk_config=cifar100_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=1001 atk_config.poison_end_round=1101 \
  alpha=0.5 num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint

# Tiny-ImageNet
python main.py --config-name tinyimagenet \
  aggregator=<DEFENCE> \
  atk_config=tinyimagenet_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=501 atk_config.poison_end_round=601 \
  alpha=0.5 num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

---

## Performance under Non-IID Data Distributions (`tab:non_iid`)

```bash
python main.py --config-name cifar10 \
  aggregator=<DEFENCE> \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=<0.2|0.5|0.7> num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

---

## Non-IID Severity Sweep across Seven Attacks (`tab:iid_sensitivity`)

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=<TRIGGER> \
  atk_config.model_poison_method=<ATTACK> \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=<0.1|0.2|0.5|0.7|1.0> num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

---

## Cross-Silo FL — 10 Clients, Full Participation (`tab:cross_device`)

```bash
python main.py --config-name cifar10 \
  aggregator=<DEFENCE> \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=base \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  num_clients=10 num_clients_per_round=10 \
  alpha=<0.9|0.5> num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

---

## A3FL on CIFAR-10 (`tab:a3fl_cifar10`)

```bash
python main.py --config-name cifar10 \
  aggregator=<DEFENCE> \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=a3fl \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=0.5 num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

---

## Adaptive BadNet Attack (`tab:adaptive_attacks`)

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=badnets \
  atk_config.model_poison_method=adaptive_badnet \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=0.5 num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

---

## Decorrelated Backdoor Attack (`tab:decorr_attack`)

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=badnets \
  atk_config.model_poison_method=decorrelated \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=0.5 num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

---

## FeRA under No-Attack Conditions (`tab:no_attack_convergence`)

```bash
# FeRA
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  no_attack=True \
  alpha=<1.0|0.5|0.2> num_rounds=100 seed=<SEED>

# FedAvg baseline
python main.py --config-name cifar10 \
  aggregator=unweighted_fedavg \
  no_attack=True \
  alpha=<1.0|0.5|0.2> num_rounds=100 seed=<SEED>
```

---

## Poisoning Percentage Impact (`tab:poison_percentage`)

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.fraction_adversaries=<0.05|0.10|0.20|0.30> \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=0.5 num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

---

## EMNIST and FEMNIST Performance (`tab:mnist_family_shallow`)

```bash
python main.py --config-name <emnist|femnist> \
  aggregator=<DEFENCE> \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=<ATTACK> \
  atk_config.poison_start_round=5 atk_config.poison_end_round=100 \
  num_rounds=100 seed=<SEED>
```

---

## Defense Performance on MNIST and F-MNIST under DBA (`tab:more_datasets`)

```bash
python main.py --config-name <mnist|fmnist> \
  aggregator=<DEFENCE> \
  atk_config.data_poison_method=distributed \
  atk_config.model_poison_method=dba \
  atk_config.poison_start_round=5 atk_config.poison_end_round=100 \
  num_rounds=100 seed=<SEED>
```

---

## MNIST Performance (`tab:mnist_defended`)

```bash
python main.py --config-name mnist \
  aggregator=<DEFENCE> \
  atk_config.data_poison_method=<TRIGGER> \
  atk_config.model_poison_method=<ATTACK> \
  atk_config.poison_start_round=5 atk_config.poison_end_round=100 \
  num_rounds=100 seed=<SEED>
```

---

## GTSRB Performance (`fig:gtsrb`)

```bash
python main.py --config-name gtsrb \
  aggregator=<DEFENCE> \
  atk_config=gtsrb_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=<ATTACK> \
  alpha=0.5 num_rounds=100 seed=<SEED>
```

---

## Detection Mechanism Ablation (`tab:ablation`)

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=<badnets|pattern> \
  atk_config.model_poison_method=<base|neurotoxin> \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  aggregator_config.fera_visualize.scaled_norm_filter.enabled=<true|false> \
  aggregator_config.fera_visualize.default_filter.enabled=<true|false> \
  alpha=0.5 num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

---

## Root Dataset Size Sensitivity (`tab:root_size`)

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  aggregator_config.fera_visualize.root_size=<16|32|64|128|256> \
  alpha=0.5 num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

---

## Root Dataset Size Sensitivity — OOD Root (`tab:root_size_ood`)

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
  alpha=0.5 num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

---

## Performance across Model Architectures (`tab:architectures`)

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=base \
  atk_config.poison_start_round=1200 atk_config.poison_end_round=1450 \
  model=<resnet18|resnet34|vgg16> \
  alpha=0.5 num_rounds=250 seed=<SEED>
```

---

## Representation Extraction Depth (`tab:layer_selection`)

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  aggregator_config.fera_visualize.extraction_layer=<penultimate|layer4|layer3|layer2> \
  alpha=0.5 num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

---

## Feature Dimension Impact (`tab:feature_dimension`)

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  aggregator_config.fera_visualize.feature_dim=<64|128|256|512> \
  alpha=0.5 num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

---

## DAS Combined Score Weight Ablation (`tab:weight_ablation`)

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
  alpha=0.5 num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

---

## Consistency Filter Threshold Sensitivity (`tab:consistency_ablation`)

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
  alpha=0.5 num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

---

## Norm-Inflation Filter MAD Threshold Ablation (`tab:fera-norm-ablation`)

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
  alpha=0.5 num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

---

## BSP versus Discard Aggregation (`tab:bsp_vs_discard`)

```bash
python main.py --config-name cifar10 \
  aggregator=fera_visualize \
  aggregator_config.fera_visualize.flagged_client_treatment=<project|discard> \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=neurotoxin \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=0.5 num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
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
