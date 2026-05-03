# FeRA Experiments

**Checkpoints.** A small sample checkpoint is included only to enable a quick local run:

`checkpoints/CIFAR10_unweighted_fedavg/resnet18_round_2000_dir_0.9.pth`

Full pretraining checkpoints used in the paper are stored on institutional systems and are not redistributed here, to preserve anonymity.

**Seeds.** Reported results aggregate runs over multiple random seeds (e.g. 42, 123, 1337, 2024).

**Training mode.** All experiments use `training_mode=sequential` unless a command explicitly sets `training_mode=parallel`.

**Scratch training.** `checkpoint=null` starts training from scratch; supply a pretrained `.pth` for continued-training (post-pretrain attack) experiments.

## Performance Comparison on IID CIFAR-10

```bash
python main.py --config-name cifar10 \
  aggregator=<DEFENCE> \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=<TRIGGER> \
  atk_config.model_poison_method=<ATTACK> \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  alpha=1.0 num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

## Performance on CIFAR-100 and Tiny-ImageNet

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

## Performance under Non-IID Data Distributions

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

## Non-IID Severity Sweep across Seven Attacks

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

## Cross-Silo FL — 10 Clients, Full Participation

```bash
python main.py --config-name cifar10 \
  aggregator=<DEFENCE> \
  atk_config=cifar10_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=base \
  atk_config.poison_start_round=2001 atk_config.poison_end_round=2101 \
  num_clients=10 num_clients_per_round=10 \
  alpha=<1.0|0.5> num_rounds=100 seed=<SEED> \
  checkpoint=path_to_checkpoint
```

## A3FL on CIFAR-10

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

## Adaptive BadNet Attack

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

## Decorrelated Backdoor Attack

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

## FeRA under No-Attack Conditions

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

## Poisoning Percentage Impact

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

## EMNIST and FEMNIST Performance

```bash
python main.py --config-name <emnist|femnist> \
  aggregator=<DEFENCE> \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=<ATTACK> \
  atk_config.poison_start_round=5 atk_config.poison_end_round=100 \
  num_rounds=100 seed=<SEED>
```

## Defense Performance on MNIST and F-MNIST under DBA

```bash
python main.py --config-name <mnist|fmnist> \
  aggregator=<DEFENCE> \
  atk_config.data_poison_method=distributed \
  atk_config.model_poison_method=dba \
  atk_config.poison_start_round=5 atk_config.poison_end_round=100 \
  num_rounds=100 seed=<SEED>
```

## MNIST Performance

```bash
python main.py --config-name mnist \
  aggregator=<DEFENCE> \
  atk_config.data_poison_method=<TRIGGER> \
  atk_config.model_poison_method=<ATTACK> \
  atk_config.poison_start_round=5 atk_config.poison_end_round=100 \
  num_rounds=100 seed=<SEED>
```

## FeRA performance on GTSRB

```bash
python main.py --config-name gtsrb \
  aggregator=<DEFENCE> \
  atk_config=gtsrb_multishot \
  atk_config.data_poison_method=pattern \
  atk_config.model_poison_method=<ATTACK> \
  alpha=0.5 num_rounds=100 seed=<SEED>
```

## Detection Mechanism Ablation

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

## Root Dataset Size Sensitivity

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

## Root Dataset Size Sensitivity — OOD Root

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

## Performance across Model Architectures

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

## Representation Extraction Depth

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

## Feature Dimension Impact

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

## DAS Combined Score Weight Ablation

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

## Consistency Filter Threshold Sensitivity

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

## Norm-Inflation Filter MAD Threshold Ablation

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

## BSP versus Discard Aggregation

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
