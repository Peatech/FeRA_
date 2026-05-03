import torch
from torch.utils.data import DataLoader, Subset
from typing import Optional
from logging import INFO
from backfed.utils.system_utils import log

def create_root_dataset_loader(
    testset,
    root_size: int,
    batch_size: int = None,
    num_workers: int = 0,
    device: torch.device = None,
    ood_dataset=None,
) -> DataLoader:
    if batch_size is None:
        batch_size = root_size
    source_dataset = ood_dataset if ood_dataset is not None else testset
    total_samples = len(source_dataset)
    if root_size > total_samples:
        log(INFO, f"Requested root_size {root_size} > available samples {total_samples}. Using all samples.")
        root_size = total_samples
    indices = torch.randperm(total_samples)[:root_size].tolist()
    root_subset = Subset(source_dataset, indices)
    root_loader = DataLoader(
        root_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device and device.type == "cuda" else False,
    )
    return root_loader

def load_ood_dataset(base_dataset: str, config) -> Optional[torch.utils.data.Dataset]:
    from torchvision import datasets, transforms

    base_dataset_upper = base_dataset.upper()
    try:
        if base_dataset_upper == "CIFAR10":
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ]
            )
            return datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
        if base_dataset_upper == "EMNIST":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            return datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        if base_dataset_upper == "FEMNIST":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1722,), (0.3309,))])
            return datasets.EMNIST(root="./data", split="byclass", train=False, download=True, transform=transform)
        return None
    except Exception as e:
        log(INFO, f"Failed to load OOD dataset: {e}")
        return None
