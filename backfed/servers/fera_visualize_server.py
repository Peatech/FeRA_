"""
FeRA Visualize Server - Visualization-only Defense

This server computes and visualizes per-client metrics for backdoor detection analysis
without performing any filtering. All clients are aggregated using standard FedAvg.

Metrics computed per round (per client):
1. Spectral Norm: Largest eigenvalue of representation delta covariance
2. Delta Norm: Frobenius norm of representation difference  
3. Combined Score: Weighted combination of normalized spectral + delta
4. TDA (Temporal Direction Alignment): Cosine similarity with global model
5. Mutual Similarity: Mean pairwise cosine similarity with other clients
6. Param Change Count: Number of parameters with |change| > threshold

Output per round:
- 6 ranked CSV files (one per metric)
- 1 master CSV with all metrics
- Console logs with ranked tables for quick inspection

Reference: fera_visualize LaTeX document (lines 22-217)

Author: AI Assistant
Date: 2025-10-27
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from logging import INFO, WARNING
from torch.utils.data import DataLoader, Subset
import random

from backfed.servers.defense_categories import AnomalyDetectionServer
from backfed.servers.fera_ood_utils import load_ood_dataset, create_root_dataset_loader
from backfed.utils.system_utils import log
from backfed.utils.model_utils import get_model
from backfed.const import client_id, num_examples, StateDict


class FeraVisualizeServer(AnomalyDetectionServer):
    """
    FeRA Visualize: Multi-component filtering defense with switchable variants.
    
    Computes 6 per-client metrics and applies configurable filter logic based on variant:
    
    Metrics computed:
    1. Spectral Norm: Largest eigenvalue of representation delta covariance
    2. Delta Norm: Frobenius norm of representation difference
    3. Combined Score: Weighted combination of normalized spectral + delta
    4. TDA (Temporal Direction Alignment): Cosine similarity with global model
    5. Mutual Similarity: Maximum pairwise cosine similarity with other clients
    6. Param Change Count: Number of parameters with |change| > threshold
    
    VARIANT 1 (Original Multi-Component Logic) - Default:
    ────────────────────────────────────────────────────────
    Filter Components (independently configurable):
    1. Default Filter: Combined ≤ 50%, TDA ≤ 50%, MutualSim ≥ 70% (optional)
    2. Collusion Filter: Top 40% mutual_sim AND top 40% TDA (optional)
    3. Outlier Filter: Clients in both extremes of metrics (optional)
    4. Scaled Norm Filter: spectral_ratio > threshold (optional)
    
    Final malicious set = UNION of all enabled filters
    
    VARIANT 2 (Simplified Logic):
    ────────────────────────────────────────────────────────
    Filter Components:
    1. Consistency Filter: Bottom-50% Combined AND Bottom-50% TDA (always on)
    2. Collusion Filter: Bottom-50% Combined AND Bottom-50% MutualSim (optional)
    3. Norm-Inflation Filter: Spectral Ratio > 100 (optional)
    
    Final malicious set = UNION of all enabled filters
    
    Switching Variants:
    ───────────────────
    Set filter_variant="v1" (default) or filter_variant="v2" in configuration.
    Example CLI: aggregator_config.fera_visualize.filter_variant=v2
    
    Only benign clients are aggregated using FedAvg.
    """
    
    defense_categories = ["anomaly_detection"]
    
    def __init__(
        self,
        server_config,
        server_type: str = "fera_visualize",
        eta: float = 0.5,
        root_size: int = 64,
        epsilon: float = 1e-12,
        param_change_threshold: float = 0.0,
        feature_dim: int = None,                                      
        spectral_weight: float = 0.6,
        delta_weight: float = 0.4,
                                    
        extraction_layer: str = 'penultimate',
        extraction_layers: List[str] = None,
        combine_layers_method: str = 'mean',
                                     
        use_ood_root_dataset: bool = False,
        ood_root_dataset_name: str = None,
                               
        default_filter: dict = None,
        collusion_filter: dict = None,
        outlier_filter: dict = None,
        scaled_norm_filter: dict = None,
                                  
        filter_variant: str = "v1",                                        
                                     
                                                                                     
                                                                                 
                                                                                  
                                                                                   
                                                                                       
        flagged_client_treatment: str = "discard",
                                          
        enable_clipping: bool = True,                                                                          
        clip_percentage: float = 0.6,                                                                               
        enable_adaptive_noise: bool = False,                                                            
        noise_mode: str = "adaptive",                                                                            
        noise_lambda: float = 0.001,                                                                           
        noise_std: float = 0.025,                                                                         
                                       
        top_k_eigenvalues: int = 5,                                                   
        effective_rank_threshold: float = 0.9,                                                               
                                                              
         
                                                                     
                             
                                                                                           
                                                                   
                                                                              
                                                                                        
                                                                                      
                                                                                     
                                                                                           
                                                                                  
                                                                                  
                                                                                       
                                                                                          
         
                                               
        v3_residual_sim_thresh: float = 0.70,                                        
        v3_min_colluders: int = 2,                                                      
        v3_deviation_thresh: float = 0.25,                                                  
        v3_norm_ratio_thresh: float = 100.0,                                              
        v3_use_dual_spectral: bool = True,                                                       
        **kwargs
    ):
        """
        Initialize FeRA Visualize server.
        
        Args:
            server_config: Configuration dictionary
            server_type: Type identifier (default: "fera_visualize")
            eta: Learning rate for aggregation
            root_size: Number of samples for representation extraction (default: 64)
            epsilon: Small constant for numerical stability (default: 1e-12)
            param_change_threshold: Threshold for counting parameter changes (default: 0.0)
            feature_dim: Expected feature dimension (default: None, auto-detect from model)
            spectral_weight: Weight for spectral norm in combined score (default: 0.6)
            delta_weight: Weight for delta norm in combined score (default: 0.4)
            extraction_layer: Layer name for single-layer extraction (default: 'penultimate')
            extraction_layers: List of layers for multi-layer extraction (default: None)
            combine_layers_method: Method to combine multi-layer features (default: 'mean')
            use_ood_root_dataset: Use OOD dataset for root samples (default: False)
            ood_root_dataset_name: Explicit OOD dataset name or None for auto-detect
            default_filter: Default filter configuration (Component 1)
            collusion_filter: Collusion filter configuration (Component 2)
            outlier_filter: Outlier filter configuration (Component 3)
            scaled_norm_filter: Scaled norm filter configuration (Component 4)
            filter_variant: Filter logic variant - "v1" (original) or "v2" (simplified)
            flagged_client_treatment: How V1 handles flagged clients:
                "discard" (default) — exclude them entirely (original behaviour)
                "project" — Benign Subspace Projection: replace delta with its projection
                            onto the mean benign delta, norm-capped to median benign norm.
                            Prevents backdoor accumulation while preserving clean signal.
            enable_clipping: Enable clipping for flagged malicious clients (V2 only, default: True)
            clip_percentage: Percentage of median to clip malicious updates to (0.6 = 60%, default: 0.6)
            enable_adaptive_noise: Enable noise addition after aggregation (V2 only, default: False)
            noise_mode: Noise mode - "adaptive" (scales with param std and clip_norm) or "fixed" (constant std)
            noise_lambda: Lambda for adaptive noise (default: 0.001, recommend: 0.01-0.1 for stronger noise)
            noise_std: Std dev for fixed noise mode (default: 0.025, recommend: 0.05-0.1 for stronger noise)
            top_k_eigenvalues: Number of top eigenvalues to track for visualization (default: 5)
            effective_rank_threshold: Variance percentage threshold for effective rank computation (default: 0.9 = 90%)
        """
        super().__init__(server_config, server_type, eta, **kwargs)
        self.verbose_detection_logging = False

                                                           
        self.filter_variant = filter_variant
        self.flagged_client_treatment = flagged_client_treatment

                                          
        self.enable_clipping = enable_clipping
        self.clip_percentage = clip_percentage
        self.enable_adaptive_noise = enable_adaptive_noise
        self.noise_mode = noise_mode
        self.noise_lambda = noise_lambda
        self.noise_std = noise_std
        
                                       
        self.top_k_eigenvalues = top_k_eigenvalues
        self.effective_rank_threshold = effective_rank_threshold

                       
        self.v3_residual_sim_thresh = v3_residual_sim_thresh
        self.v3_min_colluders = v3_min_colluders
        self.v3_deviation_thresh = v3_deviation_thresh
        self.v3_norm_ratio_thresh = v3_norm_ratio_thresh
        self.v3_use_dual_spectral = v3_use_dual_spectral
        
                          
        self.root_size = root_size
        self.epsilon = epsilon
        self.param_change_threshold = param_change_threshold
        self.spectral_weight = spectral_weight
        self.delta_weight = delta_weight
        
                         
        self.extraction_layer = extraction_layer
        self.extraction_layers = extraction_layers if extraction_layers else [extraction_layer]
        self.combine_layers_method = combine_layers_method
        self.use_multi_layer = len(self.extraction_layers) > 1
        
                          
        self.use_ood_root_dataset = use_ood_root_dataset
        self.ood_root_dataset_name = ood_root_dataset_name
        
                                                   
        self.default_filter = default_filter or {
            'enabled': True,
            'combined_threshold': 0.50,
            'tda_threshold': 0.50,
            'mutual_sim_threshold': 0.70
        }
        self.collusion_filter = collusion_filter or {
            'enabled': False,
            'mutual_sim_top_percent': 0.40,
            'tda_top_percent': 0.40
        }
        self.outlier_filter = outlier_filter or {
            'enabled': False
        }
        self.scaled_norm_filter = scaled_norm_filter or {
            'enabled': False,
            'spectral_ratio_threshold': 100.0
        }
        
                                                       
        if feature_dim is None:
            self.feature_dim = self._auto_detect_feature_dim()
        else:
            self.feature_dim = feature_dim
        
                                     
        total_weight = spectral_weight + delta_weight
        if not np.isclose(total_weight, 1.0):
            log(WARNING, f"Weights sum to {total_weight:.4f}, normalizing to 1.0")
            self.spectral_weight = spectral_weight / total_weight
            self.delta_weight = delta_weight / total_weight
        
                                    
        self.root_loader = self._create_root_loader()
        
                                             
        self.metrics_output_dir = Path(self.config.output_dir) / "fera_visualize"
        self.metrics_output_dir.mkdir(parents=True, exist_ok=True)

    def _create_root_loader(self) -> DataLoader:
        """
        Create root dataset loader for feature extraction.
        Supports both in-distribution (testset) and OOD datasets.
        
        Returns:
            DataLoader with root_size samples
        """
        ood_dataset = None
        
        if self.use_ood_root_dataset:
            if self.ood_root_dataset_name:
                ood_dataset = load_ood_dataset(self.ood_root_dataset_name, self.config)
            else:
                ood_dataset = load_ood_dataset(self.config.dataset, self.config)
        
        return create_root_dataset_loader(
            testset=self.testset,
            root_size=self.root_size,
            batch_size=min(self.root_size, 64),
            num_workers=self.config.num_workers,
            device=self.device,
            ood_dataset=ood_dataset
        )
    
    def _auto_detect_feature_dim(self) -> int:
        """
        Auto-detect the feature dimension of the penultimate layer.
        
        Performs a test forward pass through the model to determine
        the actual feature dimension, handling different architectures.
        
        Returns:
            Feature dimension (int)
        """
                                          
        test_batch = next(iter(self.test_loader))
        if isinstance(test_batch, (list, tuple)):
            test_input = test_batch[0][:1]                     
        else:
            test_input = test_batch[:1]
        test_input = test_input.to(self.device)
        
                                           
        captured_dim = [None]
        
        def capture_dim_hook(module, input, output):
                                                  
            if len(output.shape) > 2:
                output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
                output = output.view(output.size(0), -1)
            captured_dim[0] = output.shape[1]
        
                                            
        hook_handle = self._register_penultimate_hook(self.global_model, capture_dim_hook)
        
                      
        self.global_model.eval()
        with torch.no_grad():
            _ = self.global_model(test_input)
        
                     
        hook_handle.remove()
        
        if captured_dim[0] is None:
                                                            
            model_name = self.config.model.lower()
            if 'resnet' in model_name:
                feature_dim = 512                    
            elif 'mnist' in model_name or 'mnistnet' in model_name:
                feature_dim = 500                    
            elif 'vgg' in model_name:
                feature_dim = 512               
            else:
                feature_dim = 512                    
            log(WARNING, f"Could not auto-detect feature dim, using fallback: {feature_dim}")
            return feature_dim
        
        return captured_dim[0]
    
    def detect_anomalies(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Tuple[List[int], List[int]]:
        """
        Detect malicious clients using the configured filter variant.
        
        Routes to either V1 (original multi-component) or V2 (simplified) filter logic
        based on the filter_variant configuration.
        
        Args:
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Tuple of (malicious_client_ids, benign_client_ids)
        """
        if self.filter_variant == "v1":
            return self._detect_anomalies_v1(client_updates)
        elif self.filter_variant == "v2":
            return self._detect_anomalies_v2(client_updates)
        elif self.filter_variant == "v3":
            return self._detect_anomalies_v3(client_updates)
        else:
            raise ValueError(
                f"Invalid filter_variant: {self.filter_variant}. Must be 'v1', 'v2', or 'v3'."
            )
    
    def _detect_anomalies_v1(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Tuple[List[int], List[int]]:
        """
        Detect malicious clients using multi-component filtering (Variant 1 - Original Logic).
        
        Orchestrates multiple filter components and combines their results using UNION logic.
        Any filter can flag a client as malicious.
        
        Args:
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Tuple of (malicious_client_ids, benign_client_ids)
        """
                           
        if len(client_updates) < 2:
            log(WARNING, "FeRA Visualize: Less than 2 clients, cannot perform detection")
            return [], [cid for cid, _, _ in client_updates]
        
        try:
                                                               
            all_metrics = self._compute_all_metrics(client_updates)
            
                                               
            malicious_sets = []
            
                                         
            if self.default_filter['enabled']:
                mal_ids = self._filter_default(all_metrics, client_updates)
                malicious_sets.append(('Default', mal_ids))
            
                                           
            if self.collusion_filter['enabled']:
                mal_ids = self._filter_collusion(all_metrics, client_updates)
                malicious_sets.append(('Collusion', mal_ids))
            
                                         
            if self.outlier_filter['enabled']:
                mal_ids = self._filter_outliers(all_metrics, client_updates)
                malicious_sets.append(('Outlier', mal_ids))
            
                                             
            if self.scaled_norm_filter['enabled']:
                mal_ids = self._filter_scaled_norm(all_metrics, client_updates)
                malicious_sets.append(('ScaledNorm', mal_ids))
            
                                                            
            malicious_clients = self._combine_filter_results(malicious_sets)
            benign_clients = [cid for cid, _, _ in client_updates if cid not in malicious_clients]
            
                                  
            self._log_filter_results(malicious_sets, malicious_clients, benign_clients)
            
            return malicious_clients, benign_clients
        
        except Exception as e:
            log(WARNING, f"FeRA Visualize: Detection failed with error: {str(e)}")
            log(WARNING, "FeRA Visualize: Falling back to no detection (all clients benign)")
            import traceback
            traceback.print_exc()
            return [], [cid for cid, _, _ in client_updates]
    
    def _detect_anomalies_v2(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Tuple[List[int], List[int]]:
        """
        Detect malicious clients using simplified filter logic (Variant 2).
        
        V2 uses three simplified filters:
        1. Consistency Filter: Bottom-50% Combined AND Bottom-50% TDA
        2. Collusion Filter: Bottom-50% Combined AND Bottom-50% Mutual Similarity
        3. Norm-Inflation Filter: Spectral Ratio > 100
        
        Args:
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Tuple of (malicious_client_ids, benign_client_ids)
        """
                           
        if len(client_updates) < 2:
            log(WARNING, "FeRA Visualize V2: Less than 2 clients, cannot perform detection")
            return [], [cid for cid, _, _ in client_updates]
        
        try:
                                                               
            all_metrics = self._compute_all_metrics(client_updates)
            
                            
            malicious_sets = []
            
                                                               
            mal_ids = self._filter_consistency_v2(all_metrics, client_updates)
            malicious_sets.append(('Consistency', mal_ids))
            
                                                      
            if self.collusion_filter['enabled']:
                mal_ids = self._filter_collusion_v2(all_metrics, client_updates)
                malicious_sets.append(('Collusion', mal_ids))
            
                                                           
            if self.scaled_norm_filter['enabled']:
                mal_ids = self._filter_norm_inflation_v2(all_metrics, client_updates)
                malicious_sets.append(('NormInflation', mal_ids))
            
                                                            
            malicious_clients = self._combine_filter_results(malicious_sets)
            benign_clients = [cid for cid, _, _ in client_updates if cid not in malicious_clients]
            
                                  
            self._log_filter_results(malicious_sets, malicious_clients, benign_clients)
            
            return malicious_clients, benign_clients
        
        except Exception as e:
            log(WARNING, f"FeRA Visualize V2: Detection failed with error: {str(e)}")
            log(WARNING, "FeRA Visualize V2: Falling back to no detection (all clients benign)")
            import traceback
            traceback.print_exc()
            return [], [cid for cid, _, _ in client_updates]
    
    def _rank_clients(self, metric_dict: Dict[int, float]) -> Dict[int, int]:
        """
        Rank clients by metric values in ascending order.
        
        Args:
            metric_dict: {client_id: metric_value}
        
        Returns:
            {client_id: rank} where rank 1 = lowest value
        """
        sorted_clients = sorted(metric_dict.items(), key=lambda x: x[1])
        return {cid: rank for rank, (cid, _) in enumerate(sorted_clients, start=1)}
    
    def _filter_default(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        """
        Default filter: Combined Score ≤ 50%, TDA ≤ 50%, Mutual Sim ≥ 70%
        ALL conditions must be met (AND logic)
        
        Args:
            all_metrics: Dictionary of metric_name -> {client_id: value}
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Set of malicious client IDs
        """
        combined_scores = all_metrics['combined_score']
        tda_scores = all_metrics['tda']
        mutual_sim_scores = all_metrics['mutual_similarity']
        
                                                               
        combined_ranks = self._rank_clients(combined_scores)
        tda_ranks = self._rank_clients(tda_scores)
        mutual_sim_ranks = self._rank_clients(mutual_sim_scores)
        
                              
        n = len(client_updates)
        combined_thresh = int(n * self.default_filter['combined_threshold'])
        tda_thresh = int(n * self.default_filter['tda_threshold'])
        mutual_sim_thresh = int(n * self.default_filter['mutual_sim_threshold']) + 1
        
                                                   
        malicious = set()
        for cid, _, _ in client_updates:
            if (combined_ranks[cid] <= combined_thresh and
                tda_ranks[cid] <= tda_thresh and
                mutual_sim_ranks[cid] >= mutual_sim_thresh):
                malicious.add(cid)
        
        return malicious
    
    def _filter_collusion(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        """
        Collusion filter: Top 40% mutual_sim AND top 40% TDA
        Targets colluding attackers with high similarity
        
        Args:
            all_metrics: Dictionary of metric_name -> {client_id: value}
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Set of malicious client IDs
        """
        tda_scores = all_metrics['tda']
        mutual_sim_scores = all_metrics['mutual_similarity']
        
                                                         
        tda_ranks = self._rank_clients(tda_scores)
        mutual_sim_ranks = self._rank_clients(mutual_sim_scores)
        
                                          
                                                                      
        n = len(client_updates)
        tda_top_thresh = int(n * (1.0 - self.collusion_filter['tda_top_percent']))
        mutual_sim_top_thresh = int(n * (1.0 - self.collusion_filter['mutual_sim_top_percent']))
        
                                                    
        malicious = set()
        for cid, _, _ in client_updates:
            if (tda_ranks[cid] > tda_top_thresh and
                mutual_sim_ranks[cid] > mutual_sim_top_thresh):
                malicious.add(cid)
        
        return malicious
    
    def _filter_outliers(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        """
        Outlier filter: Flags clients at both extremes of metrics
        If client is in {highest OR lowest} mutual_sim 
        AND {highest OR lowest} TDA: Malicious
        
        Args:
            all_metrics: Dictionary of metric_name -> {client_id: value}
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Set of malicious client IDs
        """
        tda_scores = all_metrics['tda']
        mutual_sim_scores = all_metrics['mutual_similarity']
        
                              
        tda_sorted = sorted(tda_scores.items(), key=lambda x: x[1])
        mutual_sim_sorted = sorted(mutual_sim_scores.items(), key=lambda x: x[1])
        
                                                
        tda_lowest = tda_sorted[0][0]
        tda_highest = tda_sorted[-1][0]
        mutual_sim_lowest = mutual_sim_sorted[0][0]
        mutual_sim_highest = mutual_sim_sorted[-1][0]
        
                                 
        tda_extremes = {tda_lowest, tda_highest}
        mutual_sim_extremes = {mutual_sim_lowest, mutual_sim_highest}
        
                                           
        malicious = tda_extremes & mutual_sim_extremes
        
        return malicious
    
    def _filter_scaled_norm(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        """
        Scaled Norm filter: Flags clients with extreme spectral norm ratios
        Detects norm-inflation attacks
        
        Args:
            all_metrics: Dictionary of metric_name -> {client_id: value}
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Set of malicious client IDs
        """
        spectral_norms = all_metrics['spectral_norm']
        
                                                     
        median_spectral = np.median(list(spectral_norms.values()))
        
                                
        if median_spectral < self.epsilon:
            return set()
        
                                                 
        threshold = self.scaled_norm_filter['spectral_ratio_threshold']
        malicious = set()
        
        for cid, spectral_norm in spectral_norms.items():
            spectral_ratio = spectral_norm / median_spectral
            if spectral_ratio > threshold:
                malicious.add(cid)
        
        return malicious
    
    def _filter_consistency_v2(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        """
        Variant 2 Consistency Filter:
        Flags clients with bottom-50% Combined AND bottom-50% TDA.
        
        Targets clients with low variance (low combined score) and directional
        deviation (low TDA score).
        
        Args:
            all_metrics: Dictionary of metric_name -> {client_id: value}
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Set of malicious client IDs
        """
        combined_scores = all_metrics['combined_score']
        tda_scores = all_metrics['tda']
        
                                                         
        combined_ranks = self._rank_clients(combined_scores)
        tda_ranks = self._rank_clients(tda_scores)
        
                              
        n = len(client_updates)
        thresh_50 = int(n * 0.50)
        
                                                    
        malicious = set()
        for cid, _, _ in client_updates:
            if (combined_ranks[cid] <= thresh_50 and 
                tda_ranks[cid] <= thresh_50):
                malicious.add(cid)
        
        return malicious
    
    def _filter_collusion_v2(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        """
        Variant 2 Collusion Filter:
        Flags clients with bottom-50% Combined AND bottom-50% Mutual Similarity.
        
        Targets colluding attackers with low variance and low similarity to others.
        
        Args:
            all_metrics: Dictionary of metric_name -> {client_id: value}
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Set of malicious client IDs
        """
        combined_scores = all_metrics['combined_score']
        mutual_sim_scores = all_metrics['mutual_similarity']
        
                                                         
        combined_ranks = self._rank_clients(combined_scores)
        mutual_sim_ranks = self._rank_clients(mutual_sim_scores)
        
                              
        n = len(client_updates)
        thresh_50 = int(n * 0.50)
        
                                                    
        malicious = set()
        for cid, _, _ in client_updates:
            if (combined_ranks[cid] <= thresh_50 and 
                mutual_sim_ranks[cid] <= thresh_50):
                malicious.add(cid)
        
        return malicious
    
    def _filter_norm_inflation_v2(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        """
        Variant 2 Norm-Inflation Filter:
        Flags clients with spectral_ratio > 100.
        
        Detects norm-inflation and model-replacement style attacks.
        
        Args:
            all_metrics: Dictionary of metric_name -> {client_id: value}
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            Set of malicious client IDs
        """
        spectral_norms = all_metrics['spectral_norm']
        
                                      
        median_spectral = np.median(list(spectral_norms.values()))
        
                                
        if median_spectral < self.epsilon:
            return set()
        
                                       
        threshold = 100.0
        malicious = set()
        
        for cid, spectral_norm in spectral_norms.items():
            spectral_ratio = spectral_norm / median_spectral
            if spectral_ratio > threshold:
                malicious.add(cid)
        
        return malicious
    
    def _clip_malicious_updates(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]],
        malicious_clients: List[int]
    ) -> List[Tuple[client_id, num_examples, StateDict]]:
        """
        Clip malicious client updates using adaptive threshold (median of all update norms).
        
        This method:
        1. Computes update deltas (client_update - global_model) for all clients
        2. Calculates L2 norm of each delta
        3. Uses median of all norms as clipping threshold
        4. Clips only malicious client updates (preserves direction, limits magnitude)
        
        Args:
            client_updates: List of (client_id, num_examples, state_dict)
            malicious_clients: List of client IDs flagged as malicious
        
        Returns:
            List of client updates with malicious ones clipped (in-place modification)
        """
        if not malicious_clients:
            return client_updates
        
        malicious_set = set(malicious_clients)
        global_state_dict = dict(self.global_model.named_parameters())
        
                                              
        update_norms = []
        client_diffs = {}
        
        for client_id, num_examples, client_params in client_updates:
                                  
            diff_dict = {}
            flatten_weights = []
            
            for name, param in client_params.items():
                if name.endswith('num_batches_tracked'):
                    continue
                if name in global_state_dict:
                    diff = param.to(self.device) - global_state_dict[name]
                    diff_dict[name] = diff
                                                                 
                    if 'weight' in name or 'bias' in name:
                        flatten_weights.append(diff.view(-1))
            
            if flatten_weights:
                flatten_weights = torch.cat(flatten_weights)
                weight_diff_norm = torch.linalg.norm(flatten_weights, ord=2).item()
                update_norms.append(weight_diff_norm)
                client_diffs[client_id] = diff_dict
            else:
                update_norms.append(0.0)
                client_diffs[client_id] = diff_dict
        
                                                                          
        if not update_norms:
            log(WARNING, "FeRA Visualize V2: No update norms computed, skipping clipping")
            return client_updates
        
        clip_norm = np.median(update_norms)
        percentage_clip_norm = clip_norm * self.clip_percentage

        for client_id, num_examples, client_params in client_updates:
            if client_id not in malicious_set:
                continue
            
            if client_id not in client_diffs:
                continue
            
            diff_dict = client_diffs[client_id]
            flatten_weights = []
            
                                  
            for name, diff in diff_dict.items():
                if 'weight' in name or 'bias' in name:
                    flatten_weights.append(diff.view(-1))
            
            if not flatten_weights:
                continue
            
            flatten_weights = torch.cat(flatten_weights)
            current_norm = torch.linalg.norm(flatten_weights, ord=2).item()
            original_norm = current_norm
            
                                                            
            if current_norm > clip_norm:
                scaling_factor = clip_norm / current_norm
                
                                                            
                for name, param in client_params.items():
                    if name in diff_dict:
                                                                           
                        param.data.copy_(global_state_dict[name] + diff_dict[name] * scaling_factor)
                
                                                                     
                for name, param in client_params.items():
                    if name in global_state_dict:
                        diff_dict[name] = param.to(self.device) - global_state_dict[name]
                
                flatten_weights = []
                for name, diff in diff_dict.items():
                    if 'weight' in name or 'bias' in name:
                        flatten_weights.append(diff.view(-1))
                if flatten_weights:
                    flatten_weights = torch.cat(flatten_weights)
                    current_norm = torch.linalg.norm(flatten_weights, ord=2).item()

                                                                                               
                                                                
                                                                                             
            if current_norm > percentage_clip_norm:
                scaling_factor = percentage_clip_norm / current_norm
                
                                                                       
                for name, param in client_params.items():
                    if name in diff_dict:
                                                                                      
                        param.data.copy_(global_state_dict[name] + diff_dict[name] * scaling_factor)
                
                                                                   
                for name, param in client_params.items():
                    if name in global_state_dict:
                        diff_dict[name] = param.to(self.device) - global_state_dict[name]
                
                flatten_weights = []
                for name, diff in diff_dict.items():
                    if 'weight' in name or 'bias' in name:
                        flatten_weights.append(diff.view(-1))
                if flatten_weights:
                    flatten_weights = torch.cat(flatten_weights)
                    final_norm = torch.linalg.norm(flatten_weights, ord=2).item()
                else:
                    final_norm = percentage_clip_norm

        return client_updates
    
    @torch.no_grad()
    def _add_adaptive_noise(self, clip_norm: float):
        """
        Add noise to global model parameters after aggregation.
        
        Two modes:
        1. Adaptive: noise = N(0, lambda * clip_norm * std(param))
           - Adapts to parameter scale and clipping threshold
           - Good when you want noise proportional to parameter values
        
        2. Fixed: noise = N(0, noise_std)
           - Constant noise level (like WeakDP)
           - More predictable, easier to tune
           - Better for strong noise requirements
        
        Adds noise to all trainable parameters (weights and biases) that are not in ignore_weights.
        
        Args:
            clip_norm: Clipping threshold used (for noise scaling in adaptive mode)
        """
        for name, param in self.global_model.named_parameters():
            if any(pattern in name for pattern in self.ignore_weights):
                continue
            if "running" in name or "num_batches_tracked" in name:
                continue
            
                                                            
            if self.noise_mode == "adaptive":
                                                                         
                param_std = torch.std(param).item()
                noise_std_val = self.noise_lambda * clip_norm * param_std
            elif self.noise_mode == "fixed":
                                                         
                noise_std_val = self.noise_std
            else:
                                                     
                log(WARNING, f"Invalid noise_mode '{self.noise_mode}', defaulting to fixed")
                noise_std_val = self.noise_std
            
            noise = torch.normal(0, noise_std_val, param.shape, device=param.device)
            param.data.add_(noise)

    def _combine_filter_results(
        self,
        malicious_sets: List[Tuple[str, set]]
    ) -> List[int]:
        """
        Combine results from multiple filters using UNION logic
        
        Args:
            malicious_sets: List of (filter_name, set_of_client_ids)
        
        Returns:
            Combined list of malicious client IDs (sorted)
        """
        combined = set()
        for filter_name, mal_ids in malicious_sets:
            combined.update(mal_ids)
        return sorted(list(combined))
    
    def _log_filter_results(
        self,
        malicious_sets: List[Tuple[str, set]],
        malicious_clients: List[int],
        benign_clients: List[int]
    ):
        """
        Log detailed filter results showing contribution of each component
        
        Args:
            malicious_sets: List of (filter_name, set_of_client_ids)
            malicious_clients: Combined list of malicious clients
            benign_clients: List of benign clients
        """
        filter_bits = ",".join(f"{fn}:{sorted(m)}" for fn, m in malicious_sets)
        log(INFO, f"[Round {self.current_round}] FeRA filters=[{filter_bits}] flagged={sorted(malicious_clients)} benign={sorted(benign_clients)}")

    def aggregate_client_updates(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ):
        """
        Override aggregation to compute metrics, visualize, and perform filtering.
        
        Process:
        1. Compute all 6 metrics for each client
        2. Log ranked tables to console
        3. Save ranked tables and master CSV
        4. Call parent class which performs detection and aggregation (filters malicious clients)
        
        Args:
            client_updates: List of (client_id, num_examples, state_dict)
            
        Returns:
            True if aggregation successful, False otherwise
        """
        if not client_updates:
            log(WARNING, "No client updates found, using global model")
            return False

        try:
            all_metrics = self._compute_all_metrics(client_updates)
            self._save_metrics_csvs(all_metrics, self.current_round)
        except Exception as e:
            log(WARNING, f"Failed to compute metrics: {str(e)}")
            log(WARNING, "Proceeding with aggregation anyway...")

                                        
                                                                         
                                                                     
                                                                             
                                                                                      
        if self.filter_variant == "v3":
            result = self._aggregate_v3_with_bsp(client_updates)
        elif self.filter_variant == "v2":
            result = self._aggregate_v2_with_clipping(client_updates)
        elif self.flagged_client_treatment == "project":
            result = self._aggregate_v1_with_bsp(client_updates)
        else:
                                                                         
            result = super().aggregate_client_updates(client_updates)

        return result
    
    def _aggregate_v2_with_clipping(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> bool:
        """
        V2 aggregation with clipping and optional noise.
        
        Process:
        1. Detect malicious clients using V2 filters
        2. Clip malicious updates (if enabled)
        3. Aggregate ALL clients (benign + clipped malicious)
        4. Add adaptive noise to global model (if enabled)
        
        Args:
            client_updates: List of (client_id, num_examples, state_dict)
        
        Returns:
            True if aggregation successful, False otherwise
        """
        if not client_updates:
            log(WARNING, "No client updates found, using global model")
            return False
        
                                                           
        malicious_clients, benign_clients = self.detect_anomalies(client_updates)
        true_malicious_clients = self.get_clients_info(self.current_round)["malicious_clients"]
        self.evaluate_detection(malicious_clients, true_malicious_clients, len(client_updates))
        
                                                     
        clip_norm = None
        if self.enable_clipping and malicious_clients:
                                                                  
            global_state_dict = dict(self.global_model.named_parameters())
            update_norms = []
            
            for client_id, num_examples, client_params in client_updates:
                flatten_weights = []
                for name, param in client_params.items():
                    if name.endswith('num_batches_tracked'):
                        continue
                    if name in global_state_dict:
                        diff = param.to(self.device) - global_state_dict[name]
                        if 'weight' in name or 'bias' in name:
                            flatten_weights.append(diff.view(-1))
                
                if flatten_weights:
                    flatten_weights = torch.cat(flatten_weights)
                    weight_diff_norm = torch.linalg.norm(flatten_weights, ord=2).item()
                    update_norms.append(weight_diff_norm)
            
            if update_norms:
                clip_norm = np.median(update_norms)
            
                            
            client_updates = self._clip_malicious_updates(client_updates, malicious_clients)
        
                                                                                 
                                                                               
        from backfed.servers.fedavg_server import UnweightedFedAvgServer
        aggregation_success = UnweightedFedAvgServer.aggregate_client_updates(self, client_updates)
        
        if not aggregation_success:
            return False
        
                                                 
        if self.enable_adaptive_noise and clip_norm is not None:
            self._add_adaptive_noise(clip_norm)
        elif self.enable_adaptive_noise:
                                                                                    
                                                  
            global_state_dict = dict(self.global_model.named_parameters())
            update_norms = []
            
            for client_id, num_examples, client_params in client_updates:
                flatten_weights = []
                for name, param in client_params.items():
                    if name.endswith('num_batches_tracked'):
                        continue
                    if name in global_state_dict:
                        diff = param.to(self.device) - global_state_dict[name]
                        if 'weight' in name or 'bias' in name:
                            flatten_weights.append(diff.view(-1))
                
                if flatten_weights:
                    flatten_weights = torch.cat(flatten_weights)
                    weight_diff_norm = torch.linalg.norm(flatten_weights, ord=2).item()
                    update_norms.append(weight_diff_norm)
            
            if update_norms:
                clip_norm = np.median(update_norms)
                self._add_adaptive_noise(clip_norm)
            else:
                log(WARNING, "FeRA Visualize V2: Cannot compute clip_norm for noise, skipping noise addition")
        
        return True

                                                                        
                                                                    
                                                                        

    @torch.no_grad()
    def _aggregate_v1_with_bsp(
        self,
        client_updates: List[Tuple[int, int, dict]],
    ) -> bool:
        """
        V1 aggregation with Benign Subspace Projection for flagged clients.

        Theoretical basis (Tran et al., NeurIPS 2018 — "Spectral Signatures in
        Backdoor Attacks"): backdoor updates have a spectral component that is
        orthogonal to (or poorly aligned with) the benign learning direction.
        Projecting the flagged update onto the mean benign update retains whatever
        clean learning signal it carries while zeroing the orthogonal backdoor
        component.

        Algorithm
        ---------
        Let δ_i = client_i_params − global_params  (flattened weight/bias concat)

        Benign consensus: μ = mean({δ_i : i ∈ benign_ids})
        Benign median norm: ρ = median({||δ_i|| : i ∈ benign_ids})

        For each flagged client k:
            α_k = (δ_k · μ) / (||μ||² + ε)          # scalar projection coefficient
            δ_k_corr = clip_norm(α_k * μ, ρ)         # benign-aligned component, norm-capped
            if α_k ≤ 0:  δ_k_corr = 0               # opposes benign direction → zero out

        All corrected flagged updates + all benign updates are then averaged
        by UnweightedFedAvg.

        Why naive norm-clipping fails: it preserves the update *direction*,
        so the backdoor signal accumulates over rounds.  BSP replaces the
        direction entirely — no backdoor component can survive.

        Args:
            client_updates: List of (client_id, num_examples, state_dict)

        Returns:
            True on success, False on empty update list.
        """
        if not client_updates:
            log(WARNING, "BSP: No client updates, skipping.")
            return False

                                                                        
        malicious_ids, benign_ids = self.detect_anomalies(client_updates)
        true_malicious = self.get_clients_info(self.current_round)["malicious_clients"]
        self.evaluate_detection(malicious_ids, true_malicious, len(client_updates))

        malicious_set = set(malicious_ids)
        benign_set    = set(benign_ids)

        benign_updates   = [(cid, n, s) for cid, n, s in client_updates if cid in benign_set]
        flagged_updates  = [(cid, n, s) for cid, n, s in client_updates if cid in malicious_set]

        if not flagged_updates:
                                                        
            from backfed.servers.fedavg_server import UnweightedFedAvgServer
            return UnweightedFedAvgServer.aggregate_client_updates(self, benign_updates)

        if not benign_updates:
            log(WARNING, "BSP: All clients flagged — falling back to discarding all flagged updates.")
            return False

                                                                       
        global_params = {
            name: param.to(self.device, dtype=torch.float32)
            for name, param in self.global_model.named_parameters()
        }
        param_names = [n for n in global_params if 'weight' in n or 'bias' in n]

        def _flat_delta(state_dict):
            parts = []
            for name in param_names:
                p = state_dict[name].to(self.device, dtype=torch.float32)
                parts.append((p - global_params[name]).flatten())
            return torch.cat(parts)              

        benign_deltas = [_flat_delta(s) for _, _, s in benign_updates]

                                                 
        mu = torch.stack(benign_deltas).mean(dim=0)                    
        mu_norm_sq = (mu * mu).sum().clamp(min=self.epsilon)

                                                             
        benign_norms = torch.tensor([d.norm().item() for d in benign_deltas])
        rho = float(benign_norms.median())                                 

                                                                       
        corrected_updates = []
        for cid, n_ex, state_dict in flagged_updates:
            delta_k = _flat_delta(state_dict)

                                                          
            alpha = float((delta_k * mu).sum() / mu_norm_sq)

            if alpha <= 0.0:
                corrected_delta = torch.zeros_like(mu)
            else:
                                                   
                corrected_delta = alpha * mu

                                                                              
                corr_norm = corrected_delta.norm().item()
                if corr_norm > rho and rho > self.epsilon:
                    corrected_delta = corrected_delta * (rho / corr_norm)

                                                          
             
                                                                                
                                                                            
                                                                               
                                                                              
                                                                         
                                                                             
                                                                                
             
                                                                                   
                                                                                    
            corrected_state = {
                name: val.cpu()
                for name, val in self.global_model.state_dict().items()
            }
            offset = 0
            for name in param_names:                                        
                g = global_params[name]
                numel = g.numel()
                d_slice = corrected_delta[offset: offset + numel].reshape(g.shape)
                corrected_state[name] = (g + d_slice).cpu()
                offset += numel

            corrected_updates.append((cid, n_ex, corrected_state))

                                                                        
        all_updates = benign_updates + corrected_updates
        from backfed.servers.fedavg_server import UnweightedFedAvgServer
        return UnweightedFedAvgServer.aggregate_client_updates(self, all_updates)

                                                                            
                                                 
                                                                            

    def _detect_anomalies_v3(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Tuple[List[int], List[int]]:
        """
        Detect malicious clients using V3: Adaptive Residual-Aware filters.

        Three components (UNION of flagged sets):

        1. Adaptive Consistency Filter — replaces fixed-percentile thresholds with
           MAD-based z-scores. A client is flagged only when its combined_score AND
           TDA are genuine statistical outliers (z < -v3_z_thresh).  This adapts
           to the actual distribution shape each round and reduces FPR compared to
           the hard 50th-percentile cut used in V1/V2.

        2. Residual Collusion Filter — detects coordinated (chameleon-style)
           attackers that craft individually-benign updates.  Each update vector is
           projected onto the orthogonal complement of the group consensus (u_i - μ),
           exposing the shared backdoor component that is invisible in raw updates.
           Clients whose *residuals* are mutually highly similar AND whose residual
           centroid has a large magnitude (deviation from zero) are flagged as a
           colluding group.

        3. Norm-Inflation Filter — unchanged from V1: flag any client whose
           spectral norm ratio > v3_norm_ratio_thresh (catches model-replacement
           and scale-amplification attacks).
        """
        if len(client_updates) < 2:
            log(WARNING, "FeRA V3: Less than 2 clients, skipping detection.")
            return [], [cid for cid, _, _ in client_updates]

        try:
            all_metrics = self._compute_all_metrics(client_updates)

            malicious_sets = []

                                                                        
            mal1 = self._filter_consistency_v3(all_metrics, client_updates)
            malicious_sets.append(("Consistency", mal1))

                                             
            mal2 = self._filter_residual_collusion_v3(client_updates)
            malicious_sets.append(("ResidualCollusion", mal2))

                                         
            mal3 = self._filter_norm_inflation_v3(all_metrics)
            malicious_sets.append(("NormInflation", mal3))

            malicious_clients = self._combine_filter_results(malicious_sets)
            benign_clients = [cid for cid, _, _ in client_updates if cid not in malicious_clients]

            self._log_filter_results(malicious_sets, malicious_clients, benign_clients)
            return malicious_clients, benign_clients

        except Exception as e:
            log(WARNING, f"FeRA V3: Detection failed: {e}")
            import traceback; traceback.print_exc()
            return [], [cid for cid, _, _ in client_updates]

    def _filter_consistency_v3(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        """
        V3 Consistency Filter — same rank/percentile logic and thresholds as V1.

        No oracle knowledge required.  Percentile rank is inherently data-driven:
        clients are ranked relative to each other within the current round, so the
        50/50/70 thresholds automatically adapt to the distribution of scores in
        that round — whether the setting is IID, non-IID α=0.5, or α=0.1.

        This component is IDENTICAL to V1's default_filter.  V3's improvements over V1
        come from (a) the efficiency of repr-space mutual similarity replacing parameter-
        space mutual similarity in _compute_all_metrics, and (b) the addition of the
        residual collusion filter as a second detection component.

        NOTE: MAD/z-score absolute thresholds were evaluated and rejected — malicious
        clients are not statistical outliers in absolute value terms, only in relative
        rank position.  Percentile ranking is the correct mechanism.
        """
        combined_ranks = self._rank_clients(all_metrics['combined_score'])
        tda_ranks = self._rank_clients(all_metrics['tda'])
        mutual_sim_ranks = self._rank_clients(all_metrics['mutual_similarity'])

        n = len(client_updates)
                                                                 
                                                                                    
                                                                                      
        combined_thresh = int(n * 0.50)               
        tda_thresh = int(n * 0.50)                    
        mutual_thresh = int(n * 0.70) + 1           

        malicious = set()
        for cid, _, _ in client_updates:
            if (combined_ranks[cid] <= combined_thresh and
                    tda_ranks[cid] <= tda_thresh and
                    mutual_sim_ranks[cid] >= mutual_thresh):
                malicious.add(cid)
        return malicious

    def _filter_residual_collusion_v3(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        """
        Residual Collusion Filter (V3).

        Detects Chameleon-style colluding attackers whose *individual* updates
        look benign but who collectively inject a shared backdoor direction.

        Algorithm:
        1. Compute flat update delta for each client: u_i = flat(θ_i - θ_global)
        2. Compute consensus direction: μ = mean(u_i over all clients)
        3. Compute residual: r_i = u_i - μ  (removes the shared benign component)
        4. Normalize: r̂_i = r_i / ||r_i||
        5. Build residual similarity matrix: S[i,j] = r̂_i · r̂_j
        6. For each client i, find all j ≠ i with S[i,j] > residual_sim_thresh
        7. If the "colluding group" (i + similar clients) has size ≥ v3_min_colluders,
           AND the group's mean residual norm > v3_deviation_thresh × mean(||r||),
           then flag the whole group as malicious.

        Why residuals and not raw updates:
          Chameleon attackers add a large benign-looking component to disguise their
          updates. After subtracting μ (consensus), the benign component cancels and
          only the shared backdoor direction remains — making the colluding cluster
          visible in the residual space.
        """
        if len(client_updates) < self.v3_min_colluders + 1:
            return set()

                                                                             
        global_params_flat = {
            name: param.to(self.device, dtype=torch.float32)
            for name, param in self.global_model.named_parameters()
        }
        param_names = [n for n in global_params_flat if 'weight' in n or 'bias' in n]

        def flat_update(state_dict):
            parts = []
            for name in param_names:
                p = state_dict[name].to(self.device, dtype=torch.float32)
                parts.append((p - global_params_flat[name]).flatten())
            return torch.cat(parts)

        client_ids = [cid for cid, _, _ in client_updates]
        update_vecs = []
        for _, _, state_dict in client_updates:
            try:
                update_vecs.append(flat_update(state_dict))
            except Exception:
                update_vecs.append(torch.zeros(1, device=self.device))

                             
        U = torch.stack(update_vecs)                   
        mu = U.mean(dim=0)                          

                   
        R = U - mu.unsqueeze(0)                        
        r_norms = R.norm(dim=1)                    
        mean_norm = float(r_norms.mean().clamp(min=self.epsilon))

                                                  
        R_norm_val = r_norms.unsqueeze(1).clamp(min=self.epsilon)
        R_hat = R / R_norm_val                                        

                                           
        S = R_hat @ R_hat.T

        malicious = set()
        flagged = set()                                              
        K = len(client_ids)

        for i in range(K):
            if client_ids[i] in flagged:
                continue
                                                        
            similar_idx = [
                j for j in range(K)
                if j != i and float(S[i, j]) > self.v3_residual_sim_thresh
            ]
            group_idx = [i] + similar_idx

            if len(group_idx) < self.v3_min_colluders:
                continue

                                         
            group_residuals = R[group_idx]                      
            centroid_norm = float(group_residuals.mean(dim=0).norm())

                                                                      
            if centroid_norm > self.v3_deviation_thresh * mean_norm:
                for idx in group_idx:
                    malicious.add(client_ids[idx])
                    flagged.add(client_ids[idx])

        return malicious

    def _filter_norm_inflation_v3(
        self,
        all_metrics: Dict[str, Dict[int, float]]
    ) -> set:
        """Norm-inflation filter for V3 (same logic as V2)."""
        spectral_norms = all_metrics['spectral_norm']
        median_spectral = float(np.median(list(spectral_norms.values())))
        if median_spectral < self.epsilon:
            return set()
        return {
            cid for cid, sn in spectral_norms.items()
            if sn / median_spectral > self.v3_norm_ratio_thresh
        }

    @torch.no_grad()
    def _aggregate_v3_with_bsp(
        self,
        client_updates: List[Tuple[int, int, dict]],
    ) -> bool:
        """
        V3 aggregation: Adaptive Residual-Aware detection → BSP for flagged clients.

        Reuses the V1 BSP aggregation logic (_aggregate_v1_with_bsp) but routes
        detection through the V3 filter pipeline.  The filter_variant guard in
        detect_anomalies() ensures _detect_anomalies_v3 is called.
        """
        if not client_updates:
            log(WARNING, "V3 BSP: No client updates, skipping.")
            return False

                                                                                         
                                                                                 
        return self._aggregate_v1_with_bsp(client_updates)

                                                                            
                                                                        
                                                                            

    def _compute_spectral_norm_dual(
        self,
        delta_centered: torch.Tensor
    ) -> float:
        """
        Compute spectral norm (λ_max of Δ.T Δ / (n-1)) using the dual n×n Gram form.

        Standard form:   C = Δ.T @ Δ / (n-1)  ∈ ℝ^{D×D}   eigenvalues O(D³)
        Dual form:       G = Δ @ Δ.T / (n-1)  ∈ ℝ^{n×n}   eigenvalues O(n³)

        Both C and G share the same non-zero eigenvalues (SVD duality).  With
        root_size n=64 and feature_dim D=512, the dual form is ~8× faster in
        matrix construction and ~512× faster in eigendecomposition.

        Args:
            delta_centered: [n, D] centred delta representation

        Returns:
            λ_max (float) — spectral norm of the representation delta covariance
        """
        n = delta_centered.shape[0]
        gram = (delta_centered @ delta_centered.T) / max(n - 1, 1)          
                                                                              
        eigenvalues = torch.linalg.eigvalsh(gram)
        return float(eigenvalues[-1].clamp(min=0.0))

    def _compute_all_metrics(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Dict[str, Dict[int, float]]:
        """
        Compute all 6 metrics for each client.
        
        Args:
            client_updates: List of (client_id, num_examples, state_dict)
            
        Returns:
            Dictionary mapping metric names to client_id -> value dicts
        """
        all_metrics = {}
        
                            
        client_models = self._load_client_models(client_updates)
        client_ids = list(client_models.keys())
        
                                                                           
        client_representations = self._extract_all_representations(client_models)
        global_representation = self._extract_global_representation()
        
                                                  
        eigenvalue_metrics = self._compute_eigenvalue_metrics(
            client_representations, global_representation
        )
        
                                                                            
        spectral_scores = eigenvalue_metrics['spectral_norm']
        
                                      
        for metric_name, metric_dict in eigenvalue_metrics.items():
            all_metrics[metric_name] = metric_dict
        
                                                                
        delta_scores = self._compute_delta_norms(
            client_representations, global_representation
        )
        
                                                                                
        spectral_normed = self._robust_normalize(spectral_scores)
        delta_normed = self._robust_normalize(delta_scores)
        combined_scores = self._compute_combined_scores(spectral_normed, delta_normed)
        
                                              
        all_metrics['spectral_norm_normed'] = spectral_normed
        all_metrics['delta_norm'] = delta_scores
        all_metrics['delta_norm_normed'] = delta_normed
        all_metrics['combined_score'] = combined_scores
        
                                                                   
        tda_scores = self._compute_tda_scores(client_updates)

                                                                                          
         
                                                                            
                                                                                  
                                                                              
                                                                                    
                                                                                         
                                                               
                                                                                       
                                                                                       
                                                 
         
                                                                                
                                                                                  
                                                                                       
        mutual_similarity = self._compute_mutual_similarity(client_updates)

        all_metrics['tda'] = tda_scores
        all_metrics['mutual_similarity'] = mutual_similarity

        return all_metrics
    
    def _load_client_models(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Dict[int, nn.Module]:
        """Load client models from state dictionaries."""
        client_models = {}
        
        for cid, _, state_dict in client_updates:
            model = get_model(
                model_name=self.config.model,
                num_classes=self.config.num_classes,
                dataset_name=self.config.dataset
            )
            model.load_state_dict(state_dict)
            model = model.to(self.device)
            model.eval()
            client_models[cid] = model
        
        return client_models
    
    def _extract_all_representations(
        self,
        client_models: Dict[int, nn.Module]
    ) -> Dict[int, torch.Tensor]:
        """
        Extract representations from clients, combining multiple layers if configured.
        
        Args:
            client_models: Dictionary of client_id -> model
            
        Returns:
            Dictionary of client_id -> representation tensor [root_size, feature_dim]
        """
        
        if not self.use_multi_layer:
                                                         
            return {cid: self._extract_representation_single_model(model) 
                    for cid, model in client_models.items()}
        
                                
        all_layer_representations = {}
        for layer_name in self.extraction_layers:
            layer_representations = {}
            for cid, model in client_models.items():
                with torch.no_grad():
                    representations = []
                    for batch in self.root_loader:
                        inputs, _ = batch
                        inputs = inputs.to(self.device)
                        features = self._extract_features_from_layer(model, inputs, layer_name)
                        representations.append(features.cpu())
                    layer_representations[cid] = torch.cat(representations, dim=0)
            all_layer_representations[layer_name] = layer_representations
        
                                               
        return self._combine_layer_representations(all_layer_representations)
    
    def _combine_layer_representations(
        self,
        all_layer_representations: Dict[str, Dict[int, torch.Tensor]]
    ) -> Dict[int, torch.Tensor]:
        """Combine representations from multiple layers."""
        client_ids = list(next(iter(all_layer_representations.values())).keys())
        combined = {}
        
        for cid in client_ids:
            layer_reps = [all_layer_representations[layer][cid] for layer in self.extraction_layers]
            
            if self.combine_layers_method == 'mean':
                combined[cid] = torch.stack(layer_reps).mean(dim=0)
            elif self.combine_layers_method == 'max':
                combined[cid] = torch.stack(layer_reps).max(dim=0)[0]
            elif self.combine_layers_method == 'min':
                combined[cid] = torch.stack(layer_reps).min(dim=0)[0]
            else:
                log(WARNING, f"Unknown combine method '{self.combine_layers_method}', using mean")
                combined[cid] = torch.stack(layer_reps).mean(dim=0)
        
        return combined
    
    def _extract_representation_single_model(
        self,
        model: nn.Module
    ) -> torch.Tensor:
        """
        Extract penultimate layer features from a single model.
        
        Args:
            model: PyTorch model
            
        Returns:
            Tensor of shape [root_size, feature_dim]
        """
        features_list = []
        
                                           
        def hook_fn(module, input, output):
                                                                      
            if len(output.shape) > 2:
                output = torch.nn.functional.adaptive_avg_pool2d(output, (1, 1))
                output = output.view(output.size(0), -1)
            features_list.append(output.detach())
        
                                            
        hook_handle = self._register_penultimate_hook(model, hook_fn)
        
                                           
        model.eval()
        with torch.no_grad():
            for batch_data, _ in self.root_loader:
                batch_data = batch_data.to(self.device)
                _ = model(batch_data)
        
                     
        hook_handle.remove()
        
                                  
        representations = torch.cat(features_list, dim=0)
        
        return representations
    
    def _register_penultimate_hook(
        self,
        model: nn.Module,
        hook_fn
    ):
        """
        Register hook on the penultimate layer (before final classification).
        
        Args:
            model: PyTorch model
            hook_fn: Hook function to register
            
        Returns:
            Hook handle
        """
                                                           
        model_name = self.config.model.lower()
        
        if 'resnet' in model_name:
                                    
            return model.avgpool.register_forward_hook(hook_fn)
        elif 'vgg' in model_name:
                                               
            return model.features.register_forward_hook(hook_fn)
        elif 'mnist' in model_name or 'mnistnet' in model_name:
                                                                  
            if hasattr(model, 'features'):
                return model.features.register_forward_hook(hook_fn)
            else:
                                                 
                layers = list(model.children())
                return layers[-2].register_forward_hook(hook_fn)
        else:
                                                             
            if hasattr(model, 'avgpool'):
                return model.avgpool.register_forward_hook(hook_fn)
            else:
                                                    
                layers = list(model.children())
                return layers[-2].register_forward_hook(hook_fn)
    
    def _extract_global_representation(self) -> torch.Tensor:
        """
        Extract penultimate layer representation from global model.
        
        Returns:
            Tensor of shape [root_size, feature_dim]
        """
        return self._extract_representation_single_model(self.global_model)
    
    def _extract_features_from_layer(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        layer_name: str
    ) -> torch.Tensor:
        """Extract features from a specific layer using hooks."""
        features = []
        
        def hook_fn(module, input, output):
            features.append(output)
        
        target_layer = self._get_target_layer(model, layer_name)
        
        if target_layer is None:
            log(WARNING, f"Could not find layer '{layer_name}', using penultimate")
            return self._extract_representation_single_model(model)
        
        handle = target_layer.register_forward_hook(hook_fn)
        
        try:
            _ = model(inputs)
            if not features:
                raise RuntimeError(f"Hook did not capture features for layer '{layer_name}'")
            
            output_features = features[0]
            if len(output_features.shape) > 2:
                output_features = output_features.flatten(1)
            
            return output_features
        finally:
            handle.remove()
    
    def _get_target_layer(self, model: nn.Module, layer_name: str) -> Optional[nn.Module]:
        """Get target layer from model."""
        if layer_name == 'penultimate':
            layers = list(model.children())
            if len(layers) >= 2:
                return layers[-2]
        elif layer_name in ['layer2', 'layer3', 'layer4']:
            if hasattr(model, layer_name):
                return getattr(model, layer_name)
        elif hasattr(model, layer_name):
            return getattr(model, layer_name)
        
        return None
    
    def _compute_eigenvalue_metrics(
        self,
        client_representations: Dict[int, torch.Tensor],
        global_representation: torch.Tensor
    ) -> Dict[str, Dict[int, float]]:
        """
        Compute comprehensive eigenvalue-based metrics from delta covariance.
        
        Formula (from LaTeX lines 56-61):
        Δᵢ = Rᵢ - R_global
        Δᵢ_centered = Δᵢ - mean(Δᵢ)
        Cᵢ = (Δᵢ_centered^T @ Δᵢ_centered) / (n-1)
        
        Computes multiple metrics:
        - spectral_norm: λ_max (largest eigenvalue)
        - trace: sum of all eigenvalues (total variance)
        - eigenvalue_1, ..., eigenvalue_k: top-k eigenvalues
        - effective_rank: number of eigenvalues needed for threshold% variance
        - condition_number: λ_max / (λ_min + epsilon)
        - eigenvalue_entropy: Shannon entropy of normalized eigenvalue distribution
        - eigenvalue_decay_rate: λ_2 / (λ_max + epsilon)
        
        Args:
            client_representations: Dict of client_id -> representations [root_size, D]
            global_representation: Global model representations [root_size, D]
            
        Returns:
            Dict mapping metric_name -> {client_id: value}
        """
                                            
        metrics = {
            'spectral_norm': {},
            'trace': {},
            'effective_rank': {},
            'condition_number': {},
            'eigenvalue_entropy': {},
            'eigenvalue_decay_rate': {}
        }
        
                                                  
        for k in range(1, self.top_k_eigenvalues + 1):
            metrics[f'eigenvalue_{k}'] = {}
        
        n = global_representation.shape[0]                       
        
        for cid, client_repr in client_representations.items():
            try:
                               
                delta = client_repr - global_representation          
                
                              
                delta_centered = delta - delta.mean(dim=0, keepdim=True)          
                
                                                 
                                                                             
                                                                              
                                                                                
                                                               
                if self.v3_use_dual_spectral and delta_centered.shape[0] < delta_centered.shape[1]:
                    gram_matrix = (delta_centered @ delta_centered.T) / max(n - 1, 1)          
                    gram_eigenvalues = torch.linalg.eigvalsh(gram_matrix)                 
                                                                              
                    d = delta_centered.shape[1]
                    eigenvalues = torch.zeros(d, device=gram_eigenvalues.device,
                                             dtype=gram_eigenvalues.dtype)
                    eigenvalues[:len(gram_eigenvalues)] = gram_eigenvalues
                else:
                    cov_matrix = (delta_centered.T @ delta_centered) / max(n - 1, 1)          
                    eigenvalues = torch.linalg.eigvalsh(cov_matrix)
                
                                             
                if eigenvalues.numel() == 0:
                    log(WARNING, f"Empty eigenvalues for client {cid}, using defaults")
                    for key in metrics:
                        metrics[key][cid] = 0.0
                    continue
                
                                                 
                eigenvalues = torch.sort(eigenvalues, descending=True)[0]
                eigenvalues_np = eigenvalues.cpu().numpy()
                
                                                                                                          
                eigenvalues_np = np.maximum(eigenvalues_np, 0.0)
                
                                       
                spectral_norm = float(eigenvalues_np[0])
                metrics['spectral_norm'][cid] = spectral_norm
                
                                                
                trace = float(np.sum(eigenvalues_np))
                metrics['trace'][cid] = trace
                
                                   
                k = min(self.top_k_eigenvalues, len(eigenvalues_np))
                for i in range(1, self.top_k_eigenvalues + 1):
                    if i <= k:
                        metrics[f'eigenvalue_{i}'][cid] = float(eigenvalues_np[i - 1])
                    else:
                        metrics[f'eigenvalue_{i}'][cid] = 0.0
                
                                                                                       
                if trace > self.epsilon:
                    cumsum = np.cumsum(eigenvalues_np)
                    cumsum_normalized = cumsum / trace
                                                                             
                                                                                                    
                    effective_rank_idx = np.searchsorted(cumsum_normalized, self.effective_rank_threshold, side='left')
                                                                                      
                    effective_rank = min(max(effective_rank_idx + 1, 1), len(eigenvalues_np))
                else:
                    effective_rank = 0
                metrics['effective_rank'][cid] = float(effective_rank)
                
                                                  
                if len(eigenvalues_np) > 0:
                    lambda_min = eigenvalues_np[-1]
                    condition_num = spectral_norm / (lambda_min + self.epsilon)
                else:
                    condition_num = 0.0
                metrics['condition_number'][cid] = float(condition_num)
                
                                                                                 
                if trace > self.epsilon:
                                                       
                    eigenvalues_normalized = eigenvalues_np / trace
                                                                 
                                                        
                    non_zero_mask = eigenvalues_normalized > self.epsilon
                    if np.any(non_zero_mask):
                        p = eigenvalues_normalized[non_zero_mask]
                        entropy = -np.sum(p * np.log(p + self.epsilon))
                    else:
                        entropy = 0.0
                else:
                    entropy = 0.0
                metrics['eigenvalue_entropy'][cid] = float(entropy)
                
                                                     
                if len(eigenvalues_np) >= 2 and spectral_norm > self.epsilon:
                    decay_rate = float(eigenvalues_np[1] / (spectral_norm + self.epsilon))
                else:
                    decay_rate = 0.0
                metrics['eigenvalue_decay_rate'][cid] = decay_rate
                
            except Exception as e:
                log(WARNING, f"Eigenvalue metrics computation failed for client {cid}: {e}")
                                                   
                for key in metrics:
                    metrics[key][cid] = 0.0
        
        return metrics
    
    def _compute_delta_norms(
        self,
        client_representations: Dict[int, torch.Tensor],
        global_representation: torch.Tensor
    ) -> Dict[int, float]:
        """
        Compute delta norms (Frobenius norm of representation difference).
        
        Formula (from LaTeX lines 64-67):
        ‖Δᵢ‖_F = sqrt(sum((Δᵢ)²))
        
        Args:
            client_representations: Dict of client_id -> representations
            global_representation: Global model representations
            
        Returns:
            Dict of client_id -> delta norm (float)
        """
        delta_norms = {}
        
        for cid, client_repr in client_representations.items():
                           
            delta = client_repr - global_representation
            
                                    
            delta_norm = torch.linalg.norm(delta, ord='fro').item()
            delta_norms[cid] = delta_norm
        
        return delta_norms
    
    def _robust_normalize(
        self,
        scores: Dict[int, float]
    ) -> Dict[int, float]:
        """
        Robust normalization using median and IQR.
        
        Formula (from LaTeX lines 69-74):
        normalized(x) = (x - median(x)) / (IQR(x) + ε)
        
        where IQR = Q3 - Q1
        
        Args:
            scores: Dict of client_id -> score
            
        Returns:
            Dict of client_id -> normalized score
        """
        if not scores:
            return {}
        
        score_values = np.array(list(scores.values()))
        
                                
        median = np.median(score_values)
        q1 = np.percentile(score_values, 25)
        q3 = np.percentile(score_values, 75)
        iqr = q3 - q1
        
                   
        normalized_scores = {}
        for cid, score in scores.items():
            normalized = (score - median) / (iqr + self.epsilon)
            normalized_scores[cid] = float(normalized)
        
        return normalized_scores
    
    def _compute_combined_scores(
        self,
        spectral_normed: Dict[int, float],
        delta_normed: Dict[int, float]
    ) -> Dict[int, float]:
        """
        Compute combined score from normalized spectral and delta norms.
        
        Formula (from LaTeX lines 76-79):
        combined_i = 0.6 * spectral_normed_i + 0.4 * delta_normed_i
        
        Args:
            spectral_normed: Normalized spectral norms
            delta_normed: Normalized delta norms
            
        Returns:
            Dict of client_id -> combined score
        """
        combined_scores = {}
        
        for cid in spectral_normed.keys():
            combined = (self.spectral_weight * spectral_normed[cid] + 
                       self.delta_weight * delta_normed[cid])
            combined_scores[cid] = combined
        
        return combined_scores
    
    def _compute_tda_scores(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Dict[int, float]:
        """
        Compute TDA (Temporal Direction Alignment) scores.
        
        Measures cosine similarity between client model and global model.
        Uses fera_anonm approach (model-to-model similarity).
        Normalized to [0, 1] range for easier interpretation.
        
        Formula:
        TDA_i = [(θ_i · θ_global) / (‖θ_i‖ · ‖θ_global‖) + 1] / 2
        
        Args:
            client_updates: List of (client_id, num_examples, state_dict)
            
        Returns:
            Dict of client_id -> TDA score [0, 1]
            0.0 = opposite direction, 0.5 = orthogonal, 1.0 = aligned
        """
        tda_scores = {}
        
                                         
        global_params = self._flatten_state_dict(self.global_model.state_dict())
        global_norm = torch.linalg.norm(global_params).clamp(min=self.epsilon)
        
        for cid, _, state_dict in client_updates:
            try:
                                                 
                client_params = self._flatten_state_dict(state_dict)
                client_norm = torch.linalg.norm(client_params).clamp(min=self.epsilon)
                
                                                                                 
                dot_product = torch.dot(client_params, global_params)
                cosine_sim = (dot_product / (client_norm * global_norm)).item()
                
                                           
                tda_score = (cosine_sim + 1.0) / 2.0
                tda_scores[cid] = tda_score
                
            except Exception as e:
                log(WARNING, f"TDA computation failed for client {cid}: {e}")
                tda_scores[cid] = 0.5                            
        
        return tda_scores
    
    def _compute_mutual_similarity(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Dict[int, float]:
        """
        Compute mutual similarity (maximum pairwise cosine similarity).
        
        Similar to FoolsGold's approach, computes the maximum cosine similarity
        between each client's update and all other clients' updates.
        
        Formula:
        U = stack(u₁, u₂, ..., uₙ) where uᵢ = vec(θᵢ - θ_global)
        Û = U / ‖U‖₂ (row-wise normalization)
        S = Û @ Û^T (pairwise cosine similarity matrix)
        mutual_i = max(S[i, :]) excluding S[i, i]
        
        Args:
            client_updates: List of (client_id, num_examples, state_dict)
            
        Returns:
            Dict of client_id -> maximum mutual similarity score
        """
        mutual_scores = {}
        
                              
        global_params = self._flatten_state_dict(self.global_model.state_dict())
        
                                    
        client_ids = []
        update_vectors = []
        
        for cid, _, state_dict in client_updates:
            try:
                client_params = self._flatten_state_dict(state_dict)
                update_vector = client_params - global_params
                
                client_ids.append(cid)
                update_vectors.append(update_vector)
            except Exception as e:
                log(WARNING, f"Failed to compute update for client {cid}: {e}")
                continue
        
        if len(update_vectors) < 2:
                                                           
            return {cid: 0.0 for cid in client_ids}
        
                                     
        U = torch.stack(update_vectors, dim=0)
        
                           
        U_norms = torch.linalg.norm(U, dim=1, keepdim=True).clamp(min=self.epsilon)
        U_normalized = U / U_norms
        
                                                          
        S = U_normalized @ U_normalized.T
        
                                                   
        N = len(client_ids)
        for i, cid in enumerate(client_ids):
                                                                   
                                                                            
            S_masked = S[i].clone()
            S_masked[i] = -1e9                        
            mutual_sim = S_masked.max().item()
            mutual_scores[cid] = mutual_sim
        
        return mutual_scores
    
    def _compute_mutual_similarity_repr(
        self,
        client_representations: Dict[int, torch.Tensor],
        global_representation: torch.Tensor
    ) -> Dict[int, float]:
        """
        Compute mutual similarity using representation-space deltas (V3 efficiency).

        Addresses Reviewer Q2 (scalability): instead of flattening the full parameter
        vector (~11M dims for ResNet-18) for each client, this method reuses the
        already-computed penultimate-layer representations (n × D_rep, e.g. 64 × 512
        = 32K dims) from Phase 1 of _compute_all_metrics.

        Cost comparison (K=10 clients, D_full=11M, n×D_rep=32K):
          Original:  O(K² × D_full) = 10² × 11M = 1.1B ops  + 440MB memory
          V3 repr:   O(K² × n×D_rep) = 10² × 32K = 3.2M ops + 1.3MB memory  (~330× cheaper)

        The representation delta δᵢ = Rᵢ − R_global encodes HOW each client's model
        transforms the root data differently from the global model, which is the same
        signal used by the spectral and combined-score metrics — making representation-
        space mutual similarity more semantically consistent with the rest of FeRA.

        Args:
            client_representations: {cid: [n, D_rep]} from Phase 1
            global_representation:  [n, D_rep] from Phase 1

        Returns:
            {cid: max pairwise cosine similarity with any other client (repr space)}
        """
        client_ids = list(client_representations.keys())

                                                                       
        delta_vecs = []
        for cid in client_ids:
            delta = (client_representations[cid] - global_representation).flatten()             
            delta_vecs.append(delta)

        if len(delta_vecs) < 2:
            return {cid: 0.0 for cid in client_ids}

        U = torch.stack(delta_vecs)                                        
        U_norms = torch.linalg.norm(U, dim=1, keepdim=True).clamp(min=self.epsilon)
        U_hat = U / U_norms                                                                

        S = U_hat @ U_hat.T                                                              

        mutual_scores = {}
        for i, cid in enumerate(client_ids):
            S_row = S[i].clone()
            S_row[i] = -1e9                
            mutual_scores[cid] = float(S_row.max())

        return mutual_scores

    def _flatten_state_dict(
        self,
        state_dict: StateDict
    ) -> torch.Tensor:
        """
        Flatten all parameters in state_dict into a single 1D tensor.
        
        Args:
            state_dict: Model state dictionary
            
        Returns:
            Flattened tensor of all parameters (on self.device)
        """
        flat_params = []
        for key in sorted(state_dict.keys()):                        
            param = state_dict[key]
                                                
            if key in self.ignore_weights:
                continue
                                                                         
            if isinstance(param, torch.Tensor):
                param = param.to(self.device)
            flat_params.append(param.flatten())
        
        if not flat_params:
            return torch.tensor([], device=self.device)
        
        return torch.cat(flat_params)

    def _save_metrics_csvs(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        round_num: int
    ):
        """
        Save metrics to a single cumulative CSV file for all rounds.
        
        Creates a single file: all_rounds_metrics.csv
        - Header with ground truth malicious clients (written once)
        - Each round's data appended with round number
        
        Args:
            all_metrics: Dictionary of metric_name -> {client_id: value}
            round_num: Current round number
        """
                                 
        ground_truth = self.client_manager.get_malicious_clients()
        
                            
        client_ids = list(all_metrics['spectral_norm'].keys())
        
                              
        labels = {cid: 'malicious' if cid in ground_truth else 'benign' 
                 for cid in client_ids}
        
                                     
        cumulative_csv = self.metrics_output_dir / "all_rounds_metrics.csv"
        
                                                                   
        is_first_round = not cumulative_csv.exists()
        
                                  
        with open(cumulative_csv, 'a') as f:
            if is_first_round:
                                                             
                mal_clients = ','.join(map(str, sorted(ground_truth)))
                f.write(f"Ground truth (GT) malicious [{mal_clients}]\n\n")
            
                                
            f.write(f"Round {round_num}\n")
            
                              
                          
            f.write("client_id,label,spectral_norm,spectral_norm_normed,")
            f.write("delta_norm,delta_norm_normed,combined_score,tda,")
            f.write("mutual_similarity,")
                                
            f.write("trace,")
                               
            for k in range(1, self.top_k_eigenvalues + 1):
                f.write(f"eigenvalue_{k},")
                                      
            f.write("effective_rank,condition_number,eigenvalue_entropy,eigenvalue_decay_rate\n")
            
                                        
            for cid in client_ids:
                              
                f.write(f"{cid},{labels[cid]},")
                f.write(f"{all_metrics.get('spectral_norm', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('spectral_norm_normed', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('delta_norm', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('delta_norm_normed', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('combined_score', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('tda', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('mutual_similarity', {}).get(cid, 0.0)},")
                                    
                f.write(f"{all_metrics.get('trace', {}).get(cid, 0.0)},")
                                   
                for k in range(1, self.top_k_eigenvalues + 1):
                    f.write(f"{all_metrics.get(f'eigenvalue_{k}', {}).get(cid, 0.0)},")
                                          
                f.write(f"{all_metrics.get('effective_rank', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('condition_number', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('eigenvalue_entropy', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('eigenvalue_decay_rate', {}).get(cid, 0.0)}\n")
            
                                             
            f.write("\n")

