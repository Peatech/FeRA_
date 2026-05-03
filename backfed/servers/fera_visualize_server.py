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
                               
        consistency_filter: dict = None,
        collusion_filter: dict = None,
        outlier_filter: dict = None,
        scaled_norm_filter: dict = None,
                                  
        filter_variant: str = "v1",                                        
                                     
                                                                                     
                                                                                 
                                                                                  
                                                                                   
                                                                                       
        log_ranked_metric_tables: bool = False,
        flagged_client_treatment: str = "project",
                                          
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
       
        super().__init__(server_config, server_type, eta, **kwargs)
        self.verbose_detection_logging = False
        self.log_ranked_metric_tables = log_ranked_metric_tables

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
        
                                                   
        self.consistency_filter = consistency_filter or {
            'enabled': True,
            'combined_threshold': 0.50,
            'das_threshold': 0.50,
            'mutual_sim_threshold': 0.60
        }
        self.collusion_filter = collusion_filter or {
            'enabled': False,
            'mutual_sim_top_percent': 0.40,
            'das_top_percent': 0.40
        }
        self.outlier_filter = outlier_filter or {
            'enabled': False
        }
        self.scaled_norm_filter = scaled_norm_filter or {
            'enabled': True,
            'k_mad': 150.0
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
        
                           
        if len(client_updates) < 2:
            log(WARNING, "FeRA Visualize: Less than 2 clients, cannot perform detection")
            return [], [cid for cid, _, _ in client_updates]
        
        try:
                                                               
            all_metrics = self._compute_all_metrics(client_updates)
            
                                               
            malicious_sets = []
            
                                         
            if self.consistency_filter['enabled']:
                mal_ids = self._filter_consistency(all_metrics, client_updates)
                malicious_sets.append(('Consistency', mal_ids))
            
                                           
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
        
        sorted_clients = sorted(metric_dict.items(), key=lambda x: x[1])
        return {cid: rank for rank, (cid, _) in enumerate(sorted_clients, start=1)}
    
    def _filter_consistency(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        
        combined_scores = all_metrics['combined_score']
        das_scores = all_metrics['das']
        mutual_sim_scores = all_metrics['mutual_similarity']
        
                                                               
        combined_ranks = self._rank_clients(combined_scores)
        das_ranks = self._rank_clients(das_scores)
        mutual_sim_ranks = self._rank_clients(mutual_sim_scores)
        
                              
        n = len(client_updates)
        combined_thresh = int(n * self.consistency_filter['combined_threshold'])
        das_thresh = int(n * self.consistency_filter['das_threshold'])
        mutual_sim_thresh = int(n * self.consistency_filter['mutual_sim_threshold']) + 1
        
                                                   
        malicious = set()
        for cid, _, _ in client_updates:
            if (combined_ranks[cid] <= combined_thresh and
                das_ranks[cid] <= das_thresh and
                mutual_sim_ranks[cid] >= mutual_sim_thresh):
                malicious.add(cid)
        
        return malicious
    
    def _filter_collusion(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        
        das_scores = all_metrics['das']
        mutual_sim_scores = all_metrics['mutual_similarity']
        
                                                         
        das_ranks = self._rank_clients(das_scores)
        mutual_sim_ranks = self._rank_clients(mutual_sim_scores)
        
                                          
                                                                      
        n = len(client_updates)
        das_top_thresh = int(n * (1.0 - self.collusion_filter['das_top_percent']))
        mutual_sim_top_thresh = int(n * (1.0 - self.collusion_filter['mutual_sim_top_percent']))
        
                                                    
        malicious = set()
        for cid, _, _ in client_updates:
            if (das_ranks[cid] > das_top_thresh and
                mutual_sim_ranks[cid] > mutual_sim_top_thresh):
                malicious.add(cid)
        
        return malicious
    
    def _filter_outliers(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        
        das_scores = all_metrics['das']
        mutual_sim_scores = all_metrics['mutual_similarity']
        
                              
        das_sorted = sorted(das_scores.items(), key=lambda x: x[1])
        mutual_sim_sorted = sorted(mutual_sim_scores.items(), key=lambda x: x[1])
        
                                                
        das_lowest = das_sorted[0][0]
        das_highest = das_sorted[-1][0]
        mutual_sim_lowest = mutual_sim_sorted[0][0]
        mutual_sim_highest = mutual_sim_sorted[-1][0]
        
                                 
        das_extremes = {das_lowest, das_highest}
        mutual_sim_extremes = {mutual_sim_lowest, mutual_sim_highest}
        
                                           
        malicious = das_extremes & mutual_sim_extremes
        
        return malicious
    
    def _filter_scaled_norm(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        
        spectral_norms = all_metrics['spectral_norm']

        median_spectral = np.median(list(spectral_norms.values()))
        if median_spectral < self.epsilon:
            return set()

        # Spectral ratio r_i = spectral_norm_i / median (Eq. spectral-ratio in paper)
        ratios = {cid: sn / median_spectral for cid, sn in spectral_norms.items()}
        ratio_vals = np.array(list(ratios.values()))
        median_ratio = np.median(ratio_vals)
        mad = np.median(np.abs(ratio_vals - median_ratio))

        k = self.scaled_norm_filter.get('k_mad', 150.0)
        threshold = median_ratio + k * mad
        return {cid for cid, r in ratios.items() if r > threshold}
    
    def _filter_consistency_v2(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        
        combined_scores = all_metrics['combined_score']
        das_scores = all_metrics['das']
        
                                                         
        combined_ranks = self._rank_clients(combined_scores)
        das_ranks = self._rank_clients(das_scores)
        
                              
        n = len(client_updates)
        thresh_50 = int(n * 0.50)
        
                                                    
        malicious = set()
        for cid, _, _ in client_updates:
            if (combined_ranks[cid] <= thresh_50 and 
                das_ranks[cid] <= thresh_50):
                malicious.add(cid)
        
        return malicious
    
    def _filter_collusion_v2(
        self,
        all_metrics: Dict[str, Dict[int, float]],
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        
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
        
        spectral_norms = all_metrics['spectral_norm']

        median_spectral = np.median(list(spectral_norms.values()))
        if median_spectral < self.epsilon:
            return set()

        ratios = {cid: sn / median_spectral for cid, sn in spectral_norms.items()}
        ratio_vals = np.array(list(ratios.values()))
        median_ratio = np.median(ratio_vals)
        mad = np.median(np.abs(ratio_vals - median_ratio))

        k = self.scaled_norm_filter.get('k_mad', 150.0)
        threshold = median_ratio + k * mad
        return {cid for cid, r in ratios.items() if r > threshold}
    
    def _clip_malicious_updates(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]],
        malicious_clients: List[int]
    ) -> List[Tuple[client_id, num_examples, StateDict]]:
        
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
        if not self.log_ranked_metric_tables:
            return
        filter_bits = ",".join(f"{fn}:{sorted(m)}" for fn, m in malicious_sets)
        log(INFO, f"[Round {self.current_round}] FeRA filters=[{filter_bits}] flagged={sorted(malicious_clients)} benign={sorted(benign_clients)}")

    def aggregate_client_updates(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ):
       
        if not client_updates:
            log(WARNING, "No client updates found, using global model")
            return False

        if self.log_ranked_metric_tables:
            log(INFO, "")
            log(INFO, "═══════════════════════════════════════════════")
            log(INFO, f"  FeRA Visualize Metrics - Round {self.current_round}")
            log(INFO, "═══════════════════════════════════════════════")

        try:
            all_metrics = self._compute_all_metrics(client_updates)
            self._save_metrics_csvs(all_metrics, self.current_round)
            if self.log_ranked_metric_tables:
                log(INFO, "")
                log(INFO, f"✓ Metrics saved to: {self.metrics_output_dir}")
                log(INFO, "═══════════════════════════════════════════════")
                log(INFO, "")
            else:
                log(INFO, f"FeRA metrics round {self.current_round}: written under {self.metrics_output_dir}")
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
        
        combined_ranks = self._rank_clients(all_metrics['combined_score'])
        das_ranks = self._rank_clients(all_metrics['das'])
        mutual_sim_ranks = self._rank_clients(all_metrics['mutual_similarity'])

        n = len(client_updates)
                                                                 
                                                                                    
                                                                                      
        combined_thresh = int(n * 0.50)               
        das_thresh = int(n * 0.50)                    
        mutual_thresh = int(n * 0.70) + 1           

        malicious = set()
        for cid, _, _ in client_updates:
            if (combined_ranks[cid] <= combined_thresh and
                    das_ranks[cid] <= das_thresh and
                    mutual_sim_ranks[cid] >= mutual_thresh):
                malicious.add(cid)
        return malicious

    def _filter_residual_collusion_v3(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> set:
        
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
        
        if not client_updates:
            log(WARNING, "V3 BSP: No client updates, skipping.")
            return False

                                                                                         
                                                                                 
        return self._aggregate_v1_with_bsp(client_updates)

                                                                            
                                                                        
                                                                            

    def _compute_spectral_norm_dual(
        self,
        delta_centered: torch.Tensor
    ) -> float:
        
        n = delta_centered.shape[0]
        gram = (delta_centered @ delta_centered.T) / max(n - 1, 1)          
                                                                              
        eigenvalues = torch.linalg.eigvalsh(gram)
        return float(eigenvalues[-1].clamp(min=0.0))

    def _compute_all_metrics(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Dict[str, Dict[int, float]]:
       
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
        
                                                                   
        das_scores = self._compute_das_scores(client_updates)

                                                                                          
         
                                                                            
                                                                                  
                                                                              
                                                                                    
                                                                                         
                                                               
                                                                                       
                                                                                       
                                                 
         
                                                                                
                                                                                  
                                                                                       
        mutual_similarity = self._compute_mutual_similarity(client_updates)

        all_metrics['das'] = das_scores
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
        
        return self._extract_representation_single_model(self.global_model)
    
    def _extract_features_from_layer(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        layer_name: str
    ) -> torch.Tensor:
        
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
        
        combined_scores = {}
        
        for cid in spectral_normed.keys():
            combined = (self.spectral_weight * spectral_normed[cid] + 
                       self.delta_weight * delta_normed[cid])
            combined_scores[cid] = combined
        
        return combined_scores
    
    def _compute_das_scores(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Dict[int, float]:
        
        das_scores = {}
        
                                         
        global_params = self._flatten_state_dict(self.global_model.state_dict())
        global_norm = torch.linalg.norm(global_params).clamp(min=self.epsilon)
        
        for cid, _, state_dict in client_updates:
            try:
                                                 
                client_params = self._flatten_state_dict(state_dict)
                client_norm = torch.linalg.norm(client_params).clamp(min=self.epsilon)
                
                                                                                 
                dot_product = torch.dot(client_params, global_params)
                cosine_sim = (dot_product / (client_norm * global_norm)).item()
                
                                           
                # DAS (Directional Alignment Score): cosine similarity between client
                # and global update delta, scaled to [0,1].  Previously referred to
                # as TDA (Targeted Directional Attention) in earlier drafts.
                das_score = (cosine_sim + 1.0) / 2.0
                das_scores[cid] = das_score
                
            except Exception as e:
                log(WARNING, f"DAS computation failed for client {cid}: {e}")
                das_scores[cid] = 0.5                            
        
        return das_scores
    
    def _compute_mutual_similarity(
        self,
        client_updates: List[Tuple[client_id, num_examples, StateDict]]
    ) -> Dict[int, float]:
        
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
            f.write("delta_norm,delta_norm_normed,combined_score,das,")
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
                f.write(f"{all_metrics.get('das', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('mutual_similarity', {}).get(cid, 0.0)},")
                                    
                f.write(f"{all_metrics.get('trace', {}).get(cid, 0.0)},")
                                   
                for k in range(1, self.top_k_eigenvalues + 1):
                    f.write(f"{all_metrics.get(f'eigenvalue_{k}', {}).get(cid, 0.0)},")
                                          
                f.write(f"{all_metrics.get('effective_rank', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('condition_number', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('eigenvalue_entropy', {}).get(cid, 0.0)},")
                f.write(f"{all_metrics.get('eigenvalue_decay_rate', {}).get(cid, 0.0)}\n")
            
                                             
            f.write("\n")

