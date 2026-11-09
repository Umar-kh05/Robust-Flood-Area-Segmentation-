import os
import json
import time
from pathlib import Path

import torch
from src.experiment_engine import FloodUNet, FloodTileDataset, set_seed, device
from src.coordinator import ExperimentCoordinator
from src.visualizer import (
    plot_convergence_curves,
    plot_per_client_performance,
    plot_unequal_work_comparison,
    plot_qualitative_predictions,
    plot_error_breakdown
)

def main():
    print("====================================================================")
    print("CROSS-DATASET FEDERATED LEARNING FOR FLOODWATER SEGMENTATION")
    print(f"Device: {device}")
    print("====================================================================")

    data_root = 'data/federated_ready'
    coordinator = ExperimentCoordinator(data_root=data_root, batch_size=4, lr=0.01)

    all_results = {}
    models_dict = {}

    # 1. Main Federated Experiments (Equal Computation)
    # Algorithms: FedAvg, FedProx, FedNova, SCAFFOLD
    algorithms = ['fedavg', 'fedprox', 'fednova', 'scaffold']
    
    for algo in algorithms:
        print(f"\n>>> Running Main FL Experiment: {algo.upper()} <<<", flush=True)
        res = coordinator.run_federated(
            algorithm=algo,
            num_rounds=3,
            local_steps=2,
            seed=42,
            patience=5
        )
        all_results[algo] = res
        
        # Load best model for visualization
        model = FloodUNet().to(device)
        ckpt_path = Path("checkpoints") / algo / f"{algo}_seed42_best.pt"
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        models_dict[algo] = model

    # 2. Centralized Baseline (Pooled Data)
    print("\n>>> Running Baseline: CENTRALIZED POOLED <<<", flush=True)
    cent_res = coordinator.run_centralized(max_steps=6, check_interval=2, seed=42, patience=5)
    all_results['centralized'] = cent_res
    
    cent_model = FloodUNet().to(device)
    cent_model.load_state_dict(torch.load(Path("checkpoints") / "centralized_seed42_best.pt", map_location=device))
    models_dict['centralized'] = cent_model

    # 3. Local-Only Baselines (One per client)
    for c_name in ['etci', 'mmflood', 'kurosiwo']:
        print(f"\n>>> Running Baseline: LOCAL-ONLY [{c_name.upper()}] <<<", flush=True)
        loc_res = coordinator.run_local_only(client_name=c_name, max_steps=4, check_interval=2, seed=42, patience=5)
        all_results[f"local_{c_name}"] = loc_res

    # 4. Unequal Local Computation Experiment (Stress Test)
    # Step allocation: ETCI=1, MMFlood=2, Kuro Siwo=3
    print("\n====================================================================", flush=True)
    print("UNEQUAL COMPUTATION EXPERIMENT (Stress Test for FedNova)", flush=True)
    print("====================================================================", flush=True)
    unequal_results = {}
    unequal_steps = [1, 2, 3]
    
    for algo in algorithms:
        print(f"\n>>> Running Unequal FL Experiment: {algo.upper()} (Steps={unequal_steps}) <<<", flush=True)
        res_u = coordinator.run_federated(
            algorithm=algo,
            num_rounds=2,
            local_steps=unequal_steps,
            seed=42,
            patience=5
        )
        unequal_results[algo] = res_u

    all_results['unequal_experiment'] = unequal_results

    # Save all results to JSON
    results_path = Path("results") / "experimental_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Pipeline] Successfully saved experimental results to {results_path}")

    # 5. Generate Figures
    print("\n>>> Generating High-Resolution Figures for Report & Notebook <<<")
    datasets_dict = {
        'etci': FloodTileDataset('etci', 'test', data_root=data_root),
        'mmflood': FloodTileDataset('mmflood', 'test', data_root=data_root),
        'kurosiwo': FloodTileDataset('kurosiwo', 'test', data_root=data_root)
    }

    plot_convergence_curves(all_results, "figures/fig1_val_convergence.png")
    plot_per_client_performance(all_results, "figures/fig2_per_client_performance.png")
    plot_unequal_work_comparison(all_results, unequal_results, "figures/fig3_unequal_computation.png")
    plot_qualitative_predictions(models_dict, datasets_dict, device, "figures/fig4_qualitative_predictions.png")
    plot_error_breakdown(models_dict, datasets_dict, device, "figures/fig5_error_decomposition.png")

    print("\n====================================================================")
    print("ALL EXPERIMENTS & VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("====================================================================")

if __name__ == '__main__':
    main()
