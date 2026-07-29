#!/usr/bin/env python3
"""
Visualization script for scheduling experiment results.

Generates:
1. Heatmap of throughput (scheduler vs workload) for heterogeneous config
2. Ranking heatmap sorted by average throughput
3. Scatter plot comparing heterogeneous vs homogeneous by price
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from simulator.configs.hardware import hardware_params


# Hardware configuration prices ($/hr)
HW_CONFIG_PRICES = {
    'hetero-5gpu': {
        # RTX3090 + L40S + A6000 + A100_80G:SXM + H100:SXM
        'gpus': ['NVDA:RTX3090', 'NVDA:L40S', 'NVDA:A6000', 'NVDA:A100_80G:SXM', 'NVDA:H100:SXM'],
        'total_price': 0.13 + 0.44 + 0.39 + 0.68 + 1.56  # $3.20/hr
    },
    'homo-6gpu': {
        # 6x A100_80G:SXM
        'gpus': ['NVDA:A100_80G:SXM'] * 6,
        'total_price': 0.68 * 6  # $4.08/hr
    }
}


def load_results(results_dir='experiment_results'):
    """Load all experiment results from JSON files."""
    results_file = os.path.join(results_dir, 'all_results.json')

    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found")
        print(f"Please run the experiment first: python cli/run_scheduling_experiment.py")
        sys.exit(1)

    with open(results_file, 'r') as f:
        results = json.load(f)

    return results


def create_latency_heatmap(results, hw_config='hetero-5gpu', output_file='heatmap_latency.png'):
    """
    Create heatmap of latency: scheduler vs workload for given hw_config.
    Includes 'avg' column showing average across workloads.
    """
    # Filter results for the specified hw_config
    filtered = [r for r in results if r['hw_config'] == hw_config]

    if not filtered:
        print(f"No results found for hw_config: {hw_config}")
        return

    # Get unique schedulers and workloads
    schedulers = sorted(list(set(r['scheduler'] for r in filtered)))
    workloads = sorted(list(set(r['workload'] for r in filtered)))

    # Create data matrix with extra column for average (use NaN for missing data)
    data = np.full((len(schedulers), len(workloads) + 1), np.nan)

    for i, scheduler in enumerate(schedulers):
        latencies = []
        for j, workload in enumerate(workloads):
            matching = [r for r in filtered
                       if r['scheduler'] == scheduler and r['workload'] == workload]
            if matching:
                latency = matching[0]['avg_latency']
                data[i, j] = latency
                latencies.append(latency)

        # Calculate average
        if latencies:
            data[i, -1] = np.mean(latencies)

    # Create custom annotations with "-" for missing data
    annot = np.empty_like(data, dtype=object)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isnan(data[i, j]):
                annot[i, j] = "-"
            else:
                annot[i, j] = f"{data[i, j]:.2f}"

    # Create plot
    plt.figure(figsize=(14, 10))

    # Add 'avg' to workload labels
    workload_labels = workloads + ['avg']

    # Create heatmap with custom annotations
    ax = sns.heatmap(
        data,
        annot=annot,
        fmt='',
        cmap='RdYlGn_r',
        xticklabels=workload_labels,
        yticklabels=schedulers,
        cbar_kws={'label': 'Latency (s)'},
        linewidths=0.5
    )

    plt.title(f'Latency Heatmap - {hw_config.upper()}\n'
              f'Hardware: {HW_CONFIG_PRICES[hw_config]["total_price"]:.2f} $/hr',
              fontsize=14, fontweight='bold')
    plt.xlabel('Workload', fontsize=12)
    plt.ylabel('Scheduler', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_ranking_heatmap(results, hw_config='hetero-5gpu', output_file='heatmap_ranking.png'):
    """
    Create ranking heatmap: schedulers sorted by average latency.
    """
    # Filter results for the specified hw_config
    filtered = [r for r in results if r['hw_config'] == hw_config]

    if not filtered:
        print(f"No results found for hw_config: {hw_config}")
        return

    # Get unique schedulers and workloads
    schedulers = sorted(list(set(r['scheduler'] for r in filtered)))
    workloads = sorted(list(set(r['workload'] for r in filtered)))

    # Calculate average latency for each scheduler
    scheduler_avgs = []
    for scheduler in schedulers:
        latencies = []
        for workload in workloads:
            matching = [r for r in filtered
                       if r['scheduler'] == scheduler and r['workload'] == workload]
            if matching:
                latencies.append(matching[0]['avg_latency'])

        if latencies:
            avg_latency = np.mean(latencies)
            scheduler_avgs.append((scheduler, avg_latency))

    # Sort by average latency (ascending - lower is better)
    scheduler_avgs.sort(key=lambda x: x[1])
    sorted_schedulers = [s[0] for s in scheduler_avgs]

    # Create data matrix with extra column for average
    data = np.zeros((len(sorted_schedulers), len(workloads) + 1))
    ranks = np.zeros((len(sorted_schedulers), len(workloads) + 1))

    for i, scheduler in enumerate(sorted_schedulers):
        latencies = []
        for j, workload in enumerate(workloads):
            matching = [r for r in filtered
                       if r['scheduler'] == scheduler and r['workload'] == workload]
            if matching:
                latency = matching[0]['avg_latency']
                data[i, j] = latency
                latencies.append(latency)

        # Calculate average
        if latencies:
            data[i, -1] = np.mean(latencies)

    # Calculate ranks for each workload (1 = best = lowest latency)
    for j in range(data.shape[1]):
        column = data[:, j]
        temp = column.argsort()  # Ascending order (lower is better)
        ranks[temp, j] = np.arange(len(column)) + 1

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Add 'avg' to workload labels
    workload_labels = workloads + ['avg']

    # Heatmap 1: Latency values
    sns.heatmap(
        data,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        xticklabels=workload_labels,
        yticklabels=sorted_schedulers,
        cbar_kws={'label': 'Latency (s)'},
        linewidths=0.5,
        ax=ax1
    )
    ax1.set_title(f'Latency (Ranked by Avg)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Workload', fontsize=11)
    ax1.set_ylabel('Scheduler (Sorted by Avg)', fontsize=11)
    ax1.tick_params(axis='x', rotation=45)

    # Heatmap 2: Ranks
    sns.heatmap(
        ranks,
        annot=True,
        fmt='.0f',
        cmap='RdYlGn_r',
        xticklabels=workload_labels,
        yticklabels=sorted_schedulers,
        cbar_kws={'label': 'Rank (1=Best)'},
        linewidths=0.5,
        ax=ax2,
        vmin=1,
        vmax=len(sorted_schedulers)
    )
    ax2.set_title(f'Ranking', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Workload', fontsize=11)
    ax2.set_ylabel('Scheduler (Sorted by Avg)', fontsize=11)
    ax2.tick_params(axis='x', rotation=45)

    plt.suptitle(f'Scheduler Ranking - {hw_config.upper()}\n'
                 f'Hardware: {HW_CONFIG_PRICES[hw_config]["total_price"]:.2f} $/hr',
                 fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_price_scatter(results, output_file='scatter_price_latency.png'):
    """
    Create scatter plot: latency vs price.
    Blue dots = heterogeneous, Red dots = homogeneous.
    Each point represents a scheduler-workload combination.
    """
    # Separate by hw_config
    hetero_results = [r for r in results if r['hw_config'] == 'hetero-5gpu']
    homo_results = [r for r in results if r['hw_config'] == 'homo-6gpu']

    # Extract data
    hetero_latencies = [r['avg_latency'] for r in hetero_results]
    hetero_prices = [HW_CONFIG_PRICES['hetero-5gpu']['total_price']] * len(hetero_results)

    homo_latencies = [r['avg_latency'] for r in homo_results]
    homo_prices = [HW_CONFIG_PRICES['homo-6gpu']['total_price']] * len(homo_results)

    # Create plot
    plt.figure(figsize=(12, 8))

    # Add some jitter to x-axis for better visualization
    hetero_prices_jittered = np.array(hetero_prices) + np.random.normal(0, 0.02, len(hetero_prices))
    homo_prices_jittered = np.array(homo_prices) + np.random.normal(0, 0.02, len(homo_prices))

    plt.scatter(hetero_prices_jittered, hetero_latencies,
               c='blue', alpha=0.6, s=100, label='Heterogeneous (5 GPU)', edgecolors='black', linewidth=0.5)
    plt.scatter(homo_prices_jittered, homo_latencies,
               c='red', alpha=0.6, s=100, label='Homogeneous (6 GPU)', edgecolors='black', linewidth=0.5)

    # Add mean markers
    if hetero_latencies:
        hetero_mean = np.mean(hetero_latencies)
        plt.scatter([HW_CONFIG_PRICES['hetero-5gpu']['total_price']], [hetero_mean],
                   c='darkblue', s=300, marker='*', label=f'Hetero Mean: {hetero_mean:.2f}s',
                   edgecolors='black', linewidth=2, zorder=5)

    if homo_latencies:
        homo_mean = np.mean(homo_latencies)
        plt.scatter([HW_CONFIG_PRICES['homo-6gpu']['total_price']], [homo_mean],
                   c='darkred', s=300, marker='*', label=f'Homo Mean: {homo_mean:.2f}s',
                   edgecolors='black', linewidth=2, zorder=5)

    plt.xlabel('Hardware Cost ($/hr)', fontsize=12)
    plt.ylabel('Latency (s)', fontsize=12)
    plt.title('Latency vs Price: Heterogeneous vs Homogeneous\n'
              'Each point = scheduler-workload combination',
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)

    # Add vertical lines at actual prices
    plt.axvline(x=HW_CONFIG_PRICES['hetero-5gpu']['total_price'],
               color='blue', linestyle='--', alpha=0.3, linewidth=2)
    plt.axvline(x=HW_CONFIG_PRICES['homo-6gpu']['total_price'],
               color='red', linestyle='--', alpha=0.3, linewidth=2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    # Create output directory
    output_dir = 'experiment_results/plots'
    os.makedirs(output_dir, exist_ok=True)

    # Load results
    print("Loading experiment results...")
    results = load_results('experiment_results')
    print(f"Loaded {len(results)} experiment results")

    # Check available hw_configs
    hw_configs = sorted(list(set(r['hw_config'] for r in results)))
    print(f"Available hardware configs: {hw_configs}")

    print("\nGenerating visualizations...")

    # Plot 1: Latency heatmap for heterogeneous config
    print("\n1. Creating latency heatmap (heterogeneous)...")
    create_latency_heatmap(
        results,
        hw_config='hetero-5gpu',
        output_file=os.path.join(output_dir, 'heatmap_latency_hetero.png')
    )

    # Also create for homogeneous
    print("   Creating latency heatmap (homogeneous)...")
    create_latency_heatmap(
        results,
        hw_config='homo-6gpu',
        output_file=os.path.join(output_dir, 'heatmap_latency_homo.png')
    )

    # Plot 2: Ranking heatmap
    print("\n2. Creating ranking heatmap (heterogeneous)...")
    create_ranking_heatmap(
        results,
        hw_config='hetero-5gpu',
        output_file=os.path.join(output_dir, 'heatmap_ranking_hetero.png')
    )

    print("   Creating ranking heatmap (homogeneous)...")
    create_ranking_heatmap(
        results,
        hw_config='homo-6gpu',
        output_file=os.path.join(output_dir, 'heatmap_ranking_homo.png')
    )

    # Plot 3: Price scatter plot
    print("\n3. Creating price vs latency scatter plot...")
    create_price_scatter(
        results,
        output_file=os.path.join(output_dir, 'scatter_price_latency.png')
    )

    print(f"\n{'='*70}")
    print("VISUALIZATION COMPLETE")
    print(f"{'='*70}")
    print(f"All plots saved to: {output_dir}/")
    print(f"  - heatmap_throughput_hetero.png")
    print(f"  - heatmap_throughput_homo.png")
    print(f"  - heatmap_ranking_hetero.png")
    print(f"  - heatmap_ranking_homo.png")
    print(f"  - scatter_price_throughput.png")


if __name__ == "__main__":
    main()
