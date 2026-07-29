# Visualization Tools

This directory contains all visualization scripts and notebooks for analyzing experiment results.

## Files

- **`plot_results.py`** - Main visualization script for scheduling experiments
  - Generates heatmaps of throughput (scheduler vs workload)
  - Creates ranking visualizations sorted by average performance
  - Produces scatter plots comparing heterogeneous vs homogeneous configs by price

- **`plot.ipynb`** - Jupyter notebook for interactive analysis

- **`plot_utils.py`** - Utility functions for plotting

- **`visualize_analyzer.py`** - Roofline analyzer visualization tools

## Usage

### Generate Scheduling Experiment Plots

After running the scheduling experiments:

```bash
# Run from the repository root
python cli/visualization/plot_results.py
```

This will generate 5 plots in `experiment_results/plots/`:

1. **`heatmap_throughput_hetero.png`** - Throughput heatmap for 5-GPU heterogeneous config
2. **`heatmap_throughput_homo.png`** - Throughput heatmap for 6-GPU homogeneous config
3. **`heatmap_ranking_hetero.png`** - Scheduler ranking (sorted by avg) for heterogeneous
4. **`heatmap_ranking_homo.png`** - Scheduler ranking (sorted by avg) for homogeneous
5. **`scatter_price_throughput.png`** - Price vs throughput comparison (blue=hetero, red=homo)

### Hardware Configurations

The plots compare two configurations:

**Heterogeneous (5 GPUs) - $3.20/hr**
- 1x RTX3090 (71 TFLOPS) - $0.13/hr
- 1x L40S (181 TFLOPS) - $0.44/hr
- 1x A6000 (210 TFLOPS) - $0.39/hr
- 1x A100 80GB SXM (312 TFLOPS) - $0.68/hr
- 1x H100 SXM (989 TFLOPS) - $1.56/hr
- **Total: ~1763 TFLOPS**

**Homogeneous (6 GPUs) - $4.08/hr**
- 6x A100 80GB SXM (312 TFLOPS each) - $0.68/hr each
- **Total: ~1872 TFLOPS**

### Dependencies

```bash
pip install matplotlib seaborn numpy
```

## Interactive Analysis

Use the Jupyter notebook for interactive exploration:

```bash
jupyter notebook cli/visualization/plot.ipynb
```
