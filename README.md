# Benchmark Module

Comprehensive benchmarking and analysis tools for evaluating molecular dynamics simulations and machine learning models against ground truth data.

## Overview

This module provides a complete framework for:
- Evaluating ML-driven and traditional MD simulations
- Computing statistical metrics (KL divergence, Wasserstein distance, RMSD)
- Analyzing trajectory quality through TICA/PCA decomposition
- Generating detailed comparison reports with visualizations
- Building Markov State Models (MSM) for macrostate analysis
-
## Status

⚠️ **IMPORTANT**: Our code is currently being ported and refactored from private repositories for public release. The full codebase with documentation and tutorials will be provided within one to two weeks.

## Features

### Analysis Capabilities

- **Dimensionality Reduction**: TICA and PCA analysis for trajectory comparison
- **Contact Maps**: Generation and comparison of contact frequency maps
- **Reaction Coordinates**: End-to-end distance and radius of gyration analysis
- **Bond Geometry**: Bond length, angle, and dihedral distribution analysis
- **MSM Analysis**: Macrostate identification and equilibrium probability calculations
- **Statistical Metrics**:
  - Kullback-Leibler (KL) divergence
  - Wasserstein distance (1D and 2D)
  - RMSD to experimental structures
  
### Visualization

- TICA/PCA space projections with macrostate coloring
- Contour plots comparing model vs. ground truth distributions
- Contact map difference plots
- PDF comparisons for various structural metrics
- Comprehensive HTML reports with embedded visualizations

### Supported Input Types

- **Model Trajectories**: HDF5 trajectory files from trained models
- **WESTPA Simulations**: Weighted ensemble trajectories with importance sampling
- **Traditional MD**: Standard molecular dynamics trajectory files
- **Previous Benchmarks**: Re-analysis of existing benchmark results

## Installation

### Prerequisites

```bash
# Core dependencies
- Python 3.10+
- CUDA-capable GPU (for KDE calculations)
- MDTraj
- DeepTime
- NumPy, SciPy, Matplotlib
- Weights & Biases (optional, for logging)
```

### Setup

*Detailed installation instructions will be added soon.*

## Usage

### Basic Benchmark Run

```bash
python gen_benchmark.py \
    --temperature 300 \
    --ref-data /path/to/reference/data \
    --proteins bba chignolin trpcage \
    --trajs-folder /path/to/trajectories \
    --gpus 0,1,2
```

### WESTPA Analysis

```bash
python gen_benchmark.py \
    --temperature 300 \
    --ref-data /path/to/reference/data \
    --proteins proteinb \
    --trajs-folder /path/to/westpa/output \
    --westpa-weights /path/to/weights.h5 \
    --traj-extension dcd
```

### Re-running Previous Benchmark

```bash
python gen_benchmark.py \
    --temperature 300 \
    --ref-data /path/to/reference/data \
    --proteins bba chignolin \
    --old-benchmark-dir /path/to/previous/benchmark
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--temperature` | Temperature in Kelvin (300 or 350) | Required |
| `--ref-data` | Reference data root directory | Required |
| `--proteins` | List of proteins to benchmark | Required |
| `--output-dir` | Output directory for results | Auto-generated |
| `--use-cache` | Use cached intermediate results | True |
| `--only-gen-cache` | Only regenerate cached data | False |
| `--trajs-folder` | Directory with trajectory files | None |
| `--component-analysis-type` | TICA or PCA | TICA |
| `--gpus` | Comma-separated GPU IDs | All available |
| `--disable-wandb` | Disable Weights & Biases logging | False |
| `--enable-msm-metrics` | Enable MSM macrostate statistics | True |
| `--calc-kl-divergence` | Calculate KL divergence metrics | False |
| `--westpa-weights` | Path to WESTPA weights file | None |
| `--westpa-cut` | Use only 1 frame per trajectory | False |
| `--max-westpa-trajs` | Maximum trajectories to load | None |
| `--ssmsm` | Enable MSM KDE density overlay | False |

## Output Structure

```
benchmark_output/
├── benchmark.json              # Metadata and file paths
├── metrics.json                # Computed metrics summary
├── tica_spaces_{protein}.png   # TICA projection plots
├── tica_contours_{protein}.png # Contour comparisons
├── contact_map_{protein}.png   # Contact map differences
├── plot_pdfs_{protein}.png     # PDF comparisons
├── plot_gyration_pdf_{protein}.png  # Radius of gyration
└── {protein}_model_replicas.pkl     # Trajectory data
```

## Report Generation

The `gen_report.py` script processes benchmark results and generates comprehensive visualizations:

```bash
python gen_report.py /path/to/benchmark.json \
    --also-plot-locally \
    --do-rmsd-metrics \
    --calc-kl-divergence
```

### Metrics Calculated

1. **TICA/PCA Metrics**
   - 1D KL divergence per component
   - 2D KL divergence (first two components)
   - 1D and 2D Wasserstein distances

2. **Structural Metrics**
   - Bond length, angle, and dihedral distributions
   - Contact map similarity
   - Radius of gyration distributions
   - Reaction coordinate (end-to-end distance)

3. **MSM Metrics** (if enabled)
   - Native macrostate identification
   - Equilibrium probability of native state
   - Mean and standard deviation of RMSD to experimental structure

## Architecture

### Core Modules

- **`gen_benchmark.py`**: Main benchmarking orchestration
- **`gen_report.py`**: Visualization and metrics reporting
- **`report_generator/`**: Analysis components
  - `tica_plots.py`: Dimensionality reduction
  - `contact_maps.py`: Contact frequency analysis
  - `bond_and_angle_analysis.py`: Geometric analysis
  - `msm_analysis.py`: Markov State Model construction
  - `reaction_coordinate.py`: Reaction coordinate calculations
  - `kullback_leibler_divergence.py`: Statistical comparisons

### Caching System

The benchmark module implements intelligent caching to avoid redundant computations:
- TICA/PCA models
- Native trajectory preprocessed data
- Contact maps
- Bond/angle distributions
- MSM models

Use `--use-cache` (default) to leverage cached results, or `--no-use-cache` to regenerate all data.

## Reference Data Structure

Expected directory structure for `--ref-data`:

```
ref_data/
├── data300K/          # 300K reference trajectories
│   ├── bba/
│   ├── chignolin/
│   └── ...
├── data350K/          # 350K reference trajectories if they exist
├── cache/             # Cached intermediate results
├── sims/              # Benchmark output storage
└── rmsd/              # Experimental structures (.pdb)
```

## Performance Considerations

### Memory Usage

- Large proteins (e.g., proteinb) can consume 150-200GB RAM
- Use `--max-westpa-trajs` to limit memory for WESTPA analysis
- Stride parameters in code control memory footprint

### Parallelization

- Multi-GPU support for trajectory generation
- Thread pool for parallel protein processing
- ProcessPoolExecutor for trajectory loading

### Optimization Tips

1. **Use caching**: Keep `--use-cache` enabled for iterative analysis
2. **GPU allocation**: Assign one GPU per protein with `--gpus`
3. **Memory management**: Use trajectory striding for large datasets
4. **WESTPA trajectories**: Consider `--westpa-cut` for memory-constrained systems

## Integration with Other Modules

### WESTPA Analysis

The benchmark module integrates with WESTPA simulations through:
- Weighted trajectory analysis
- Stationary distribution calculations
- Importance sampling corrections

### Model Evaluation

Supports evaluation of:
- CGSchNet-based generative models
- Traditional MD simulations
- Hybrid ML/MD approaches

## Troubleshooting

### Common Issues

**Out of Memory Errors**
```bash
# Solution 1: Reduce trajectory count
--max-westpa-trajs 1000

# Solution 2: Increase stride
# Edit NATIVE_PATHS_STRIDE in gen_benchmark.py

# Solution 3: Use trajectory cutting
--westpa-cut
```

**Missing Cache Files**
```bash
# Regenerate all caches
python gen_benchmark.py --no-use-cache ...
```

**GPU Errors**
```bash
# Specify available GPUs explicitly
--gpus 0,1
```

## Contributing

We welcome contributions! Please use [GitHub Issues](../../issues) to:
- Report bugs
- Request features
- Submit improvements
- Ask questions

## License

*License information will be added soon.*

## Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This module is under active development. Documentation, tutorials, and additional features will be added as the codebase is finalized for public release.
