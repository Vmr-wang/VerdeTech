# VerdeTech 

**SustainableML: Energy-Performance Trade-offs in Python Libraries for Machine Learning**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Ubuntu 20.04](https://img.shields.io/badge/Ubuntu-20.04-orange.svg)](https://ubuntu.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Comprehensive benchmarking framework for measuring energy consumption and performance trade-offs across Python machine learning libraries.

## Overview

VerdeTech systematically evaluates the energy efficiency and performance characteristics of different Python ML libraries implementing identical algorithms. This research addresses the critical gap between ML performance optimization and energy sustainability.

### Key Contributions

- **Empirical Energy Analysis**: Measure real energy consumption using RAPL (CPU) and NVML (GPU)
- **Cross-Library Comparison**: Compare scikit-learn, XGBoost, PyTorch, TensorFlow, and scikit-learn-intelex
- **Statistical Rigor**: 780 experimental runs with 20 repetitions per configuration
- **Hardware Coverage**: CPU-only and GPU-accelerated implementations
- **Practical Insights**: Energy-performance trade-offs for sustainable ML development

## Research Questions

**RQ1**: What are the energy usage differences between Python ML libraries for identical algorithms?

**RQ2**: How does energy usage relate to performance aspects and what trade-offs emerge?
- **RQ2.1**: Energy vs. Memory utilization correlation
- **RQ2.2**: Energy vs. Execution time trade-offs  
- **RQ2.3**: Energy vs. CPU/GPU utilization relationships
- **RQ2.4**: Energy vs. ML effectiveness metrics balance

## Experimental Setup

### Hardware Configuration
- **OS**: Ubuntu 20.04.6 LTS (GNU/Linux 5.15.0-107-generic, x86_64)
- **CPU**: 11th Gen Intel Core i7-1165G7 @ 2.80 GHz (4 cores, 8 threads)
- **GPU**: NVIDIA GeForce MX450 Laptop
- **RAM**: 16 GB
- **Storage**: 1.4 TB SSD

### Measurement Tools
- **EnergiBridge**: Primary energy measurement (RAPL for CPU, NVML for GPU)
- **ExperimentRunner**: Orchestration and scheduling framework
- **ps**: Real-time system monitoring
- **Python 3**: All implementations with pinned dependencies

### Experimental Design
- **Factors**: Algorithm type (4 levels), Library (5 levels), Hardware (2 levels), Dataset size (3 levels)
- **Treatments**: 13 valid algorithm-library-hardware combinations × 3 dataset sizes = 39 treatments
- **Repetitions**: 20 runs per treatment (780 total runs)
- **Duration**: ~50 hours estimated execution time
- **Cool-down**: 3-minute intervals between runs for thermal stability

## Installation

### Prerequisites
- Ubuntu 20.04+ (recommended) or similar Linux distribution
- Python 3.8+
- NVIDIA GPU with CUDA support (optional, for GPU benchmarks)
- 16 GB RAM minimum

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/Vmr-wang/VerdeTech.git
cd VerdeTech
```

2. **Create conda environment**
```bash
# Create environment from configuration
conda env create -f environment.yml

# Activate environment
conda activate verdetech

# Verify installation
python -c "import sklearn, xgboost, torch, tensorflow as tf; print('All libraries installed successfully!')"
```

3. **Alternative: pip installation**
```bash
# Create virtual environment
python3 -m venv verdetech
source verdetech/bin/activate

# Install requirements
pip install -r requirements.txt
```

4. **Install additional tools**
```bash
# Install EnergiBridge (if not included)
# Follow EnergiBridge installation instructions for your system

# Verify energy measurement capability
sudo python -c "import subprocess; print('RAPL available:', 'intel-rapl' in str(subprocess.run(['ls', '/sys/class/powercap/'], capture_output=True)))"
```

## Library Coverage

| Library | Version | Hardware Support | Algorithms |
|---------|---------|------------------|------------|
| **scikit-learn** | 1.6.1 | CPU | Logistic Regression, Decision Tree, K-Means, Ridge Regression |
| **scikit-learn-intelex** | 2025.0.0 | CPU (Intel optimized) | Logistic Regression, K-Means, Ridge Regression |
| **XGBoost** | 3.0.1 | CPU + GPU | Decision Tree Classifier, Ridge Regression |
| **PyTorch** | 2.8.0 | CPU + GPU | Logistic Regression, Ridge Regression |
| **TensorFlow** | 2.16.1 | CPU + GPU | K-Means |

## Usage

### Running the Full Benchmark

```bash
# Execute complete experimental suite
python experiments/run_full_benchmark.py

# Monitor progress
tail -f logs/experiment.log
```

### Individual Algorithm Testing

```bash
# Test specific algorithm-library combination
python experiments/single_test.py --algorithm "Logistic Regression" --library scikit-learn --hardware cpu

# Test with specific dataset size
python experiments/single_test.py --algorithm "K-Means" --library tensorflow --hardware gpu --dataset_size medium
```

### Custom Configuration

Create your own configuration file:

```yaml
# config/custom_experiment.yml
experiment:
  repetitions: 20
  cooldown_minutes: 3
  warmup_runs: 3

measurements:
  energy_sampling_rate: 10  # Hz
  monitor_memory: true
  monitor_cpu: true
  monitor_gpu: true

algorithms:
  - name: "Logistic Regression"
    libraries: ["scikit-learn", "pytorch", "scikit-learn-intelex"]
    datasets: ["small", "medium", "large"]
    hardware: ["cpu", "gpu"]

output:
  directory: "results/"
  format: "csv"
  include_raw: true
```

Then run:
```bash
python experiments/run_custom.py --config config/custom_experiment.yml
```

## Project Structure

```
VerdeTech/
├── experiments/
│   ├── run_full_benchmark.py    # Main experiment script
│   ├── single_test.py           # Individual test runner
│   ├── run_custom.py           # Custom configuration runner
│   └── config/                 # Experiment configurations
├── src/
│   ├── benchmarks/             # Benchmarking framework
│   ├── energy/                 # Energy measurement integration
│   ├── algorithms/             # Algorithm implementations
│   ├── datasets/               # Dataset management
│   └── analysis/               # Statistical analysis tools
├── data/
│   ├── datasets/               # Benchmark datasets
│   └── results/                # Experimental results
├── scripts/
│   ├── setup_environment.sh    # Environment setup
│   ├── install_energibridge.sh # EnergiBridge installation
│   └── verify_setup.py        # Installation verification
├── analysis/
│   ├── statistical_tests.R     # R scripts for hypothesis testing
│   ├── visualizations.py       # Result plotting
│   └── report_generation.py    # Automated reporting
├── logs/                       # Execution logs
├── environment.yml             # Conda environment
├── requirements.txt            # pip requirements
├── environment-dev.yml         # Development environment
└── run_table.csv              # Results database (generated)
```

## Experimental Execution Plan

The complete experiment consists of 13 algorithm-library-hardware combinations:

| Algorithm | Library | Hardware | Dataset Sizes |
|-----------|---------|----------|---------------|
| Logistic Regression | scikit-learn | CPU | Small, Medium, Large |
| Logistic Regression | PyTorch | GPU | Small, Medium, Large |
| Logistic Regression | scikit-learn-intelex | CPU | Small, Medium, Large |
| K-Means | scikit-learn | CPU | Small, Medium, Large |
| K-Means | scikit-learn-intelex | CPU | Small, Medium, Large |
| K-Means | TensorFlow | GPU | Small, Medium, Large |
| Decision Tree | scikit-learn | CPU | Small, Medium, Large |
| Decision Tree | XGBoost | CPU | Small, Medium, Large |
| Decision Tree | XGBoost | GPU | Small, Medium, Large |
| Ridge Regression | XGBoost | GPU | Small, Medium, Large |
| Ridge Regression | PyTorch | GPU | Small, Medium, Large |
| Ridge Regression | scikit-learn | CPU | Small, Medium, Large |
| Ridge Regression | scikit-learn-intelex | CPU | Small, Medium, Large |

**Total**: 39 treatments × 20 repetitions = 780 experimental runs

### Estimated Execution Times
- **Small datasets** (Iris, Auto-mpg): ~10 seconds per run
- **Medium datasets** (Adult, California Housing): ~30 seconds per run  
- **Large datasets** (MNIST, NYC Taxi): ~90 seconds per run

**Total estimated duration**: ~50 hours including cool-down periods

## Data Analysis

### Statistical Methods
- **Normality Testing**: Shapiro-Wilk test
- **Variance Homogeneity**: Levene's test
- **Primary Analysis**: One-way ANOVA (parametric) or Kruskal-Wallis (non-parametric)
- **Post-hoc Testing**: Tukey's HSD or Dunn's test with Bonferroni correction
- **Correlation Analysis**: Pearson or Spearman correlation
- **Effect Size**: Cohen's d, Eta-squared, or Cliff's delta

### Output Files
- **run_table.csv**: Complete experimental results
- **statistical_results.json**: Hypothesis testing outcomes
- **energy_analysis.csv**: Energy consumption analysis
- **performance_correlations.csv**: Performance relationship data

## Results and Visualization

Results are automatically processed and visualized:

```bash
# Generate analysis reports
python analysis/statistical_tests.py --input run_table.csv

# Create visualizations
python analysis/visualizations.py --results run_table.csv --output plots/

# Generate final report
python analysis/report_generation.py --create-pdf
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make changes following the coding standards
4. Add tests for new functionality
5. Run the test suite (`python -m pytest tests/`)
6. Commit changes (`git commit -m "Add feature: description"`)
7. Push to branch (`git push origin feature/improvement`)
8. Create a Pull Request

## System Requirements

### Minimum Requirements
- Ubuntu 18.04+ or equivalent Linux distribution
- Python 3.8+
- 8 GB RAM
- 50 GB free disk space
- Intel processor with RAPL support

### Recommended Requirements
- Ubuntu 20.04 LTS
- Python 3.9+
- 16 GB RAM
- 100 GB free disk space
- NVIDIA GPU with CUDA support
- Intel processor with AVX-512 support

## Citation

If you use VerdeTech in your research, please cite:

```bibtex
@inproceedings{shan2025sustainableml,
  title={SustainableML: Energy–Performance Trade-offs in Python Libraries for Machine Learning},
  author={Shan, Haoru and Liu, Mingshuo and Wang, Xuan and Xia, Yuanhao and Dong, Zixin},
  booktitle={Green Lab 2025/2026 - Vrije Universiteit Amsterdam},
  year={2025},
  address={Amsterdam, The Netherlands}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- VU Amsterdam Green Lab Course 2025/2026
- EnergiBridge and ExperimentRunner frameworks
- Open-source ML library communities
- Research collaborators and advisors

---

**Making machine learning more sustainable through empirical energy analysis**
