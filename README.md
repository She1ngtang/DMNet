# When Temporal Granularity Meets Spatial Scale: A Dual-Stream Multi-Scale Framework for Traffic Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()  
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)]()  

## 🌐 Overview
Accurate traffic forecasting over heterogeneous Web of Things data is a fundamental challenge that drives large‑scale applications such as **dynamic routing, adaptive signal control, and socio‑economic analysis**.  

While recent advances in spatio-temporal deep learning have improved predictive accuracy, most existing models rely on a **single temporal resolution** and a **single spatial scale**, thereby overlooking critical dynamics that emerge at coarser or finer levels.  

To address this gap, we propose **DMNet**, a **Dual-Stream Multi-Scale Framework** that simultaneously captures traffic dynamics across **multiple temporal granularities** and **multiple spatial scales** within a unified architecture.

---

## 🚀 Key Contributions
- **Multi-Scale Spatial Encoder**  
  - Automatically discovers scale-specific metagraphs.  
  - Integrates them via localized graph convolutions, state-space diffusion, and cross-scale attention.  
  - Produces coherent sensor-level representations.  

- **Multi-Granularity Temporal Mixer**  
  - Models fine- and coarse-resolution signals jointly.  
  - Ensures temporal coherence through bidirectional refinement.  

- **Spatio-Temporal Integrator**  
  - Explicitly models interactions between temporal and spatial hierarchies.  
  - Employs intra-domain enhancement, cross-domain bridging, and a unified prediction head.  

- **Performance**  
  - On **PEMS04** and **PEMS08**, DMNet outperforms state-of-the-art baselines by up to **5.08% MAPE reduction**.  
  - Ablation studies confirm that coupling multi-granularity temporal and multi-scale spatial views yields significant gains.  

---

## 📂 Repository Structure
```
DMNet/
├── data/                 # Datasets (PEMS04, PEMS08, etc.)
├── model/                # Core model components (encoder, mixer, integrator)
├── scripts/              # Training and evaluation scripts
├── utils/                # Helper functions (metrics, preprocessing, logging)
├── configs/              # YAML configs for experiments
├── results/              # Saved models and logs
├── requirements.txt      # Dependencies
└── main.py               # Entry point
```

---

## ⚙️ Installation
```bash
# Enter the repository
cd DMNet

# Create environment (Python 3.8+ recommended)
conda create -n dmnet python=3.8
conda activate dmnet

# Install dependencies
pip install -r requirements.txt
```

---

## 📊 Datasets
We evaluate DMNet on **PEMS04** and **PEMS08** traffic datasets.  

- Download the PEMS04 and PEMS08 datasets.  
- Place the processed data under:  
  - `./data/PEMS04/`  
  - `./data/PEMS08/`  

---

## 🏃‍♂️ Usage

### Training
```bash
python main.py --config configs/pems04.yaml
```

### Evaluation
```bash
python main.py --config configs/pems04.yaml --evaluate
```

### 🔑 Key Arguments
- `--config`: Path to YAML config file.  
- `--evaluate`: Run evaluation only.  
- `--save_dir`: Directory to save checkpoints and logs.  

---

## 📈 Results

| Dataset | Metric | DMNet | Best Baseline | Improvement |
|---------|--------|-------|---------------|-------------|
| PEMS04  | MAPE   | **11.65** | 12.06 | 3.40% |
| PEMS08  | MAPE   | **8.96** | 9.44 | 5.08% |

---

## 🔬 Ablation Studies
- Removing **Multi-Granularity Temporal Mixer** → performance drops by ~7.0%.  
- Removing **Multi-Scale Spatial Encoder** → performance drops by ~2.7%.  
- Removing **Spatio-Temporal Integrator** → performance drops by ~0.7%.  

This confirms the necessity of **joint optimization** across both temporal and spatial hierarchies.  

---

## 📜 License
This project is licensed under the MIT License.  

---

## 🤝 Acknowledgements
We thank the open-source community for providing datasets and baseline implementations.
