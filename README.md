# HyTab-GNN: Layout-Respecting Minimal Connectivity Repair for Fine-Scale Fruit-Gain Prediction

## Overview

**HyTab-GNN** is a graph-learning framework for fine-scale fruit-gain prediction in wild blueberry mechanical harvesting. The core contribution is **LMCR (Layout-Respecting Minimal Connectivity Repair)** — a graph construction strategy that preserves a trusted spatial layout while adding only the minimum similarity edges needed to repair under-connected nodes.

### Key Finding

All three spatial-only GNN variants (GraphSAGE, GCN, GAT) trained on the trusted layout adjacency **underperform** a simple tabular MLP. LMCR fixes this: the Hybrid GraphSAGE achieves **test RMSE = 1708 g (R² = 0.731)**, a 21.8% improvement over spatial-only GraphSAGE, by adding just 29 edges (3.2% increase) to the layout backbone.

| Model | Graph | Test RMSE (g) | Test R² |
|-------|-------|---------------|---------|
| **Hybrid GraphSAGE** | **A_H (LMCR)** | **1708** | **0.731** |
| MLP Baseline | None | 2186 | 0.561 |
| Spatial GraphSAGE | A_S only | 2184 | 0.560 |
| Spatial GCN | A_S only | 2224 | 0.544 |
| Spatial GAT | A_S only | 2260 | 0.529 |

Multi-seed evaluation (10 seeds): mean test RMSE = 1910 ± 55 g (R² = 0.663 ± 0.019).

## Repository Structure

```
HyTab-GNN/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/                         # Source code
│   ├── generate_figures.py      # Manuscript figure generator
│   └── __init__.py
├── notebooks/                   # Colab pipeline
│   └── HyTab_GNN_Full_Pipeline.py
├── data/
│   ├── raw/GNN_FruitGain.csv    # 532 quadrat observations
│   └── processed/               # Preprocessed arrays
├── results/
│   ├── predictions/             # Per-model (observed, predicted) CSVs
│   ├── metrics/                 # Summary tables, seeds, importance
│   └── leaderboard.csv
├── figures/
│   ├── png/                     # 300 DPI figures
│   ├── pdf/                     # Vector figures
│   └── csv_evidence/            # Raw data behind every figure
└── docs/
```

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/HyTab-GNN.git
cd HyTab-GNN
pip install -r requirements.txt
python src/generate_figures.py
```

## Reproducibility

Every figure has a matching CSV in `figures/csv_evidence/`. To verify:

```python
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

df = pd.read_csv("results/predictions/preds_hybrid_sage_test.csv")
rmse = np.sqrt(mean_squared_error(df['y'], df['yhat']))
r2 = r2_score(df['y'], df['yhat'])
print(f"RMSE = {rmse:.1f} g, R² = {r2:.4f}")
# Output: RMSE = 1708.4 g, R² = 0.7309
```

## Dataset

- **Crop:** Wild blueberry
- **Scale:** 25×25 cm quadrats, n = 532
- **Seasons:** 2024–2025, 4 farms, 4 harvesting mechanisms
- **Target:** Fruit gain (g)
- **Features:** 12 variables (crop/canopy + harvesting + site)

## License

MIT License — see [LICENSE](LICENSE) for details.
