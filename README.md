# Hybrid Meta-learners for Estimating Heterogeneous Treatment Effects
#### Zhongyuan Liang, Lars van der Laan, Ahmed Alaa

This repository contains the code for the paper "Hybrid Meta-learners for Estimating Heterogeneous Treatment Effects". It includes the implementation of the proposed **Hybrid Learner (H-learner)** and code to reproduce the experiments presented in the paper.

---

## Requirements

Download the codebase from source and install all dependencies in requirements.txt.

```bash
pip install -r requirements.txt
```

---

## Datasets
The IHDP 1000 dataset can be downloaded here: https://www.fredjo.com/
The ACIC 2016 dataset can be downloaded from the official competition website:
https://jenniferhill7.wixsite.com/acic-2016/competition

## Example Usage

```python
from src.dataset import *
from src.models import *

# Load the IHDP dataset (example usage)
X_train, t_train, y_train, mu0_train, mu1_train, X_test, mu0_test, mu1_test = load_ihdp_1000_data(index=1)

# ----- First Stage of H-learner: Estimate nuisance parameters -----

# Estimate potential outcomes with TARNet
tarnet = TARNet(input_dim=X_train.shape[1], lr=[0.0001], epochs=1000)
tarnet.fit(X_train, y_train, t_train)
_, stage1_y0_pred, stage1_y1_pred = tarnet.predict(X_train, return_po=True)

# Estimate propensity scores
p = PropensityModel(input_dim=X_train.shape[1], lr=[0.0001], epochs=1000)
p.fit(X_train, t_train)
stage1_p_pred = p.predict(X_train)

# ----- Second Stage of H-learner: Fit the hybrid model -----

h_learner_x = HLearner(
    input_dim=X_train.shape[1], learner_type="X", alpha=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], 
    lr=[0.0001], epochs=1000
)
h_learner_x.fit(X_train, y_train, t_train, stage1_y0_pred, stage1_y1_pred, stage1_p_pred)

# Predict CATE on train and test sets
cate_pred_train = h_learner_x.predict(X_train)
cate_pred_test = h_learner_x.predict(X_test)
```

---

## Reproducibility

All experiments from the paper can be reproduced using the files provided in the experiments/ directory.
Semi-synthetic results can be reproduced by running run_synthetic.sh and visualized using experiments/synthetic_visualizations.ipynb.
Benchmark results for IHDP1000 and ACIC2016 can be reproduced by running run_ihdp.sh and run_acic.sh. The results can then be aggregated using experiments/benchmark_results_summary.ipynb.


