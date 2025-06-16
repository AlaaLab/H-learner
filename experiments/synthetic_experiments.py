import numpy as np
import argparse
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import *
from src.models import *
from src.utils import *

def generate_synthetic_data(setting, ratio, seed=0):
    np.random.seed(seed)

    X_train, _, _, _, _, X_test, _, _ = load_ihdp_1000_data(index=0)
    
    def mu0_fn(x):
        np.random.seed(seed)
        num_features = x.shape[1]
        mu0_features = np.random.choice(num_features, size=10, replace=False)
        coefs = np.zeros(num_features)
        coefs[mu0_features] = 1
        first_order = np.dot(x, coefs)
        second_order = np.sum(x[:, mu0_features] ** 2, axis=1)
        interaction_effects = np.sum([
            x[:, i] * x[:, j]
            for idx_i, i in enumerate(mu0_features)
            for j in mu0_features[idx_i + 1:]
        ], axis=0)
        return mu0_features, 2*first_order+2*second_order+interaction_effects

    def mu1_fn(mu0_features, x, overlap_ratio):
        np.random.seed(seed)
        num_features = x.shape[1]
        shared_features = np.random.choice(mu0_features, size=int(len(mu0_features) * overlap_ratio), replace=False)
        remaining_pool = np.setdiff1d(np.arange(num_features), mu0_features)
        new_features = np.random.choice(remaining_pool, size=10 - len(shared_features), replace=False)
        mu1_features = np.concatenate([shared_features, new_features])
        coefs = np.zeros(num_features)
        coefs[mu1_features] = 1
        first_order = np.dot(x, coefs)
        second_order = np.sum(x[:, mu1_features] ** 2, axis=1)
        interaction_effects = np.sum([
            x[:, i] * x[:, j]
            for idx_i, i in enumerate(mu1_features)
            for j in mu1_features[idx_i + 1:]
        ], axis=0)
        return mu1_features, 2*first_order+2*second_order+interaction_effects
    
    if setting == "A":
        mu0_features_train, mu0_train = mu0_fn(X_train)
        mu1_features_train, mu1_train = mu1_fn(mu0_features_train, X_train, overlap_ratio=ratio)
        mu0_features_test, mu0_test = mu0_fn(X_test)
        mu1_features_test, mu1_test = mu1_fn(mu0_features_test, X_test, overlap_ratio=ratio)
        t_train = np.random.binomial(1, p=0.5, size=X_train.shape[0])

    elif setting == "B":
        mu0_features_train, mu0_train = mu0_fn(X_train)
        mu1_features_train, mu1_train = mu1_fn(mu0_features_train, X_train, overlap_ratio=0.4)
        mu0_features_test, mu0_test = mu0_fn(X_test)
        mu1_features_test, mu1_test = mu1_fn(mu0_features_test, X_test, overlap_ratio=0.4)
        t_train = np.random.binomial(1, p=ratio, size=X_train.shape[0])

    elif setting == "C":
        mu0_features_train, mu0_train = mu0_fn(X_train)
        mu1_features_train, mu1_train = mu1_fn(mu0_features_train, X_train, overlap_ratio=0.4)
        mu0_features_test, mu0_test = mu0_fn(X_test)
        mu1_features_test, mu1_test = mu1_fn(mu0_features_test, X_test, overlap_ratio=0.4)
        confounding_features = np.union1d(mu0_features_train, mu1_features_train)
        confounding_coefs = np.random.randn(len(confounding_features))
        logits = X_train[:, confounding_features] @ confounding_coefs
        logits *= ratio
        treatment_prob = 1 / (1 + np.exp(-logits))
        t_train = np.random.binomial(1, p=treatment_prob)

    noise = np.random.normal(0, 1, size=X_train.shape[0])
    y_train = mu0_train * (1 - t_train) + mu1_train * t_train + noise

    return (
        X_train, t_train, y_train, mu0_train, mu1_train,
        X_test, mu0_test, mu1_test,
        mu0_features_train, mu1_features_train,
        mu0_features_test, mu1_features_test
    )


def main(setting, ratio, seed):
    results = []

    X_train, t_train, y_train, mu0_train, mu1_train, X_test, mu0_test, mu1_test, mu0_features_train, mu1_features_train, mu0_features_test, mu1_features_test = generate_synthetic_data(
        setting=setting, ratio=ratio, seed=seed
    )

    tarnet = TARNet(input_dim=X_train.shape[1])
    tarnet.fit(X_train, y_train, t_train)
    cate_pred_train, stage1_y0_prediction, stage1_y1_prediction = tarnet.predict(X_train, return_po=True)
    cate_pred_test = tarnet.predict(X_test)
    pehe_train, sqrt_pehe_train, pehe_test, sqrt_pehe_test = cate_evaluations(cate_pred_train, cate_pred_test, mu0_train, mu1_train, mu0_test, mu1_test)
    results.append({
        "setting": setting,
        "ratio": ratio,
        "seed": seed,
        "model": "TARNet",
        "pehe_test": pehe_test,
        "reg_lambda": None,
    })

    p = PropensityModel(input_dim=X_train.shape[1])
    p.fit(X_train, t_train)
    stage1_p_prediction = p.predict(X_train)

    X = DirectLearner(input_dim=X_train.shape[1], learner_type="X")
    X.fit(X_train, y_train, t_train, stage1_y0_prediction, stage1_y1_prediction, stage1_p_prediction)
    cate_pred_train = X.predict(X_train)
    cate_pred_test = X.predict(X_test)
    pehe_train, sqrt_pehe_train, pehe_test, sqrt_pehe_test = cate_evaluations(cate_pred_train, cate_pred_test, mu0_train, mu1_train, mu0_test, mu1_test)
    results.append({
        "setting": setting,
        "ratio": ratio,
        "seed": seed,
        "model": "X_learner",
        "pehe_test": pehe_test,
        "reg_lambda": None,
    })

    reg_lambda_lst = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    h_learner_x = HLearner(input_dim=X_train.shape[1], learner_type="X", reg_lambda=reg_lambda_lst)
    h_learner_x.fit(X_train, y_train, t_train, stage1_y0_prediction, stage1_y1_prediction, stage1_p_prediction)
    cate_pred_train = h_learner_x.predict(X_train)
    cate_pred_test = h_learner_x.predict(X_test)
    pehe_train, sqrt_pehe_train, pehe_test, sqrt_pehe_test = cate_evaluations(cate_pred_train, cate_pred_test, mu0_train, mu1_train, mu0_test, mu1_test)
    results.append({
        "setting": setting,
        "ratio": ratio,
        "seed": seed,
        "model": "H_learner (X)",
        "pehe_test": pehe_test,
        "reg_lambda": None,
    })

    for reg_lambda in reg_lambda_lst:
        h_learner_x.load_model_for_lambda(reg_lambda)
        cate_pred_train = h_learner_x.predict(X_train)
        cate_pred_test = h_learner_x.predict(X_test)
        pehe_train, sqrt_pehe_train, pehe_test, sqrt_pehe_test = cate_evaluations(cate_pred_train, cate_pred_test, mu0_train, mu1_train, mu0_test, mu1_test)
        results.append({
            "setting": setting,
            "ratio": ratio,
            "seed": seed,
            "model": "H_learner (X)",
            "pehe_test": pehe_test,
            "reg_lambda": reg_lambda,
        })

    results_df = pd.DataFrame(results)
    os.makedirs("./results", exist_ok=True)
    results_df.to_csv(f"./results/synthetic_results_{setting}_{ratio}_{seed}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setting", type=str, required=True)
    parser.add_argument("--ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    main(args.setting, args.ratio, args.seed)