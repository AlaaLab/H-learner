import argparse
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.dataset import *
from src.models import *
from src.utils import *

def main(folder, k):
    results = []

    print(f"Processing dataset index: {k}")
    X_train, t_train, y_train, mu0_train, mu1_train, X_test, mu0_test, mu1_test = load_acic_data(dgp_folder=folder, file_index=k)

    t = TLearner(input_dim=X_train.shape[1])
    t.fit(X_train, y_train, t_train)
    cate_pred_train = t.predict(X_train)
    cate_pred_test = t.predict(X_test)
    pehe_train, sqrt_pehe_train, pehe_test, sqrt_pehe_test = cate_evaluations(cate_pred_train, cate_pred_test, mu0_train, mu1_train, mu0_test, mu1_test)

    results.append({
        "folder": folder,
        "data_id": k,
        "model": "t_learner",
        "pehe_train": pehe_train,
        "sqrt_pehe_train": sqrt_pehe_train,
        "pehe_test": pehe_test,
        "sqrt_pehe_test": sqrt_pehe_test,
    })

    tarnet = TARNet(input_dim=X_train.shape[1])
    tarnet.fit(X_train, y_train, t_train)
    cate_pred_train, stage1_y0_prediction, stage1_y1_prediction = tarnet.predict(X_train, return_po=True)
    cate_pred_test = tarnet.predict(X_test)
    pehe_train, sqrt_pehe_train, pehe_test, sqrt_pehe_test = cate_evaluations(cate_pred_train, cate_pred_test, mu0_train, mu1_train, mu0_test, mu1_test)

    results.append({
        "folder": folder,
        "data_id": k,
        "model": "TARNet",
        "pehe_train": pehe_train,
        "sqrt_pehe_train": sqrt_pehe_train,
        "pehe_test": pehe_test,
        "sqrt_pehe_test": sqrt_pehe_test,
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
        "folder": folder,
        "data_id": k,
        "model": "X_learner",
        "pehe_train": pehe_train,
        "sqrt_pehe_train": sqrt_pehe_train,
        "pehe_test": pehe_test,
        "sqrt_pehe_test": sqrt_pehe_test,
    })

    DR = DirectLearner(input_dim=X_train.shape[1], learner_type="DR")
    DR.fit(X_train, y_train, t_train, stage1_y0_prediction, stage1_y1_prediction, stage1_p_prediction)
    cate_pred_train = DR.predict(X_train)
    cate_pred_test = DR.predict(X_test)
    pehe_train, sqrt_pehe_train, pehe_test, sqrt_pehe_test = cate_evaluations(cate_pred_train, cate_pred_test, mu0_train, mu1_train, mu0_test, mu1_test)

    results.append({
        "folder": folder,
        "data_id": k,
        "model": "DR_learner",
        "pehe_train": pehe_train,
        "sqrt_pehe_train": sqrt_pehe_train,
        "pehe_test": pehe_test,
        "sqrt_pehe_test": sqrt_pehe_test,
    })

    IPW = DirectLearner(input_dim=X_train.shape[1], learner_type="IPW")
    IPW.fit(X_train, y_train, t_train, stage1_y0_prediction, stage1_y1_prediction, stage1_p_prediction)
    cate_pred_train = IPW.predict(X_train)
    cate_pred_test = IPW.predict(X_test)
    pehe_train, sqrt_pehe_train, pehe_test, sqrt_pehe_test = cate_evaluations(cate_pred_train, cate_pred_test, mu0_train, mu1_train, mu0_test, mu1_test)

    results.append({
        "folder": folder,
        "data_id": k,
        "model": "IPW_learner",
        "pehe_train": pehe_train,
        "sqrt_pehe_train": sqrt_pehe_train,
        "pehe_test": pehe_test,
        "sqrt_pehe_test": sqrt_pehe_test,
    })

    h_learner_x = HLearner(input_dim=X_train.shape[1], learner_type="X", reg_lambda=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    h_learner_x.fit(X_train, y_train, t_train, stage1_y0_prediction, stage1_y1_prediction, stage1_p_prediction)
    cate_pred_train = h_learner_x.predict(X_train)
    cate_pred_test = h_learner_x.predict(X_test)
    pehe_train, sqrt_pehe_train, pehe_test, sqrt_pehe_test = cate_evaluations(cate_pred_train, cate_pred_test, mu0_train, mu1_train, mu0_test, mu1_test)
    results.append({
        "folder": folder,
        "data_id": k,
        "model": "H_learner (X)",
        "pehe_train": pehe_train,
        "sqrt_pehe_train": sqrt_pehe_train,
        "pehe_test": pehe_test,
        "sqrt_pehe_test": sqrt_pehe_test,
    })

    h_learner_dr = HLearner(input_dim=X_train.shape[1], learner_type="DR", reg_lambda=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    h_learner_dr.fit(X_train, y_train, t_train, stage1_y0_prediction, stage1_y1_prediction, stage1_p_prediction)
    cate_pred_train = h_learner_dr.predict(X_train)
    cate_pred_test = h_learner_dr.predict(X_test)
    pehe_train, sqrt_pehe_train, pehe_test, sqrt_pehe_test = cate_evaluations(cate_pred_train, cate_pred_test, mu0_train, mu1_train, mu0_test, mu1_test)
    results.append({
        "folder": folder,
        "data_id": k,
        "model": "H_learner (DR)",
        "pehe_train": pehe_train,
        "sqrt_pehe_train": sqrt_pehe_train,
        "pehe_test": pehe_test,
        "sqrt_pehe_test": sqrt_pehe_test,
    })

    results_df = pd.DataFrame(results)
    os.makedirs("./results", exist_ok=True)
    results_df.to_csv(f"./results/acic_results_{folder}_{k}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=int, required=True, help="Index folder to ACIC datasets")
    parser.add_argument("--k", type=int, required=True, help="Index k to ACIC datasets")
    args = parser.parse_args()
    main(args.folder, args.k)