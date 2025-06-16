import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import random
import copy
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def reset_weights(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

class TLearner(nn.Module):
    def __init__(self, input_dim, lr=[0.0001, 0.0005, 0.001], epochs=1000, early_stopping=False, patience=20):
        super(TLearner, self).__init__()
        self.lr = lr
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.model_treated = self._build_model(input_dim)
        self.model_control = self._build_model(input_dim)

    def _build_model(self, input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        y1_hat = self.model_treated(x)
        y0_hat = self.model_control(x)
        return y1_hat, y0_hat

    def fit(self, X, y, t):
        set_seed(0)
        X_train_all, X_val_all, y_train_all, y_val_all, t_train, t_val = train_test_split(X, y, t, test_size=0.3, random_state=0)

        def _train_single_model(X_train, X_val, y_train, y_val):
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, pin_memory=True)

            overall_best_val_loss = float("inf")
            overall_best_model_state = None
            for lr in self.lr:
                set_seed(0)
                model = self._build_model(X_train.shape[1])
                criterion = nn.MSELoss()
                optimizer = optim.AdamW(model.parameters(), lr=lr)
                scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)

                best_val_loss = float("inf")
                best_model_state = None
                patience_counter = 0

                for epoch in range(self.epochs):
                    model.train()

                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor)

                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        best_model_state = copy.deepcopy(model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if self.early_stopping and patience_counter >= self.patience:
                            break

                    scheduler.step()
                
                if best_val_loss < overall_best_val_loss:
                    overall_best_val_loss = best_val_loss
                    overall_best_model_state = best_model_state

            final_model = self._build_model(X_train.shape[1])
            final_model.load_state_dict(overall_best_model_state)
            return final_model

        X_train_control = X_train_all[t_train == 0]
        y_train_control = y_train_all[t_train == 0]
        X_val_control = X_val_all[t_val == 0]
        y_val_control = y_val_all[t_val == 0]
        self.model_control = _train_single_model(X_train_control, X_val_control, y_train_control, y_val_control)

        X_train_treated = X_train_all[t_train == 1]
        y_train_treated = y_train_all[t_train == 1]
        X_val_treated = X_val_all[t_val == 1]
        y_val_treated = y_val_all[t_val == 1]
        self.model_treated = _train_single_model(X_train_treated, X_val_treated, y_train_treated, y_val_treated)

        return self

    def predict(self, x):
        self.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            y1_hat = self.model_treated(x_tensor)
            y0_hat = self.model_control(x_tensor)
        cate_pred = (y1_hat - y0_hat).cpu().numpy().flatten()
        return cate_pred
    

class TARNet(nn.Module):
    def __init__(self, input_dim, lr=[0.0001, 0.0005, 0.001], epochs=1000, early_stopping=False, patience=20):
        super(TARNet, self).__init__()
        self.lr = lr
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.rep = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU()
        )
        self.head_treated = nn.Sequential(
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 1)
        )

        self.head_control = nn.Sequential(
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.rep(x)
        return x, self.head_treated(x), self.head_control(x)

    def fit(self, X, y, t):
        set_seed(0)

        X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(X, y, t, test_size=0.3, random_state=0)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        t_train_tensor = torch.tensor(t_train, dtype=torch.float32).unsqueeze(1)
        t_val_tensor = torch.tensor(t_val, dtype=torch.float32).unsqueeze(1)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, t_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, pin_memory=True)

        overall_best_val_loss = float("inf")
        overall_best_model_state = None
        for lr in self.lr:
            set_seed(0)
            self.apply(reset_weights)
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(
                self.parameters(), lr=lr
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)
            patience_counter = 0
            best_val_loss = float("inf")
            best_model_state = None

            for epoch in range(self.epochs):
                self.train()
        
                for batch_X, batch_y, batch_t in train_loader:
                    optimizer.zero_grad()
                    _, y_hat_treated, y_hat_control = self.forward(batch_X)
                    y_hat_treated_t1 = y_hat_treated[batch_t == 1]
                    y_hat_control_t0 = y_hat_control[batch_t == 0]
                    y_treated_t1 = batch_y[batch_t == 1]
                    y_control_t0 = batch_y[batch_t == 0]
                    loss = criterion(y_hat_treated_t1, y_treated_t1) + criterion(y_hat_control_t0, y_control_t0)
                    loss.backward()
                    optimizer.step()

                self.eval()
                with torch.no_grad():
                    _, val_y_hat_treated, val_y_hat_control = self.forward(X_val_tensor)
                    val_y_hat_treated_t1 = val_y_hat_treated[t_val_tensor == 1]
                    val_y_hat_control_t0 = val_y_hat_control[t_val_tensor == 0]
                    val_y_t1 = y_val_tensor[t_val_tensor == 1]
                    val_y_t0 = y_val_tensor[t_val_tensor == 0]
                    val_loss = criterion(val_y_hat_treated_t1, val_y_t1) + criterion(val_y_hat_control_t0, val_y_t0) 

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_model_state = copy.deepcopy(self.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if self.early_stopping and patience_counter >= self.patience:
                        break
                    
                scheduler.step()

            if best_val_loss < overall_best_val_loss:
                overall_best_val_loss = best_val_loss
                overall_best_model_state = best_model_state

        self.load_state_dict(overall_best_model_state)
        return self

    def predict(self, x, return_po=False):
        self.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            _, y1_hat, y0_hat = self.forward(x_tensor)
        cate_pred = (y1_hat - y0_hat).cpu().numpy().flatten()
        if return_po:
            y1_hat = y1_hat.cpu().numpy().flatten()
            y0_hat = y0_hat.cpu().numpy().flatten()
            return cate_pred, y0_hat, y1_hat
        return cate_pred

class DirectLearner(nn.Module):
    def __init__(self, input_dim, learner_type="X", lr=[0.0001, 0.0005, 0.001], epochs=1000, early_stopping=False, patience=20):
        super(DirectLearner, self).__init__()
        self.lr = lr
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.learner_type = learner_type
        assert learner_type in ["X", "DR", "IPW"]
        self.rep = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU()
        )
        self.head = nn.Sequential(
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.rep(x)
        return self.head(x)

    def fit(self, X, y, t, stage1_y0_prediction, stage1_y1_prediction, stage1_p_prediction):
        set_seed(0)

        if self.learner_type == "X":
            pseudo_outcome = t * (y - stage1_y0_prediction) + (1 - t) * (stage1_y1_prediction - y)
        elif self.learner_type == "DR":
            pseudo_outcome = ((t - stage1_p_prediction) * (y - (t * stage1_y1_prediction + (1 - t) * stage1_y0_prediction))) / (stage1_p_prediction * (1 - stage1_p_prediction)) + (stage1_y1_prediction - stage1_y0_prediction)
        elif self.learner_type == "IPW":
            pseudo_outcome = (t * y) / stage1_p_prediction - ((1 - t) * y) / (1 - stage1_p_prediction)

        X_train, X_val, t_train, t_val, y_train, y_val, pseudo_outcome_train, _, _, stage1_y0_prediction_val, _, stage1_y1_prediction_val = train_test_split(
            X, t, y, pseudo_outcome, stage1_y0_prediction, stage1_y1_prediction, test_size=0.3, random_state=0)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        pseudo_outcome_train_tensor = torch.tensor(pseudo_outcome_train, dtype=torch.float32).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        t_val_tensor = torch.tensor(t_val, dtype=torch.float32).unsqueeze(1)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        stage1_y0_prediction_val_tensor = torch.tensor(stage1_y0_prediction_val, dtype=torch.float32).unsqueeze(1)
        stage1_y1_prediction_val_tensor = torch.tensor(stage1_y1_prediction_val, dtype=torch.float32).unsqueeze(1)
        train_dataset = TensorDataset(X_train_tensor, pseudo_outcome_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, pin_memory=True)

        overall_best_val_loss = float("inf")
        overall_best_model_state = None

        for lr in self.lr:
            set_seed(0)
            self.apply(reset_weights)
            criterion = nn.MSELoss()
            optimizer = optim.AdamW(self.parameters(), lr=lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)

            best_val_loss = float("inf")
            best_model_state = None
            patience_counter = 0

            for epoch in range(self.epochs):
                self.train()
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    train_cate_pred = self.forward(batch_X)
                    loss = criterion(train_cate_pred, batch_y)
                    loss.backward()
                    optimizer.step()

                self.eval()
                with torch.no_grad():
                    val_cate_pred = self(X_val_tensor)
                    val_cate = t_val_tensor * (y_val_tensor - stage1_y0_prediction_val_tensor) + \
                               (1 - t_val_tensor) * (stage1_y1_prediction_val_tensor - y_val_tensor)
                    val_loss = criterion(val_cate_pred, val_cate)

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_model_state = copy.deepcopy(self.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if self.early_stopping and patience_counter >= self.patience:
                        break

                scheduler.step()

            if best_val_loss < overall_best_val_loss:
                overall_best_val_loss = best_val_loss
                overall_best_model_state = best_model_state

        self.load_state_dict(overall_best_model_state)
        return self

    def predict(self, x):
        self.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            y_hat = self.forward(x_tensor)
        cate_pred = y_hat.cpu().numpy().flatten()
        return cate_pred

class PropensityModel(nn.Module):
    def __init__(self, input_dim, lr=[0.0001, 0.0005, 0.001], epochs=1000, early_stopping=False, patience=20):
        super(PropensityModel, self).__init__()
        self.lr = lr
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.model = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
    def fit(self, X, t):
        set_seed(0)

        X_train, X_val, t_train, t_val = train_test_split(X, t, test_size=0.3, random_state=0)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        t_train_tensor = torch.tensor(t_train, dtype=torch.float32).unsqueeze(1)
        t_val_tensor = torch.tensor(t_val, dtype=torch.float32).unsqueeze(1)
        train_dataset = TensorDataset(X_train_tensor, t_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, pin_memory=True)

        overall_best_val_loss = float("inf")
        overall_best_model_state = None

        for lr in self.lr:
            set_seed(0)
            self.apply(reset_weights)
            optimizer = optim.AdamW(self.parameters(), lr=lr)
            scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
            criterion = nn.BCELoss()

            best_val_loss = float("inf")
            best_model_state = None
            patience_counter = 0

            for epoch in range(self.epochs):
                self.train()

                for batch_X, batch_t in train_loader:
                    optimizer.zero_grad()
                    pred = self.forward(batch_X)
                    loss = criterion(pred, batch_t)
                    loss.backward()
                    optimizer.step()

                self.eval()
                with torch.no_grad():
                    val_pred = self.forward(X_val_tensor)
                    val_loss = criterion(val_pred, t_val_tensor)

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    best_model_state = copy.deepcopy(self.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if self.early_stopping and patience_counter >= self.patience:
                        break

                scheduler.step()

            if best_val_loss < overall_best_val_loss:
                overall_best_val_loss = best_val_loss
                overall_best_model_state = best_model_state

        self.load_state_dict(overall_best_model_state)
        return self
    
    def predict(self, x):
        self.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            p_pred = self.forward(x_tensor).squeeze().numpy()
        p_pred = np.clip(p_pred, 1e-3, 1 - 1e-3)
        return p_pred


class HLearner(nn.Module):
    def __init__(self, input_dim, reg_lambda, learner_type="X", lr=[0.0001, 0.0005, 0.001], epochs=1000, early_stopping=False, patience=20):
        super(HLearner, self).__init__()
        self.reg_lambda = reg_lambda
        self.lr = lr
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.learner_type = learner_type
        assert learner_type in ["X", "DR", "IPW"]
        self.rep = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU()
        )
        self.head_treated = nn.Sequential(
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 1)
        )
        self.head_control = nn.Sequential(
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        x = self.rep(x)
        return x, self.head_treated(x), self.head_control(x)

    def fit(self, X, y, t, stage1_y0_prediction, stage1_y1_prediction, stage1_p_pred):
        print("Fitting H-learner")
        set_seed(0)

        if self.learner_type == "X":
            pseudo_outcome = t * (y - stage1_y0_prediction) + (1 - t) * (stage1_y1_prediction - y)
        elif self.learner_type == "DR":
            pseudo_outcome = ((t - stage1_p_pred) * (y - (t * stage1_y1_prediction + (1 - t) * stage1_y0_prediction))) / \
                            (stage1_p_pred * (1 - stage1_p_pred)) + (stage1_y1_prediction - stage1_y0_prediction)
        elif self.learner_type == "IPW":
            pseudo_outcome = (t * y) / stage1_p_pred - ((1 - t) * y) / (1 - stage1_p_pred)

        X_train, X_val, y_train, y_val, t_train, t_val, pseudo_outcome_train, _, _, stage1_y0_prediction_val, _, stage1_y1_prediction_val = train_test_split(
            X, y, t, pseudo_outcome, stage1_y0_prediction, stage1_y1_prediction, test_size=0.3, random_state=0)

        x_score = self.validation(X_val, y_val, t_val)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        t_train_tensor = torch.tensor(t_train, dtype=torch.float32).unsqueeze(1)
        t_val_tensor = torch.tensor(t_val, dtype=torch.float32).unsqueeze(1)
        pseudo_outcome_train_tensor = torch.tensor(pseudo_outcome_train, dtype=torch.float32).unsqueeze(1)
        stage1_y0_prediction_val_tensor = torch.tensor(stage1_y0_prediction_val, dtype=torch.float32).unsqueeze(1)
        stage1_y1_prediction_val_tensor = torch.tensor(stage1_y1_prediction_val, dtype=torch.float32).unsqueeze(1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, t_train_tensor, pseudo_outcome_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, pin_memory=True)

        best_overall_val_loss = float("inf")
        best_lambda = -1
        self.best_models_per_lambda = {}

        for reg_lambda in tqdm(self.reg_lambda, desc="Training H-learner for different lambdas"):
            best_val_loss = float("inf")
            best_model_state = None

            for lr in self.lr:
                set_seed(0)
                self.apply(reset_weights)
                criterion = nn.MSELoss()
                optimizer = optim.AdamW(self.parameters(), lr=lr)
                scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0.00001)
                local_best_val_loss = float("inf")
                local_best_model_state = None
                patience_counter = 0

                for epoch in range(self.epochs):
                    self.train()
                    for batch_X, batch_y, batch_t, batch_pseudo_outcome in train_loader:
                        optimizer.zero_grad()
                        _, y_hat_treated, y_hat_control = self.forward(batch_X)
                        y_hat_treated_t1 = y_hat_treated[batch_t == 1]
                        y_hat_control_t0 = y_hat_control[batch_t == 0]
                        y_treated_t1 = batch_y[batch_t == 1]
                        y_control_t0 = batch_y[batch_t == 0]

                        loss = (1 - reg_lambda) * (
                            criterion(y_hat_treated_t1, y_treated_t1) + criterion(y_hat_control_t0, y_control_t0)
                        ) + reg_lambda * criterion(y_hat_treated - y_hat_control, batch_pseudo_outcome)

                        loss.backward()
                        optimizer.step()

                    self.eval()
                    with torch.no_grad():
                        _, val_y_hat_treated, val_y_hat_control = self.forward(X_val_tensor)
                        val_cate_pred = val_y_hat_treated - val_y_hat_control
                        val_cate = t_val_tensor * (y_val_tensor - stage1_y0_prediction_val_tensor) + \
                                (1 - t_val_tensor) * (stage1_y1_prediction_val_tensor - y_val_tensor)
                        val_loss = criterion(val_cate_pred, val_cate)

                    if val_loss.item() < local_best_val_loss:
                        local_best_val_loss = val_loss.item()
                        local_best_model_state = copy.deepcopy(self.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if self.early_stopping and patience_counter >= self.patience:
                            break

                    scheduler.step()

                if local_best_val_loss < best_val_loss:
                    best_val_loss = local_best_val_loss
                    best_model_state = local_best_model_state

            self.best_models_per_lambda[reg_lambda] = copy.deepcopy(best_model_state)
            self.load_state_dict(best_model_state)
            cate_pred_val = self.predict(X_val)
            x_score_loss = np.mean((cate_pred_val - x_score) ** 2)

            if x_score_loss < best_overall_val_loss:
                best_overall_val_loss = x_score_loss
                best_lambda = reg_lambda

        self.load_state_dict(self.best_models_per_lambda[best_lambda])
        self.best_lambda = best_lambda
        return self

    def predict(self, x):
        self.eval()
        x_tensor = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            _, y1_hat, y0_hat = self.forward(x_tensor)
        cate_pred = (y1_hat - y0_hat).cpu().numpy().flatten()
        return cate_pred

    def validation(self, X_val, y_val, t_val):
        tarnet_validation = TARNet(input_dim=X_val.shape[1], lr=self.lr, epochs=self.epochs, early_stopping=self.early_stopping, patience=self.patience)
        tarnet_validation = tarnet_validation.fit(X_val, y_val, t_val)
        _, val_y0_prediction, val_y1_prediction = tarnet_validation.predict(X_val, return_po=True)
        x_score = t_val * (y_val - val_y0_prediction) + (1 - t_val) * (val_y1_prediction - y_val)
        return x_score

    def load_model_for_lambda(self, reg_lambda):
        if not hasattr(self, "best_models_per_lambda"):
            raise ValueError("No models saved. Make sure to call `fit()` first.")
        if reg_lambda not in self.best_models_per_lambda:
            raise ValueError(f"No model saved for reg_lambda = {reg_lambda}")
        self.load_state_dict(self.best_models_per_lambda[reg_lambda])
