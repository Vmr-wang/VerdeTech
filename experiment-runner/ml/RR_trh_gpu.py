import sys
import time
import numpy as np
import torch
from torch import nn
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def _load_dataset(dataset_name: str):
    """统一数据加载逻辑（与前面保持一致）"""
    if dataset_name == 'iris':
        data = load_iris()
        mask = data.target < 2
        X, y = data.data[mask], data.target[mask]
        display_name = "Iris"

    elif dataset_name == 'wine':
        data = load_wine()
        mask = data.target < 2
        X, y = data.data[mask], data.target[mask]
        display_name = "Wine"

    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        X, y = data.data, data.target
        display_name = "Breast Cancer"

    elif dataset_name == 'digits':
        data = load_digits()
        mask = data.target < 2
        X, y = data.data[mask], data.target[mask]
        display_name = "Digits"

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X.astype(np.float32), y.astype(np.float32), display_name


class RidgeRegressionTorch(nn.Module):
    """线性模型（Ridge Regression）"""
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=True)

    def forward(self, x):
        return self.linear(x).squeeze(1)


def run_ridge_regression_torch(dataset_name: str, random_state: int = 42,
                               epochs: int = 300, lr: float = 1e-2,
                               weight_decay: float = 1e-2):
    """运行 PyTorch Ridge Regression（GPU）"""
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    X, y, display_name = _load_dataset(dataset_name)
    n_features = X.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    X_test_t  = torch.from_numpy(X_test).to(device)
    y_test_t  = torch.from_numpy(y_test).to(device)

    model = RidgeRegressionTorch(n_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    start = time.time()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        preds = model(X_train_t)
        loss = criterion(preds, y_train_t)
        loss.backward()
        optimizer.step()
    runtime = time.time() - start

    # 评估性能：使用 R^2 作为 accuracy proxy
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t)
        mse = criterion(preds, y_test_t).item()
        ss_tot = torch.sum((y_test_t - y_test_t.mean()) ** 2).item()
        ss_res = torch.sum((y_test_t - preds) ** 2).item()
        r2 = 1 - ss_res / ss_tot

    print(f"DATASET_NAME: {display_name}")
    print(f"ACTUAL_SIZE: {len(y)}")
    print(f"N_FEATURES: {n_features}")
    print(f"DEVICE: {device}")
    print(f"RUNTIME: {runtime:.6f}")
    print(f"R2_SCORE: {r2:.6f}")
    print(f"MSE: {mse:.6f}")

    return {
        "dataset_name": display_name,
        "actual_size": int(len(y)),
        "n_features": int(n_features),
        "runtime": float(runtime),
        "r2": float(r2),
        "mse": float(mse),
        "device": str(device)
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Missing dataset argument")
        print("Usage: python ml/Ridge_torch.py <dataset_name> [epochs] [lr] [weight_decay]")
        print("Available datasets: iris, wine, breast_cancer, digits")
        sys.exit(1)

    dataset_name = sys.argv[1]
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    lr = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-2
    weight_decay = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-2

    result = run_ridge_regression_torch(
        dataset_name=dataset_name, 
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay
    )

    print(f"\n=== Experiment Complete ===")
    print(f"Dataset: {result['dataset_name']} ({result['actual_size']} samples, {result['n_features']} features)")
    print(f"Device: {result['device']}")
    print(f"Runtime: {result['runtime']:.4f}s")
    print(f"R² Score: {result['r2']:.4f}")
    print(f"MSE: {result['mse']:.4f}")
