import sys
import time
from sklearnex import patch_sklearn
patch_sklearn()  # 启用 Intel oneDAL 加速，必须在 sklearn 导入前

from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score


def run_ridge_regression_intelex(dataset_name: str, random_state: int = 42, alpha: float = 1.0):
    """
    运行 scikit-learn-intelex 加速版 Ridge Regression
    返回 runtime、MSE、R²
    """
    # 加载数据
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

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # 定义模型（启用 oneDAL 加速）
    model = Ridge(alpha=alpha, random_state=random_state)

    # 计时训练
    start = time.time()
    model.fit(X_train, y_train)
    runtime = time.time() - start

    # 预测与评估
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 打印结果（EnergiBridge 捕获）
    print(f"DATASET_NAME: {display_name}")
    print(f"ACTUAL_SIZE: {len(y)}")
    print(f"N_FEATURES: {X.shape[1]}")
    print(f"DEVICE: CPU (Intel oneDAL)")
    print(f"RUNTIME: {runtime:.6f}")
    print(f"R2_SCORE: {r2:.6f}")
    print(f"MSE: {mse:.6f}")

    return {
        "dataset_name": display_name,
        "actual_size": len(y),
        "n_features": X.shape[1],
        "runtime": runtime,
        "r2": float(r2),
        "mse": float(mse),
        "device": "CPU (Intel oneDAL)"
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Missing dataset argument")
        print("Usage: python ml/Ridge_intelex.py <dataset_name> [alpha]")
        print("Available datasets: iris, wine, breast_cancer, digits")
        sys.exit(1)

    dataset_name = sys.argv[1]
    alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    result = run_ridge_regression_intelex(dataset_name=dataset_name, alpha=alpha)

    print(f"\n=== Experiment Complete ===")
    print(f"Dataset: {result['dataset_name']} ({result['actual_size']} samples, {result['n_features']} features)")
    print(f"Device: {result['device']}")
    print(f"Runtime: {result['runtime']:.4f}s")
    print(f"R² Score: {result['r2']:.4f}")
    print(f"MSE: {result['mse']:.4f}")
