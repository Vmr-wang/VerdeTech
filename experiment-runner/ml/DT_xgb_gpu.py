import sys
import time
import numpy as np
from xgboost import XGBClassifier, core
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



try:
    _test = XGBClassifier(tree_method="hist", device="cuda", n_estimators=1)
    _test.fit(np.zeros((2, 2)), np.array([0, 1]))
    print("GPU acceleration detected for XGBoost 2.x (device='cuda')\n")
except core.XGBoostError as e:
    raise RuntimeError(
        "XGBoost GPU not available. Please install CUDA-enabled XGBoost "
        "(e.g. `pip install xgboost>=2.0.0`) and ensure CUDA is configured.\n"
        f"Details: {str(e)}"
    )


def run_decision_tree_xgb_gpu(dataset_name: str, random_state: int = 42):
    """
    运行 XGBoost（GPU）版 Decision Tree Classifier
    返回 runtime 和 accuracy
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

    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # 定义 XGBoost 模型（GPU 版）
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=random_state,
        eval_metric='logloss',
        tree_method='hist',
        device='cuda',
    )

    # 计时训练
    start = time.time()
    model.fit(X_train, y_train)
    runtime = time.time() - start

    # 准确率
    acc = model.score(X_test, y_test)

    # 打印结果（EnergiBridge 捕获）
    print(f"DATASET_NAME: {display_name}")
    print(f"ACTUAL_SIZE: {len(y)}")
    print(f"N_FEATURES: {X.shape[1]}")
    print(f"DEVICE: GPU")
    print(f"RUNTIME: {runtime:.6f}")
    print(f"ACCURACY: {acc:.6f}")

    return {
        "dataset_name": display_name,
        "actual_size": len(y),
        "n_features": X.shape[1],
        "runtime": runtime,
        "accuracy": float(acc),
        "device": "GPU"
    }


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Error: Missing dataset argument")
        print("Usage: python ml/DT_xgb_gpu.py <dataset_name>")
        print("Available datasets: iris, wine, breast_cancer, digits")
        sys.exit(1)

    dataset_name = sys.argv[1]
    result = run_decision_tree_xgb_gpu(dataset_name=dataset_name)

    print(f"\n=== Experiment Complete ===")
    print(f"Dataset: {result['dataset_name']} ({result['actual_size']} samples, {result['n_features']} features)")
    print(f"Device: {result['device']}")
    print(f"Runtime: {result['runtime']:.4f}s")
    print(f"Accuracy: {result['accuracy']:.4f}")
