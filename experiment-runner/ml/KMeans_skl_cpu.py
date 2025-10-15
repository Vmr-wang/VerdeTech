import sys
import time
import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import mode

def _load_dataset(dataset_name: str):
    """与逻辑回归一致的数据加载逻辑"""
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
    return X, y, display_name


def run_kmeans_sklearn(dataset_name: str, random_state: int = 42, n_clusters: int = 2):
    """
    运行 scikit-learn 的 K-Means 聚类
    返回 runtime 和 approximate accuracy
    """
    X, y, display_name = _load_dataset(dataset_name)

    # 初始化模型
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)

    # 计时训练
    start = time.time()
    model.fit(X)
    runtime = time.time() - start

    # 计算聚类的“伪准确率”
    labels = model.labels_

    # 尝试对齐聚类标签与真实标签（仅用于报告，不影响无监督算法的定义）
    if len(np.unique(y)) == n_clusters:
        new_labels = np.zeros_like(labels)
        for i in range(n_clusters):
            mask = labels == i
            if np.any(mask):
                new_labels[mask] = mode(y[mask], keepdims=False).mode
        acc = np.mean(new_labels == y)
    else:
        acc = np.nan  # 对非二分类数据无法对齐标签

    # 打印结果（EnergiBridge 捕获）
    print(f"DATASET_NAME: {display_name}")
    print(f"ACTUAL_SIZE: {len(y)}")
    print(f"N_FEATURES: {X.shape[1]}")
    print(f"RUNTIME: {runtime:.6f}")
    print(f"ACCURACY: {acc:.6f}")

    return {
        "dataset_name": display_name,
        "actual_size": len(y),
        "n_features": X.shape[1],
        "runtime": runtime,
        "accuracy": float(acc) if not np.isnan(acc) else None
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Missing dataset argument")
        print("Usage: python ml/KMeans_skl.py <dataset_name>")
        print("Available datasets: iris, wine, breast_cancer, digits")
        sys.exit(1)

    dataset_name = sys.argv[1]
    result = run_kmeans_sklearn(dataset_name=dataset_name)

    print(f"\n=== Experiment Complete ===")
    print(f"Dataset: {result['dataset_name']} ({result['actual_size']} samples, {result['n_features']} features)")
    print(f"Runtime: {result['runtime']:.4f}s")
    print(f"Accuracy: {result['accuracy']:.4f}")
