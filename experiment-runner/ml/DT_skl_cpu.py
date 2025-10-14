import sys
import time
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def run_decision_tree_sklearn(dataset_name: str, random_state: int = 42, max_depth: int = None):
    """
    运行 scikit-learn 的 Decision Tree Classifier 并返回 runtime 和 accuracy
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

    # 标准化（树模型不是必须，但保持一致性）
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练/测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # 模型定义
    model = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth)

    # 训练计时
    start = time.time()
    model.fit(X_train, y_train)
    runtime = time.time() - start

    # 计算准确率
    acc = model.score(X_test, y_test)

    # 输出结果（EnergiBridge捕获）
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
        "accuracy": float(acc)
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Missing dataset argument")
        print("Usage: python ml/DT_skl.py <dataset_name> [max_depth]")
        print("Available datasets: iris, wine, breast_cancer, digits")
        sys.exit(1)

    dataset_name = sys.argv[1]
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else None

    result = run_decision_tree_sklearn(dataset_name=dataset_name, max_depth=max_depth)

    print(f"\n=== Experiment Complete ===")
    print(f"Dataset: {result['dataset_name']} ({result['actual_size']} samples, {result['n_features']} features)")
    print(f"Runtime: {result['runtime']:.4f}s")
    print(f"Accuracy: {result['accuracy']:.4f}")
