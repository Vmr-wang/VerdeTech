import sys
import time
from sklearnex import patch_sklearn
patch_sklearn()  # 必须在 sklearn 导入前调用！

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def run_logistic_regression_sklearn_intelex(dataset_name: str, random_state: int = 42):
    """
    运行 scikit-learn-intelex 加速的 Logistic Regression
    返回 runtime 和 accuracy
    """
    # 根据数据集名称加载数据
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
        X, y, test_size=0.2, random_state=random_state
    )

    # 定义模型（启用 Intel oneDAL 加速）
    model = LogisticRegression(random_state=random_state, max_iter=1000)

    # 训练计时
    start = time.time()
    model.fit(X_train, y_train)
    runtime = time.time() - start

    # 预测与准确率
    acc = model.score(X_test, y_test)

    # 打印输出（EnergiBridge捕获）
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
        print("Usage: python ml/LR_skl_intelex.py <dataset_name>")
        print("Available datasets: iris, wine, breast_cancer, digits")
        sys.exit(1)

    dataset_name = sys.argv[1]
    result = run_logistic_regression_sklearn_intelex(dataset_name=dataset_name)

    print(f"\n=== Experiment Complete ===")
    print(f"Dataset: {result['dataset_name']} ({result['actual_size']} samples, {result['n_features']} features)")
    print(f"Runtime: {result['runtime']:.4f}s")
    print(f"Accuracy: {result['accuracy']:.4f}")
