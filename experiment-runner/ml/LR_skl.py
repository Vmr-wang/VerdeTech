import sys
sys.path.insert(0, '/home/abigale/anaconda3/envs/experiment-runner/lib/python3.10/site-packages')
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

def run_logistic_regression_sklearn(dataset_size: int, n_features: int = 20, random_state: int = 42):
    """运行 sklearn Logistic Regression 并返回 runtime 和 accuracy"""
    # 生成数据
    X, y = make_classification(n_samples=dataset_size, n_features=n_features, random_state=random_state)

    # 划分训练集 / 测试集 (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # 定义模型
    clf = LogisticRegression(max_iter=1000, solver="lbfgs")

    # 计时并训练
    start = time.time()
    clf.fit(X_train, y_train)
    runtime = time.time() - start

    # 测试集准确率
    acc = clf.score(X_test, y_test)
    return {"runtime": runtime, "accuracy": acc}
