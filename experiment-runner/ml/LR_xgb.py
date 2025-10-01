import sys
sys.path.insert(0, '/home/abigale/anaconda3/envs/experiment-runner/lib/python3.10/site-packages')
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

def run_logistic_regression_xgboost(dataset_size: int, n_features: int = 20, random_state: int = 42):
    """运行 XGBoost Logistic Regression 并返回 runtime 和 accuracy"""
    # 生成数据
    X, y = make_classification(n_samples=dataset_size, n_features=n_features, random_state=random_state)

    # 划分训练集 / 测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # 转换成 DMatrix 格式
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {"objective": "binary:logistic", "eval_metric": "logloss"}

    # 计时 & 训练
    start = time.time()
    model = xgb.train(params, dtrain, num_boost_round=50)
    runtime = time.time() - start

    # 预测 & 计算准确率
    preds = model.predict(dtest)
    acc = (preds.round() == y_test).mean()

    return {"runtime": runtime, "accuracy": float(acc)}
