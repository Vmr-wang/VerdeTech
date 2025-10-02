import sys
sys.path.insert(0, '/home/abigale/anaconda3/envs/experiment-runner/lib/python3.10/site-packages')
import xgboost as xgb
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

def run_logistic_regression_xgboost(dataset_name: str, random_state: int = 42):
    """
    运行 XGBoost Logistic Regression 并返回 runtime 和 accuracy
    
    参数:
        dataset_name: 数据集名称 ('iris', 'wine', 'breast_cancer', 'digits')
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
    
    # 划分训练集 / 测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # 转换成 DMatrix 格式
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "seed": random_state
    }
    
    # 计时 & 训练
    start = time.time()
    model = xgb.train(params, dtrain, num_boost_round=50)
    runtime = time.time() - start
    
    # 预测 & 计算准确率
    preds = model.predict(dtest)
    acc = (preds.round() == y_test).mean()
    
    # 打印详细信息（EnergiBridge 会捕获）
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
    # 从命令行读取参数
    if len(sys.argv) < 2:
        print("Error: Missing dataset argument")
        print("Usage: python ml/LR_xgb.py <dataset_name>")
        print("Available datasets: iris, wine, breast_cancer, digits")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    
    # 运行实验
    result = run_logistic_regression_xgboost(dataset_name=dataset_name)
    
    # 打印最终汇总
    print(f"\n=== Experiment Complete ===")
    print(f"Dataset: {result['dataset_name']} ({result['actual_size']} samples, {result['n_features']} features)")
    print(f"Runtime: {result['runtime']:.4f}s")
    print(f"Accuracy: {result['accuracy']:.4f}")