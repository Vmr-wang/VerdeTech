import sys
sys.path.insert(0, '/home/abigale/anaconda3/envs/experiment-runner/lib/python3.10/site-packages')
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

class LogisticRegressionModel(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def run_logistic_regression_pytorch(dataset_size: int, n_features: int = 20, random_state: int = 42):
    """运行 PyTorch Logistic Regression 并返回 runtime 和 accuracy"""
    # 生成数据
    X, y = make_classification(n_samples=dataset_size, n_features=n_features, random_state=random_state)

    # 划分训练集 / 测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # 转成 Tensor
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # 定义模型
    model = LogisticRegressionModel(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 计时 & 训练
    start = time.time()
    for epoch in range(100):  # 简单训练 100 个 epoch
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    runtime = time.time() - start

    # 在测试集上预测 & 准确率
    with torch.no_grad():
        preds = (model(X_test) >= 0.5).float()
        acc = (preds == y_test).float().mean().item()

    return {"runtime": runtime, "accuracy": acc}
