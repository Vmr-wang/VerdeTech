import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time


class LogisticRegressionModel(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))


def run_logistic_regression_pytorch(dataset_name: str, random_state: int = 42):
    """
    Run PyTorch Logistic Regression (GPU only)
    """
    # ✅ Force GPU usage
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not detected. Please enable CUDA or install a GPU-enabled PyTorch.")
    device = torch.device("cuda")
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}\n")

    # Load dataset
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
    
    # Data standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split training/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # ✅ Move data to GPU
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device).view(-1, 1)
    
    # Define model and move to GPU
    model = LogisticRegressionModel(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    # Timer & Training
    start = time.time()
    for epoch in range(100):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    runtime = time.time() - start
    
    # Predict on test set & calculate accuracy
    with torch.no_grad():
        preds = (model(X_test) >= 0.5).float()
        acc = (preds == y_test).float().mean().item()
    
    # Print details (will be captured by EnergiBridge)
    print(f"DATASET_NAME: {display_name}")
    print(f"ACTUAL_SIZE: {len(y)}")
    print(f"N_FEATURES: {X.shape[1]}")
    print(f"DEVICE: {device}")
    print(f"RUNTIME: {runtime:.6f}")
    print(f"ACCURACY: {acc:.6f}")
    
    return {
        "dataset_name": display_name,
        "actual_size": len(y),
        "n_features": X.shape[1],
        "runtime": runtime,
        "accuracy": acc,
        "device": str(device)
    }


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Error: Missing dataset argument")
        print("Usage: python ml/LR_trh.py <dataset_name>")
        print("Available datasets: iris, wine, breast_cancer, digits")
        sys.exit(1)
    
    dataset_name = sys.argv[1]

    if not torch.cuda.is_available():
        print("Error: GPU not detected or CUDA unavailable.")
        sys.exit(1)
   
    result = run_logistic_regression_pytorch(dataset_name=dataset_name)
   
    print(f"\n=== Experiment Complete ===")
    print(f"Dataset: {result['dataset_name']} ({result['actual_size']} samples, {result['n_features']} features)")
    print(f"Runtime: {result['runtime']:.4f}s")
    print(f"Accuracy: {result['accuracy']:.4f}")
