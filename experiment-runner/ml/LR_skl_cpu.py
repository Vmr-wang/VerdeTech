import sys
import os

# If there is a relative site-packages directory in the project, add it to sys.path; otherwise, don't modify sys.path
site_pkgs = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'site-packages'))
if os.path.isdir(site_pkgs):
    sys.path.insert(0, site_pkgs)

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

def run_logistic_regression_sklearn(dataset_name: str, random_state: int = 42):
    """
    Run sklearn Logistic Regression and return runtime and accuracy
    
    Parameters:
        dataset_name: name of dataset ('iris', 'wine', 'breast_cancer', 'digits')
    """
    
    # Load dataset based on dataset name
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
    
    # Define model
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    
    # Time tracking & training
    start = time.time()
    model.fit(X_train, y_train)
    runtime = time.time() - start
    
    # Predict & calculate accuracy
    acc = model.score(X_test, y_test)
    
    # Print detailed information (captured by EnergiBridge)
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
    # Read parameters from command line
    if len(sys.argv) < 2:
        print("Error: Missing dataset argument")
        print("Usage: python ml/LR_skl.py <dataset_name>")
        print("Available datasets: iris, wine, breast_cancer, digits")
        sys.exit(1)
    
    dataset_name = sys.argv[1]
    
    # Run experiment
    result = run_logistic_regression_sklearn(dataset_name=dataset_name)
    
    # Print final summary
    print(f"\n=== Experiment Complete ===")
    print(f"Dataset: {result['dataset_name']} ({result['actual_size']} samples, {result['n_features']} features)")
    print(f"Runtime: {result['runtime']:.4f}s")
    print(f"Accuracy: {result['accuracy']:.4f}")