import sys
import time
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, load_digits
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode


gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise RuntimeError("No GPU detected. Please enable CUDA or install the GPU version of TensorFlow.")
else:
    gpu_name = tf.config.experimental.get_device_details(gpus[0]).get('device_name', 'Unknown GPU')
    print(f"✅ GPU detected: {gpu_name}")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"CUDA enabled: True\n")


def _load_dataset(dataset_name: str):
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
    return X.astype(np.float32), y.astype(np.int32), display_name


@tf.function
def _assign_clusters(X, centroids):
    distances = tf.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    return tf.argmin(distances, axis=1)


@tf.function
def _update_centroids(X, labels, k):
    new_centroids = []
    for i in tf.range(k):
        cluster_points = tf.boolean_mask(X, tf.equal(labels, i))
        mean = tf.cond(
            tf.shape(cluster_points)[0] > 0,
            lambda: tf.reduce_mean(cluster_points, axis=0),
            lambda: tf.zeros(tf.shape(X)[1])
        )
        new_centroids.append(mean)
    return tf.stack(new_centroids)


def run_kmeans_tf(dataset_name: str, random_state: int = 42,
                  n_clusters: int = 2, n_iters: int = 100):
    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    X, y, display_name = _load_dataset(dataset_name)
    n_samples, n_features = X.shape


    with tf.device('/GPU:0'):
        X_tensor = tf.constant(X, dtype=tf.float32)
        indices = np.random.choice(n_samples, n_clusters, replace=False)
        centroids = tf.Variable(X_tensor[indices])

        start = time.time()
        for _ in range(n_iters):
            labels = _assign_clusters(X_tensor, centroids)
            new_centroids = _update_centroids(X_tensor, labels, n_clusters)
            centroids.assign(new_centroids)
        runtime = time.time() - start

        labels = _assign_clusters(X_tensor, centroids).numpy()

    if len(np.unique(y)) == n_clusters:
        new_labels = np.zeros_like(labels)
        for i in range(n_clusters):
            mask = labels == i
            if np.any(mask):
                new_labels[mask] = mode(y[mask], keepdims=False).mode
        acc = np.mean(new_labels == y)
    else:
        acc = np.nan

    print(f"DATASET_NAME: {display_name}")
    print(f"ACTUAL_SIZE: {n_samples}")
    print(f"N_FEATURES: {n_features}")
    print(f"RUNTIME: {runtime:.6f}")
    print(f"ACCURACY: {acc:.6f}")

    return {
        "dataset_name": display_name,
        "actual_size": n_samples,
        "n_features": n_features,
        "runtime": float(runtime),
        "accuracy": float(acc) if not np.isnan(acc) else None,
    }


if __name__ == "__main__":
    '''
    if len(sys.argv) < 2:
        print("Error: Missing dataset argument")
        print("Usage: python ml/KMeans_tf.py <dataset_name>")
        print("Available datasets: iris, wine, breast_cancer, digits")
        sys.exit(1)
    '''

    dataset_name = sys.argv[1]   
    result = run_kmeans_tf(dataset_name=dataset_name)

    print(f"\n=== Experiment Complete ===")
    print(f"Dataset: {result['dataset_name']} ({result['actual_size']} samples, {result['n_features']} features)")
    print(f"Device: {result['device']}")
    print(f"Runtime: {result['runtime']:.4f}s")
    print(f"Accuracy: {result['accuracy']:.4f}")
