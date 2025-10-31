import os
import time
import random
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
from functools import lru_cache
from collections import Counter  
from qiskit import QuantumCircuit 
from qiskit_aer import AerSimulator


# --- 1. SETUP ---

app = Flask(__name__)
CORS(app)

ENVIRONMENT = os.getenv('SERVER_ENV', 'HUGGINGFACE')

# --- 2. DATA & MODEL LOADING ---

mnist_df = None
X_full = None
y_full = None

if ENVIRONMENT != 'PYTHONANYWHERE':
    try:
        print("INFO: Loading full dataset for real-time compilation.")
        mnist_df = pd.read_csv('data/mnist_train.csv')
        y_full = mnist_df.iloc[:, 0].values
        X_full = mnist_df.iloc[:, 1:].values  
        
    except FileNotFoundError:
        print("FATAL: mnist_train.csv not found. Real-time mode will fail.")
        mnist_df = None
else:
    print("INFO: Running in pre-compiled mode for PythonAnywhere.")

# --- 3. CLASSIFICATION LOGIC ---

DEFAULT_CONFIGS = {
    "knn_classical": {"training_size": 250, "pca_components": 16, "k": 5},
    "qknn_sim": {"training_size": 250, "pca_components": 16, "k": 5, "shots": 1024}
}

def run_knn_classical(X_train_pca, y_train, X_test_pca, config):
    k = config.get('k', 3)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_pca, y_train)
    prediction = knn.predict(X_test_pca)
    return str(prediction[0])

def get_quantum_distance(vec_alpha, vec_vj, shots=1024):

    # 1. Get vector dimension and calculate qubits needed
    dim = len(vec_alpha)
    num_qubits_per_state = int(np.ceil(np.log2(dim)))
    target_dim = 2**num_qubits_per_state

    # 2. Pad vectors with zeros to reach the target dimension
    if dim != target_dim:
        padding = np.zeros(target_dim - dim)
        vec_alpha = np.concatenate((vec_alpha, padding))
        vec_vj = np.concatenate((vec_vj, padding))

    # 3. Normalize the vectors for state preparation
    norm_alpha = np.linalg.norm(vec_alpha)
    norm_vj = np.linalg.norm(vec_vj)

    if norm_alpha == 0 or norm_vj == 0:
        return 1.0 

    vec_alpha = vec_alpha / norm_alpha
    vec_vj = vec_vj / norm_vj

    # 4. Build the Quantum Circuit (Swap Test)
    qc = QuantumCircuit(1 + 2 * num_qubits_per_state, 1)
    alpha_qubits = list(range(1, num_qubits_per_state + 1))
    vj_qubits = list(range(num_qubits_per_state + 1, 1 + 2 * num_qubits_per_state))
    
    qc.initialize(vec_alpha, alpha_qubits)
    qc.initialize(vec_vj, vj_qubits)
    
    ancilla_qubit = 0
    qc.h(ancilla_qubit)
    for i in range(num_qubits_per_state):
        qc.cswap(ancilla_qubit, alpha_qubits[i], vj_qubits[i])
    qc.h(ancilla_qubit)
    
    qc.measure(ancilla_qubit, 0)
    
    # 5. Run the simulator
    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()

    return counts.get('1', 0) / shots

def run_qknn_simulated(X_train_pca, y_train, X_test_pca, config):

    print(f"INFO: Simulating QkNN with config: {config}")
    
    # 1. Get parameters from config
    k = config.get('k', 3)
    shots = config.get('shots', 1024)

    # 2. Get the user's test vector (it's already 1-row)
    user_vector_pca = X_test_pca.flatten()
    if np.linalg.norm(user_vector_pca) == 0:
        print("WARN: User vector has zero norm. Returning '0'.")
        return "0" 

    # 3. Run the QkNN comparison
    distances = []
    for i in range(X_train_pca.shape[0]):
        train_vector = X_train_pca[i]
        
        distance = get_quantum_distance(user_vector_pca, train_vector, shots=shots)
        distances.append({'label': y_train[i], 'distance': distance})
    
    if not distances:
        print("ERROR: No valid training vectors found.")
        return "ERR"

    # 4. Find the k-nearest neighbors and predict
    sorted_neighbors = sorted(distances, key=lambda x: x['distance'])
    k_nearest_labels = [n['label'] for n in sorted_neighbors[:k]]
    
    # 5. Find the most common label (the prediction)
    prediction = Counter(k_nearest_labels).most_common(1)[0][0]
    
    return str(prediction)

# --- 4. MODEL & DATA HANDLERS (Hybrid Strategy) ---

_precompiled_cache = {}

def get_models_precompiled(config):
    train_size = config.get('training_size')
    pca_components = config.get('pca_components')
    if train_size is None or pca_components is None:
        raise ValueError("Missing 'training_size' or 'pca_components' in config")
    key = (int(train_size), int(pca_components))
    if key in _precompiled_cache:
        return _precompiled_cache[key]

    base_path = f"models/train{train_size}_pca{pca_components}"
    print(f"INFO: Loading pre-compiled models from: {base_path}")
    try:
        scaler = joblib.load(f"{base_path}_scaler.pkl")
        pca = joblib.load(f"{base_path}_pca.pkl")
        X_train_pca = np.load(f"{base_path}_xtrain.npy")
        y_train = np.load(f"{base_path}_ytrain.npy")
    except FileNotFoundError:
        raise ValueError(f"No pre-compiled model found for config: {config}")

    _precompiled_cache[key] = (scaler, pca, X_train_pca, y_train)
    return _precompiled_cache[key]

@lru_cache(maxsize=8)
def _compile_realtime(train_size: int, pca_components: int):
    if mnist_df is None:
        raise RuntimeError("Dataset not loaded; real-time compilation unavailable.")
    print(f"INFO: Compiling models in real-time: train={train_size}, pca={pca_components}")

    X_train_slice, _, y_train_slice, _ = train_test_split(
        X_full, 
        y_full, 
        train_size=train_size, 
        stratify=y_full,
        random_state=42 
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_slice)

    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    return scaler, pca, X_train_pca, y_train_slice

def get_models_realtime(config):
    train_size = config.get('training_size', 250)
    pca_components = config.get('pca_components', 16)
    return _compile_realtime(int(train_size), int(pca_components))

# --- 5. API ENDPOINTS ---

@app.route('/status', methods=['GET', 'HEAD'])
def get_status():
    return jsonify({"status": "online", "environment": ENVIRONMENT}), 200

@app.route('/configurations', methods=['GET'])
def get_configurations():
    """Returns available precompiled configurations for PythonAnywhere"""
    if ENVIRONMENT == 'PYTHONANYWHERE':
        available_configs = {
            "knn_classical": [
                {"training_size": 250, "pca_components": 16, "k": 5}
            ],
            "qknn_sim": [
                {"training_size": 250, "pca_components": 16, "k": 5, "shots": 1024}
            ]
        }
        return jsonify(available_configs), 200
    else:
        
        return jsonify({"message": "Real-time compilation enabled"}), 200

@app.route('/classify', methods=['POST'])
def classify_image():
    data = request.get_json(silent=True, force=True)
    if not data or 'pixels' not in data:
        return jsonify({"error": "Invalid request payload: missing 'pixels'"}), 400

    algorithms_to_run = data.get('algorithms')
    if not algorithms_to_run:
        name = data.get('algorithm', 'knn_classical')
        config = data.get('config') or DEFAULT_CONFIGS.get(name, DEFAULT_CONFIGS['knn_classical'])
        algorithms_to_run = [{"name": name, "config": config}]

    try:
        pixels = data['pixels']
        if len(pixels) != 784:
            return jsonify({"error": f"'pixels' must be an array of length 784, got {len(pixels)}"}), 400
        X_test = np.array(pixels, dtype=np.float32).reshape(1, -1)
    except Exception as e:
        return jsonify({"error": f"Invalid 'pixels' array: {str(e)}"}), 400

    predictions = {}
    timings = {}

    for algo in algorithms_to_run:
        try:
            name = algo.get('name')
            if not name:
                raise ValueError("Algorithm entry missing 'name'")
            config = algo.get('config') or DEFAULT_CONFIGS.get(name, DEFAULT_CONFIGS['knn_classical'])

            start_time = time.time()

            if ENVIRONMENT == 'PYTHONANYWHERE':
                scaler, pca, X_train_pca, y_train = get_models_precompiled(config)
            else:
                scaler, pca, X_train_pca, y_train = get_models_realtime(config)

            X_test_scaled = scaler.transform(X_test)
            X_test_pca = pca.transform(X_test_scaled)

            if name == 'knn_classical':
                pred = run_knn_classical(X_train_pca, y_train, X_test_pca, config)
            elif name == 'qknn_sim':
                pred = run_qknn_simulated(X_train_pca, y_train, X_test_pca, config)
            else:
                pred = "Algorithm not found"

            predictions[name] = pred
            timings[name] = round((time.time() - start_time) * 1000)

        except Exception as e:
            print(f"ERROR processing algorithm '{name}': {e}")
            predictions[name or 'unknown'] = "Error"
            timings[name or 'unknown'] = -1

    return jsonify({"predictions": predictions, "timings": timings}), 200

@app.route('/get_random_mnist_image', methods=['GET'])
def get_random_mnist_image():
    if X_full is None:
        return jsonify({"error": "Dataset not loaded"}), 500

    digit_filter = request.args.get('digit', type=int)
    
    if digit_filter is not None and 0 <= digit_filter <= 9:
        
        matching_indices = [i for i, label in enumerate(y_full) if label == digit_filter]
        
        if not matching_indices:
            return jsonify({"error": f"No samples found for digit {digit_filter}"}), 404
        
        idx = random.choice(matching_indices)
    else:
        
        idx = random.randint(0, len(X_full) - 1)
    
    image_pixels = X_full[idx].tolist()
    label = int(y_full[idx])

    return jsonify({"pixels": image_pixels, "label": label}), 200

# --- 6. RUN THE APP ---

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)