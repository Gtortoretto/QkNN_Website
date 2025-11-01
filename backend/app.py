import os
import time
import random
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import joblib
from functools import lru_cache, wraps
from collections import Counter  
from qiskit import QuantumCircuit 
from qiskit_aer import AerSimulator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# --- 1. SECURITY & SETUP ---

# Configure logging (no sensitive data in logs)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Secure CORS configuration - restrict to your actual frontend domains
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', 'https://gtortoretto.github.io').split(',')
CORS(app, origins=ALLOWED_ORIGINS, methods=['GET', 'POST', 'HEAD'], max_age=3600)

# Security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response

# Input validation limits (prevent resource exhaustion)
VALIDATION_LIMITS = {
    'training_size': {'min': 250, 'max': 60000},
    'pca_components': {'min': 2, 'max': 784},
    'k': {'min': 1, 'max': 20},
    'shots': {'min': 128, 'max': 8192},
    'pixel_array_size': 784,
    'max_algorithms_per_request': 3
}

ENVIRONMENT = os.getenv('SERVER_ENV', 'HUGGINGFACE')

# Simple rate limiting (in-memory, for production use Redis-based solution)
request_counts = {}
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX_REQUESTS = 30  # requests per window

def rate_limit():
    """Simple rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            current_time = time.time()
            
            # Clean old entries
            request_counts[client_ip] = [
                req_time for req_time in request_counts.get(client_ip, [])
                if current_time - req_time < RATE_LIMIT_WINDOW
            ]
            
            # Check rate limit
            if len(request_counts.get(client_ip, [])) >= RATE_LIMIT_MAX_REQUESTS:
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
            
            # Add current request
            if client_ip not in request_counts:
                request_counts[client_ip] = []
            request_counts[client_ip].append(current_time)
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_config(config, algorithm_name):
    """Validate and sanitize algorithm configuration"""
    validated = {}
    
    # Validate training_size
    train_size = config.get('training_size')
    if train_size is not None:
        try:
            train_size = int(train_size)
            if not (VALIDATION_LIMITS['training_size']['min'] <= train_size <= VALIDATION_LIMITS['training_size']['max']):
                raise ValueError(f"training_size must be between {VALIDATION_LIMITS['training_size']['min']} and {VALIDATION_LIMITS['training_size']['max']}")
            validated['training_size'] = train_size
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid training_size: {str(e)}")
    
    # Validate pca_components
    pca_comp = config.get('pca_components')
    if pca_comp is not None:
        try:
            pca_comp = int(pca_comp)
            if not (VALIDATION_LIMITS['pca_components']['min'] <= pca_comp <= VALIDATION_LIMITS['pca_components']['max']):
                raise ValueError(f"pca_components must be between {VALIDATION_LIMITS['pca_components']['min']} and {VALIDATION_LIMITS['pca_components']['max']}")
            validated['pca_components'] = pca_comp
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid pca_components: {str(e)}")
    
    # Validate k
    k_val = config.get('k')
    if k_val is not None:
        try:
            k_val = int(k_val)
            if not (VALIDATION_LIMITS['k']['min'] <= k_val <= VALIDATION_LIMITS['k']['max']):
                raise ValueError(f"k must be between {VALIDATION_LIMITS['k']['min']} and {VALIDATION_LIMITS['k']['max']}")
            validated['k'] = k_val
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid k: {str(e)}")
    
    # Validate shots (for QkNN only)
    if algorithm_name in ['qknn_sim', 'qknn_real']:
        shots = config.get('shots')
        if shots is not None:
            try:
                shots = int(shots)
                if not (VALIDATION_LIMITS['shots']['min'] <= shots <= VALIDATION_LIMITS['shots']['max']):
                    raise ValueError(f"shots must be between {VALIDATION_LIMITS['shots']['min']} and {VALIDATION_LIMITS['shots']['max']}")
                validated['shots'] = shots
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid shots: {str(e)}")
    
    return validated

# --- 2. DATA & MODEL LOADING ---

mnist_df = None
X_full = None
y_full = None

if ENVIRONMENT != 'PYTHONANYWHERE':
    try:
        logger.info("Loading full dataset for real-time compilation.")
        mnist_df = pd.read_csv('data/mnist_train.csv')
        y_full = mnist_df.iloc[:, 0].values
        X_full = mnist_df.iloc[:, 1:].values  
        
    except FileNotFoundError:
        logger.error("mnist_train.csv not found. Real-time mode will fail.")
        mnist_df = None
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        mnist_df = None
else:
    logger.info("Running in pre-compiled mode for PythonAnywhere.")

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

    logger.info(f"Simulating QkNN with config: {config}")
    
    # 1. Get parameters from config
    k = config.get('k', 3)
    shots = config.get('shots', 1024)

    # 2. Get the user's test vector (it's already 1-row)
    user_vector_pca = X_test_pca.flatten()
    if np.linalg.norm(user_vector_pca) == 0:
        logger.warning("User vector has zero norm. Returning '0'.")
        return "0" 

    # 3. Run the QkNN comparison
    distances = []
    for i in range(X_train_pca.shape[0]):
        train_vector = X_train_pca[i]
        
        distance = get_quantum_distance(user_vector_pca, train_vector, shots=shots)
        distances.append({'label': y_train[i], 'distance': distance})
    
    if not distances:
        logger.error("No valid training vectors found.")
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
    
    # Validate and sanitize inputs to prevent path traversal
    try:
        train_size = int(train_size)
        pca_components = int(pca_components)
    except (ValueError, TypeError):
        raise ValueError("Invalid configuration parameters")
    
    key = (train_size, pca_components)
    if key in _precompiled_cache:
        return _precompiled_cache[key]

    # Secure path construction - prevent directory traversal
    models_dir = Path('models').resolve()
    base_name = f"train{train_size}_pca{pca_components}"
    
    logger.info(f"Loading pre-compiled models: {base_name}")
    
    try:
        scaler_path = (models_dir / f"{base_name}_scaler.pkl").resolve()
        pca_path = (models_dir / f"{base_name}_pca.pkl").resolve()
        xtrain_path = (models_dir / f"{base_name}_xtrain.npy").resolve()
        ytrain_path = (models_dir / f"{base_name}_ytrain.npy").resolve()
        
        # Verify all paths are within models directory (prevent path traversal)
        for path in [scaler_path, pca_path, xtrain_path, ytrain_path]:
            if not str(path).startswith(str(models_dir)):
                logger.error(f"Path traversal attempt detected: {path}")
                raise ValueError("Invalid model path")
        
        scaler = joblib.load(str(scaler_path))
        pca = joblib.load(str(pca_path))
        X_train_pca = np.load(str(xtrain_path))
        y_train = np.load(str(ytrain_path))
        
    except FileNotFoundError:
        logger.error(f"Pre-compiled model not found: {base_name}")
        raise ValueError("Model configuration not available")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise ValueError("Failed to load model")

    _precompiled_cache[key] = (scaler, pca, X_train_pca, y_train)
    return _precompiled_cache[key]

@lru_cache(maxsize=8)
def _compile_realtime(train_size: int, pca_components: int):
    if mnist_df is None:
        raise RuntimeError("Dataset not loaded; real-time compilation unavailable.")
    logger.info(f"Compiling models in real-time: train={train_size}, pca={pca_components}")

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
@rate_limit()
def classify_image():
    # Validate Content-Type
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400
    
    data = request.get_json(silent=True)
    if not data or 'pixels' not in data:
        return jsonify({"error": "Invalid request payload: missing 'pixels'"}), 400

    algorithms_to_run = data.get('algorithms')
    if not algorithms_to_run:
        name = data.get('algorithm', 'knn_classical')
        config = data.get('config') or DEFAULT_CONFIGS.get(name, DEFAULT_CONFIGS['knn_classical'])
        algorithms_to_run = [{"name": name, "config": config}]

    # Validate number of algorithms (prevent resource exhaustion)
    if len(algorithms_to_run) > VALIDATION_LIMITS['max_algorithms_per_request']:
        return jsonify({"error": f"Maximum {VALIDATION_LIMITS['max_algorithms_per_request']} algorithms allowed per request"}), 400

    # Validate pixel array
    try:
        pixels = data['pixels']
        if not isinstance(pixels, list):
            return jsonify({"error": "'pixels' must be an array"}), 400
        if len(pixels) != VALIDATION_LIMITS['pixel_array_size']:
            return jsonify({"error": f"'pixels' must be an array of length {VALIDATION_LIMITS['pixel_array_size']}"}), 400
        
        # Validate all pixel values are numeric and in valid range
        for i, pixel in enumerate(pixels):
            if not isinstance(pixel, (int, float)):
                return jsonify({"error": f"Pixel at index {i} is not numeric"}), 400
            if not (0 <= pixel <= 255):
                return jsonify({"error": f"Pixel values must be between 0 and 255"}), 400
        
        X_test = np.array(pixels, dtype=np.float32).reshape(1, -1)
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid pixel data: {str(e)}")
        return jsonify({"error": "Invalid 'pixels' array format"}), 400

    predictions = {}
    timings = {}

    for algo in algorithms_to_run:
        try:
            name = algo.get('name')
            if not name:
                raise ValueError("Algorithm entry missing 'name'")
            
            # Validate algorithm name (whitelist)
            if name not in ['knn_classical', 'qknn_sim']:
                logger.warning(f"Invalid algorithm requested: {name}")
                predictions[name] = "Error"
                timings[name] = -1
                continue
            
            config = algo.get('config') or DEFAULT_CONFIGS.get(name, DEFAULT_CONFIGS['knn_classical'])
            
            # Validate and sanitize configuration
            try:
                validated_config = validate_config(config, name)
                # Merge with defaults
                final_config = {**DEFAULT_CONFIGS.get(name, {}), **validated_config}
            except ValueError as e:
                logger.warning(f"Invalid config for {name}: {str(e)}")
                predictions[name] = "Error"
                timings[name] = -1
                continue

            start_time = time.time()

            if ENVIRONMENT == 'PYTHONANYWHERE':
                scaler, pca, X_train_pca, y_train = get_models_precompiled(final_config)
            else:
                scaler, pca, X_train_pca, y_train = get_models_realtime(final_config)

            X_test_scaled = scaler.transform(X_test)
            X_test_pca = pca.transform(X_test_scaled)

            if name == 'knn_classical':
                pred = run_knn_classical(X_train_pca, y_train, X_test_pca, final_config)
            elif name == 'qknn_sim':
                pred = run_qknn_simulated(X_train_pca, y_train, X_test_pca, final_config)
            else:
                pred = "Error"

            predictions[name] = pred
            timings[name] = round((time.time() - start_time) * 1000)

        except Exception as e:
            logger.error(f"Error processing algorithm '{name}': {str(e)}")
            predictions[name or 'unknown'] = "Error"
            timings[name or 'unknown'] = -1

    return jsonify({"predictions": predictions, "timings": timings}), 200

@app.route('/get_random_mnist_image', methods=['GET'])
@rate_limit()
def get_random_mnist_image():
    if X_full is None:
        return jsonify({"error": "Dataset not available"}), 503

    digit_filter = request.args.get('digit', type=int)
    
    # Validate digit filter input
    if digit_filter is not None:
        if not isinstance(digit_filter, int) or not (0 <= digit_filter <= 9):
            return jsonify({"error": "Invalid digit filter. Must be between 0 and 9"}), 400
        
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
    # Production-safe configuration
    DEBUG_MODE = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    PORT = int(os.getenv('PORT', 5000))
    
    if DEBUG_MODE:
        logger.warning("⚠️  DEBUG MODE ENABLED - DO NOT USE IN PRODUCTION!")
    
    app.run(
        host='0.0.0.0', 
        port=PORT, 
        debug=DEBUG_MODE,
        threaded=True  # Better concurrency
    )