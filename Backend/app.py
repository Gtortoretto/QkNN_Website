import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS # To handle requests from GitHub pages

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# --- Initialize the Flask App ---
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Load Pre-trained Models and Data (Do this ONCE at startup) ---
# NOTE: The paths are now updated to match your folder structure.
print("Loading models and data...")
try:
    with open("Data/Pickles/web_qknn_hybrid_scaler_250.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("Data/Pickles/web_qknn_hybrid_pca_250.pkl", "rb") as f:
        pca = pickle.load(f)
    with open("Data/Pickles/web_qknn_hybrid_xtrain_pca_250.pkl", "rb") as f:
        x_train_pca = pickle.load(f)
    with open("Data/Pickles/web_qknn_hybrid_ytrain_250.pkl", "rb") as f:
        y_train_new = pickle.load(f)
    print("Models and data loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    print("Please make sure the pickle files are in the 'Backend/Data/Pickles' directory.")
    # You might want to exit or handle this error appropriately
    # For now, we'll let it crash if files are missing.

# --- Your Quantum Distance Function (copied from your notebook) ---
def get_quantum_distance(vec_alpha, vec_vj, shots=1024):
    num_qubits_per_state = int(np.log2(len(vec_alpha)))
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
    simulator = AerSimulator()
    result = simulator.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts.get('1', 0) / shots

# --- Define the API Endpoint for Classification ---
@app.route('/classify', methods=['POST'])
def classify_image():
    # 1. Get the drawing from the request
    data = request.json
    # Expecting a flat array of 784 pixels (28*28)
    user_pixels = np.array(data['pixels']).reshape(1, -1) 

    # 2. Pre-process the user's drawing
    user_scaled = scaler.transform(user_pixels)
    user_pca = pca.transform(user_scaled)
    
    # 3. Normalize the vector for the quantum circuit
    norm_user_vector = np.linalg.norm(user_pca)
    if norm_user_vector == 0:
        return jsonify({'error': 'Cannot process blank image'}), 400
    normed_user_vector = user_pca.flatten() / norm_user_vector

    # 4. Run the QkNN comparison
    distances = []
    for i in range(x_train_pca.shape[0]):
        train_vector = x_train_pca[i]
        norm_train_vector = np.linalg.norm(train_vector)
        if norm_train_vector == 0: continue
        
        normed_train_vector = train_vector / norm_train_vector
        distance = get_quantum_distance(normed_user_vector, normed_train_vector)
        distances.append({'label': y_train_new[i], 'distance': distance})
    
    # 5. Find the k-nearest neighbors and predict
    k = 5
    sorted_neighbors = sorted(distances, key=lambda x: x['distance'])
    k_nearest_labels = [n['label'] for n in sorted_neighbors[:k]]
    
    # Find the most common label (the prediction)
    from collections import Counter
    prediction = Counter(k_nearest_labels).most_common(1)[0][0]

    # 6. Return the result
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) # Use a specific port