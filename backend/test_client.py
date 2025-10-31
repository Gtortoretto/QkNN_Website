import requests
import pandas as pd
import numpy as np
import json

try:
    print("Loading dataset to select a random image...")
    mnist_df = pd.read_csv('/home/gabriel/projects/QkNN_Website/docs/mnist_train.csv')
    y_train = mnist_df.iloc[:, 0].values
    X_train = mnist_df.iloc[:, 1:].values
except FileNotFoundError:
    print("FATAL: mnist_train.csv not found in this directory. Cannot run test.")
    exit()

random_index = np.random.randint(0, len(X_train))
image_pixels = X_train[random_index]
true_label = y_train[random_index]

print(f"Selected random image at index: {random_index}")
print(f"The actual label is: {true_label}")

api_url = "http://127.0.0.1:5000/classify"


payload = {
    "pixels": image_pixels.tolist(), 
    "algorithms": [
        {
            "name": "knn_classical",
            "config": {
                "training_size": 250,
                "pca_components": 15,
                "k": 3
            }
        },
        {
            "name": "qknn_sim",
            "config": {
                "training_size": 250,
                "pca_components": 10,
                "k": 3,
                "shots": 1024
            }
        }
    ]
}

try:
    print("\nSending request to Flask server...")

    response = requests.post(api_url, json=payload, timeout=300)

    if response.status_code == 200:
        print("\n✅ Success! Server responded:")

        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\nError! Server returned status code: {response.status_code}")
        print(f"Response: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"\nA request error occurred: {e}")