import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from pathlib import Path

CONFIG_TO_GENERATE = {
    "train_size": 250,
    "pca_components": 16,
    "key": "train250_pca16"
}
DATA_FILE = Path("backend/data/mnist_train.csv")
MODELS_DIR = Path("backend/models")

def generate_models():

    print(f"Loading full dataset from {DATA_FILE}...")
    if not DATA_FILE.exists():
        print(f"FATAL: Data file not found at {DATA_FILE}")
        print("Please download 'mnist_train.csv' and place it in a 'docs' directory.")
        return

    try:
        mnist_df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    y_full = mnist_df.iloc[:, 0].values
    X_full = mnist_df.iloc[:, 1:].values

    print("Full dataset loaded.")

    cfg = CONFIG_TO_GENERATE
    train_size = cfg["train_size"]
    pca_components = cfg["pca_components"]
    key = cfg["key"]
    
    print(f"\nGenerating models for config: {key}...")

    X_train_slice, _, y_train_slice, _ = train_test_split(
        X_full,
        y_full,
        train_size=train_size,
        stratify=y_full,
        random_state=42 
    )
    
    print(f"Created stratified sample of {train_size} images.")

    print("Training StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_slice)

    print(f"Training PCA with {pca_components} components...")
    pca = PCA(n_components=pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)

    print(f"Saving artifacts to {MODELS_DIR} directory...")
    MODELS_DIR.mkdir(exist_ok=True)

    joblib.dump(scaler, MODELS_DIR / f"{key}_scaler.pkl")
    joblib.dump(pca, MODELS_DIR / f"{key}_pca.pkl")
    
    np.save(MODELS_DIR / f"{key}_xtrain.npy", X_train_pca)
    np.save(MODELS_DIR / f"{key}_ytrain.npy", y_train_slice)

    print("-" * 30)
    print("Pre-compiled models generated successfully!")
    print(f"Directory '{MODELS_DIR}' now contains:")
    for f in MODELS_DIR.glob(f"{key}*"):
        print(f"  - {f.name}")
    print("-" * 30)


if __name__ == "__main__":
    generate_models()
