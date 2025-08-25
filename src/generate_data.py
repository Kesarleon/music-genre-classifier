import numpy as np
import pandas as pd
import os

def generate_data(n_samples=5000, output_dir="data"):
    """
    Generates a synthetic dataset of music tracks and saves it to a CSV file.
    """
    np.random.seed(42)

    generos = ["Pop", "Rock", "Reggaeton", "Jazz", "Clasica"]
    probs = [0.4, 0.3, 0.2, 0.07, 0.03]  # imbalanced distribution

    duracion = np.random.normal(200, 50, n_samples)
    bpm = np.random.normal(120, 30, n_samples)
    energia = np.random.uniform(0, 1, n_samples)
    acustica = np.random.uniform(0, 1, n_samples)
    popularidad = np.random.randint(0, 101, n_samples)
    instrumentalidad = np.random.beta(2, 5, n_samples)

    genero = np.random.choice(generos, size=n_samples, p=probs)

    df = pd.DataFrame({
        "duracion": duracion,
        "bpm": bpm,
        "energia": energia,
        "acustica": acustica,
        "popularidad": popularidad,
        "instrumentalidad": instrumentalidad,
        "genero_musical": genero
    })

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "raw_music_data.csv")
    df.to_csv(output_path, index=False)

    print(f"Data generated and saved to {output_path}")
    return df

if __name__ == "__main__":
    generate_data()
