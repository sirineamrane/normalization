import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

def detect_best_normalization(X):
    # Convert to numpy array if it's a DataFrame
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Compute statistics
    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    mean_vals = np.mean(X, axis=0)
    std_vals = np.std(X, axis=0)
    
    # Detect outliers using Interquartile Range (IQR)
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    outliers = ((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).sum(axis=0)
    
    # Plot the data distribution
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=X)
    plt.title("Feature Distributions")
    plt.show()
    
    print("\nFeature Statistics:")
    print(f"Min values: {min_vals}")
    print(f"Max values: {max_vals}")
    print(f"Mean values: {mean_vals}")
    print(f"Standard deviation: {std_vals}")
    print(f"Outliers detected (per feature): {outliers}\n")

    # Decision Logic
    if np.any(outliers > 5):  # If significant number of outliers
        print("🔹 Recommended Normalization: **Z-score Normalization (StandardScaler)**")
        print("Reason: Dataset contains outliers.")
        scaler = StandardScaler()
    
    elif np.any((max_vals - min_vals) > 1000):  # If features have very different scales
        print("🔹 Recommended Normalization: **Min-Max Scaling**")
        print("Reason: Features have very different scales.")
        scaler = MinMaxScaler()
    
    else:  # Default case, usually for deep learning
        print("🔹 Recommended Normalization: **Batch Normalization**")
        print("Reason: Works well with deep learning models.")
        scaler = StandardScaler()  # Placeholder, since BN is applied inside the NN

    # Apply chosen normalization
    X_scaled = scaler.fit_transform(X)
    
    print("\n✅ Normalization Applied Successfully!")
    return X_scaled

# Example dataset (modify this with your actual data)
X_example = np.array([
    [100, 0.1, 5000],
    [500, 0.5, 7000],
    [1000, 0.9, 9000],
    [2000, 1.5, 12000],
    [5000, 2.0, 15000]
])

# Run the function
X_normalized = detect_best_normalization(X_example)
