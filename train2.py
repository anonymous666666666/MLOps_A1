# train2.py
from sklearn.kernel_ridge import KernelRidge
from misc import load_data, preprocess, split_data, train_model, evaluate_model
import numpy as np

def main():
    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

    # You can tune alpha and kernel as needed; this is a working default
    model = KernelRidge(kernel='rbf', alpha=1.0)
    fitted = train_model(model, X_train, y_train)

    metrics = evaluate_model(fitted, X_test, y_test)
    print("=== KernelRidge Test Results ===")
    print(f"Test MSE: {metrics['mse']:.4f}")
    print(f"Test RMSE: {metrics['rmse']:.4f}")

if __name__ == "__main__":
    main()