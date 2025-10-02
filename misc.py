# misc.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Tuple, Any, Dict

def load_data() -> pd.DataFrame:
    """
    Loads the Boston dataset from the CMU URL as described in the assignment.
    Returns a DataFrame with feature columns and 'MEDV' as target.
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Basic preprocessing:
    - No missing values expected.
    - Returns X (features) and y (target).
    """
    X = df.drop(columns=['MEDV'])
    y = df['MEDV']
    return X, y

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train-test split."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Fit and return fitted model."""
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate model on test set and return dictionary of metrics.
    Currently returns MSE and RMSE.
    """
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    return {"mse": float(mse), "rmse": float(rmse)}

if __name__ == "__main__":
    # quick smoke test
    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    from sklearn.tree import DecisionTreeRegressor
    m = DecisionTreeRegressor(random_state=42)
    train_model(m, X_train, y_train)
    print(evaluate_model(m, X_test, y_test))