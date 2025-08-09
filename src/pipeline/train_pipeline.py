import os
import sys
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from catboost import CatBoostRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

## Paths

data_path=os.path.join("artifacts", "data.csv")
model_path=os.path.join("artifacts", "model.pkl")
preprocessor_path=os.path.join("artifacts", "preprocessor.pkl")

def load_date():
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Data file not found at {data_path}')
    df = pd.read_csv(data_path)
    logging.info(f"Data loaded successfully with shape{df.shape}")
    return df

def preprocessor_data(df):
    target_column = "math_score"
    X = df.drop(columns=[target_column])
    Y = df[target_column]
    return X, Y

def train_and_evaluate(X,Y):
    X_train, X_test, Y_train, Y_test = train_and_evaluate(
        X, Y, test_size=0.2, random_state=42
    )

    model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function="RSME",
    verbose=False
    )

    logging.info("Training model...")
    model.fit(X_train, Y_train)

    logging.info("Evaluating model...")
    Y_pred = model.predict(X_test)

    mae=mean_absolute_error(Y_test, Y_pred)
    mse=mean_squared_error(Y_test, Y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(Y_test, Y_pred)

    logging.info(f"MAE:{mae}, RMSE:{rmse}, R2:{r2}")

    return model, {"MAE":mae, "RMSE":rmse, "R2":r2}

def main():
    try:
        logging.info("Starting training pipeline...")

        df = load_date()
        X, Y = preprocessor_data(df)
        model, metrics = train_and_evaluate(X,Y)

        logging.info("Saving model...")
        save_object(file_path=model_path, obj= model)

        logging.info(f'Training completed sucessfully. Metrics: {metrics}')

    except Exception as e:
        logging.error(f"Error in training pipeline:{e}")
        raise CustomException(e, sys)
    

if __name__=="__main__":
    main()