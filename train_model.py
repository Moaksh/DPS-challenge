from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import joblib 
import os
from data_utils import load_and_preprocess_data

def train_evaluate_save_arima(data_file='data.csv', arima_order=(5, 1, 0)):
    df = load_and_preprocess_data(data_file)
    categories = df['Category'].unique()

    trained_models = {}
    evaluation_results = {}

    for category in categories:
        df_cat = df[df['Category'] == category].copy()
        df_cat.set_index('Date', inplace=True)
        df_cat = df_cat[['Value']].sort_index()

        try:
            df_cat = df_cat.asfreq('MS') 
            df_cat['Value'] = df_cat['Value'].ffill()
        except ValueError as e:
            print(f"Warning: Could not set frequency for {category}. Using original index. Error: {e}")
        except Exception as e:
             print(f"Warning: An unexpected error occurred setting frequency for {category}: {e}")

        train_end_date = '2019-12-01'
        test_start_date = '2020-01-01'

        train_data = df_cat[:train_end_date]['Value']
        test_data = df_cat[test_start_date:]['Value'] 

        print(f"Training data size: {len(train_data)}")
        print(f"Test data size: {len(test_data)}")
        print(f"Training ARIMA{arima_order} model...")
        try:
            model = ARIMA(train_data, order=arima_order)
            model_fit = model.fit()
            print("Model training complete.")
            trained_models[category] = model_fit

            print("Evaluating model on the test set (2020 data)...")
            pred_start = test_data.index.min()
            pred_end = test_data.index.max()
            predictions = model_fit.predict(start=pred_start, end=pred_end)
            predictions = predictions.reindex(test_data.index)

            mse = mean_squared_error(test_data, predictions)
            rmse = np.sqrt(mse)
            print(f"Test RMSE for {category}: {rmse:.2f}")
            evaluation_results[category] = rmse

        except Exception as e:
            print(f"Error training/evaluating ARIMA for {category}: {e}")
            print("Skipping this category.")

    if trained_models:
        if not os.path.exists('models'):
            os.makedirs('models')
        model_filename = 'models/accident_predictor_arima.joblib'
        joblib.dump(trained_models, model_filename)
        print(f"\nAll trained ARIMA models saved to {model_filename}")
        avg_rmse = 0
        count = 0
        for cat, rmse_val in evaluation_results.items():
            print(f"- {cat}: {rmse_val:.2f}")
            avg_rmse += rmse_val
            count += 1
        if count > 0:
             print(f"\nAverage RMSE across categories: {avg_rmse/count:.2f}")
    else:
        print("\nNo models were trained successfully.")

if __name__ == '__main__':
    train_evaluate_save_arima(arima_order=(5, 1, 0))
