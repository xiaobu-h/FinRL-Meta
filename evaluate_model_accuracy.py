import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import datetime

import logging


# logging
log_filename = "evaluate.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def evaluate_model_accuracy(ticker="AAPL", model_type="xgboost", start="2022-01-01", end=None, lookback_days=60, threshold=0.8):
    if end is None:
        end = datetime.datetime.today().strftime('%Y-%m-%d')

    df = yf.download(ticker, start=start, end=end)
    df['return'] = df['Close'].pct_change()
    df['target'] = (df['return'].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)

    scores, actuals = [], []

    if model_type in ["random_forest", "xgboost"]:
        features = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        labels = df['target']
        for i in range(lookback_days, len(df) - 1):
            X_train, y_train = features.iloc[:i], labels.iloc[:i]
            X_test, y_test = features.iloc[i:i+1], labels.iloc[i:i+1]

            if model_type == "random_forest":
                model = RandomForestClassifier()
            else:
                model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, verbosity=0)

            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[0][1]  # 看涨的概率
            scores.append(prob)
            actuals.append(y_test.values[0])

    elif model_type == "lstm":
        df = df[['Close', 'target']]
        scaler = MinMaxScaler()
        df['Close'] = scaler.fit_transform(df[['Close']])
        lookback = 10
        for i in range(lookback_days + lookback, len(df) - 1):
            X, y = [], []
            for j in range(lookback, i):
                X.append(df['Close'].iloc[j-lookback:j].values)
                y.append(df['target'].iloc[j])
            X = np.array(X).reshape((-1, lookback, 1))
            y = np.array(y)

            model = Sequential()
            model.add(LSTM(32, input_shape=(lookback, 1)))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy')
            model.fit(X, y, epochs=5, batch_size=8, verbose=0)

            X_test = df['Close'].iloc[i-lookback:i].values.reshape((1, lookback, 1))
            prob = model.predict(X_test)[0][0]
            scores.append(prob)
            actuals.append(df['target'].iloc[i])
    else:
        raise ValueError("Invalid model_type. Use 'random_forest', 'xgboost', or 'lstm'.")

    # 只对分数大于threshold的信号进行评估
    total_signals = 0
    correct_signals = 0
    for score, true in zip(scores, actuals):
        if score > threshold:
            total_signals += 1
            if true == 1:
                correct_signals += 1

    if total_signals == 0:
        print(f"[{model_type.upper()}] No signals with score > {threshold}")
        return 0.0

    accuracy = correct_signals / total_signals
    print(f"[{model_type.upper()}] Precision on signals (score > {threshold}) for {ticker}: {round(accuracy * 100, 2)}% ({total_signals} signals)")
    return accuracy



if __name__ == "__main__":
    ticker = "MSFT"   
    date = "2024-06-01"
    days = 150
        
    logging.info(f"Evaluating models for ticker: {ticker} starting from {date} with lookback of {days} days")
    model_type3 = "random_forest"  # Change to "xgboost" or "lstm" as needed
    accuracy3 = evaluate_model_accuracy(ticker=ticker, model_type=model_type3, start=date, end=None, lookback_days=days)
    logging.info(f"Model Type: {model_type3}, Ticker: {ticker}, Accuracy: {round(accuracy3 * 100, 2)}%")
    
    model_type = "xgboost"  # Change to "random_forest" or "lstm" as needed
    accuracy1 = evaluate_model_accuracy(ticker=ticker, model_type=model_type, start=date, end=None, lookback_days=days)
    logging.info(f"Model Type: {model_type}, Ticker: {ticker}, Accuracy:        {round(accuracy1 * 100, 2)}%")
   
    


