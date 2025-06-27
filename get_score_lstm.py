import yfinance as yf
import numpy as np
import pandas as pd
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

def get_score_lstm(ticker):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=365)
    df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))
    df = df[['Close']].dropna()

    # 构建标签（涨为1）
    df['return'] = df['Close'].pct_change()
    df['target'] = (df['return'].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)

    # 归一化
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(df[['Close']])

    # 构造时间序列数据 (lookback=10)
    X, y = [], []
    lookback = 10
    for i in range(lookback, len(scaled_close)-1):
        X.append(scaled_close[i-lookback:i, 0])
        y.append(df['target'].iloc[i])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # 构建模型
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')

    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    # 用最新的窗口预测
    last_window = scaled_close[-lookback:].reshape(1, lookback, 1)
    prob = model.predict(last_window)[0][0]
    return round(float(prob), 4)



if __name__ == "__main__":
    ticker = "HOOD"
    score = get_score_lstm(ticker)
    print(f"Ticker: {ticker}, Score: {score}")
    # 这里可以根据 score 做进一步的决策，比如是否买入等