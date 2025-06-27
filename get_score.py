import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import datetime

def get_score(ticker):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=365)

    df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))

    df['return'] = df['Close'].pct_change()
    df['target'] = (df['return'].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)

    features = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    labels = df['target']

    X_train = features.iloc[:-1]
    y_train = labels.iloc[:-1]
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # 预测今天是否上涨
    X_today = features.iloc[-1:]
    prob = model.predict_proba(X_today)[0][1]  # 明日上涨概率
    return round(prob, 4)


if __name__ == "__main__":
    ticker = "KO"
    score = get_score(ticker)
    print(f"Ticker: {ticker}, Score: {score}")
    # 这里可以根据 score 做进一步的决策，比如是否买入等