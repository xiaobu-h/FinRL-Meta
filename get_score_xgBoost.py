import yfinance as yf
import pandas as pd
import datetime
from xgboost import XGBClassifier

def get_score_xgboost(ticker):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=365)
    df = yf.download(ticker, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'))

    df['return'] = df['Close'].pct_change()
    df['target'] = (df['return'].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)

    features = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    labels = df['target']

    model = XGBClassifier( eval_metric='logloss')
    model.fit(features[:-1], labels[:-1])

    X_today = features.iloc[-1:]
    prob = model.predict_proba(X_today)[0][1]
    return round(prob, 4)


if __name__ == "__main__":
    ticker = "RCL"
    score = get_score_xgboost(ticker)
    print(f"Ticker: {ticker}, Score: {score}")
    # 这里可以根据 score 做进一步的决策，比如是否买入等