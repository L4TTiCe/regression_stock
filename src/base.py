import pandas as pd
import quandl, datetime
import numpy as np
from sklearn import preprocessing , model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

# Config
quandl.ApiConfig.api_key = ""   # API Key
# To avoid truncation in DataFrame
pd.set_option('display.width', 320)
np.set_printoptions(linewidth=320)
pd.set_option("display.max_columns", 15)

fig = plt.figure(figsize=(13.66, 7.68))

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]        # Trimming df to retain only relevant Cols.
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100            # HL_PCT is the Volatility %
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100     # % CHG in that day

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]      # Trimming df to retain only relevant Cols.

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)     # Remove NaN

forecast_out = 30        # predicting Stock Values ? days into the future
df['label'] = df[forecast_col].shift(-forecast_out)         # Creating lable col. that is 15 days(Future)'s Close Val.

X = np.array(df.drop(['label'], 1))     # Features List
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]            # To be predicted

df.dropna(inplace=True)
y = np.array(df['label'])               # Label to be predicted

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)        # Shuffle and split Dataset

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)                   # Train
accuracy = clf.score(X_test, y_test)        # Test
print("Classifier Accuracy = {} ".format(accuracy))
forecast_set = clf.predict(X_lately)
print(forecast_set, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()