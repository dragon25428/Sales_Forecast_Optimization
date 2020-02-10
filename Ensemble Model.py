import pypyodbc
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from pmdarima import auto_arima
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

pd.set_option('display.max_columns', 30, 'display.max_colwidth', 20, 'display.width', 100)
# ===================================================================================================================
# Extract Sales Data From [EDB].[dbo].[KE30]
connection = pypyodbc.connect('Driver={SQL Server};'
                              'Server=10.188.34.5;'
                              'Database=EDB;'
                              'uid=hsproot;'
                              'pwd=Pl@ceholder1')
cur = connection.cursor()
cur.execute("SELECT [Billing Date] as [date], [Division], SUM([Sales]) AS [revenue] \
             FROM [EDB].[dbo].[KE30] \
             GROUP BY [Billing Date], [Division] \
             ORDER BY [Billing Date]")
record = cur.fetchall()
sales_data = pd.DataFrame(record, columns=[x[0] for x in cur.description])
connection.close()

# ------------------------------------------- Data Manipulation -----------------------------------------------------
# Data Type Conversion
sales_data = sales_data.astype({'revenue': float})
sales_data['date'] = pd.to_datetime(sales_data['date'])
sales_data.set_index('date', inplace=True)

# Aggregation
sales_month = sales_data.resample('M')[['revenue']].sum()
sales_day = sales_data.resample('D')[['revenue']].sum()

# Seasonal Decomposition
# decomposition = seasonal_decompose(sales_month, model='additive', freq=12)
# fig, axes = plt.subplots(2, 2, figsize=(10, 7))
# fig.suptitle("Seasonality Decomposition")
# sns.lineplot(decomposition.observed.index, decomposition.observed.revenue, label="Sales Revenue", ax=axes[0, 0])
# sns.lineplot(decomposition.trend.index, decomposition.trend.revenue, label="trend", ax=axes[0, 1], color="orange")
# sns.lineplot(decomposition.seasonal.index, decomposition.seasonal.revenue, label="seasonal", ax=axes[1, 0], color="green")
# sns.lineplot(decomposition.resid.index, decomposition.resid.revenue, label="residual", ax=axes[1, 1], color="red")
# fig.autofmt_xdate()
# plt.tight_layout(rect=(0, 0, 1, 0.95))

# scale sales data to [0, 1] range
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(sales_month)


# Train Test Split
def create_lags_feature(data, lags=1):
    dataX = []
    dataY = []
    for i in range(len(data) - lags):
        dataX.append(data[i:i + lags, 0])
        dataY.append(data[i + lags, 0])
    return np.array(dataX), np.array(dataY)

train_size = round(len(dataset) * 0.7)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:]
trainX, trainY = create_lags_feature(train, lags=5)
testX, testY = create_lags_feature(test, lags=5)
trainX = np.reshape(np.array(trainX), newshape=(
    trainX.shape[0], 1, trainX.shape[1]))  # newshape=[No. of sample, time step, No. of features]
testX = np.reshape(np.array(testX), newshape=(testX.shape[0], 1, testX.shape[1]))
scale_trainY = scaler.inverse_transform(trainY.reshape(-1, 1))
scale_testY = scaler.inverse_transform(testY.reshape(-1, 1))

# ---------------------------------------------- Modelling -------------------------------------------------------
# Create LSTM model
model = Sequential()
model.add(layer=LSTM(units=100, activation='tanh', input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(layer=Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(trainX, trainY, epochs=200, batch_size=1, verbose=1)
plt.plot(history.history['loss'])

# Make Prediction
LSTM_trainPredict = model.predict(trainX)
LSTM_testPredict = model.predict(testX)
LSTM_trainPredict = scaler.inverse_transform(LSTM_trainPredict)
LSTM_testPredict = scaler.inverse_transform(LSTM_testPredict)
train_error_pct = sum(abs(LSTM_trainPredict - scale_trainY)) / (len(scale_trainY) * np.mean(scale_trainY))
test_error_pct = sum(abs(LSTM_testPredict - scale_testY)) / (len(scale_testY) * np.mean(scale_testY))
print('train error: {} \n test error: {}'.format(train_error_pct, test_error_pct))

# Plotting
fig, ax = plt.subplots(2, 1)
ax[0].plot(scale_trainY, label='actual data')
ax[0].plot(LSTM_trainPredict, label='LSTM model')
ax[0].set_title('training')
ax[1].plot(scale_testY[0:100])
ax[1].plot(LSTM_testPredict[0:100])
ax[1].set_title('testing')
fig.legend()
plt.tight_layout(rect=(0, 0, 1, 0.9))
plt.savefig('neural_network.png', dpi=500)

# Random Forest Model
rf = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=10, min_samples_split=2,
                           max_features='auto', verbose=1)
rf.fit(trainX.reshape(trainX.shape[0], trainX.shape[2]), trainY)

# Make Prediction
RF_trainPredict = rf.predict(trainX.reshape(trainX.shape[0], trainX.shape[2]))
RF_testPredict = rf.predict(testX.reshape(testX.shape[0], testX.shape[2]))
RF_trainPredict = scaler.inverse_transform(RF_trainPredict.reshape(-1, 1))
RF_testPredict = scaler.inverse_transform(RF_testPredict.reshape(-1, 1))
train_error_pct = sum(abs(RF_trainPredict - scale_trainY)) / (len(scale_trainY) * np.mean(scale_trainY))
test_error_pct = sum(abs(RF_testPredict - scale_testY)) / (len(scale_testY) * np.mean(scale_testY))
print('train error: {} \n test error: {}'.format(train_error_pct, test_error_pct))

# Plotting
fig, ax = plt.subplots(2, 1)
ax[0].plot(scale_trainY, label='actual data')
ax[0].plot(RF_trainPredict, label='random forest')
ax[0].set_title('training')
ax[1].plot(scale_testY)
ax[1].plot(RF_testPredict)
ax[1].set_title('testing')
ax[1].set_ylim(0.5e8, 1.3e8)
fig.legend()
plt.tight_layout(rect=(0, 0, 1, 0.9))
plt.savefig('random_forest.png', dpi=500)

# Linear Regression Model
fig, ax = plt.subplots(2, 1)
ax[0].plot(scale_trainY, label='actual data')
ax[1].plot(scale_testY)
ax[1].set_ylim(0.5e8, 1.3e8)

for alpha in [0.01, 0.001]:
    lr = Lasso(alpha=alpha)
    lr.fit(trainX.reshape(trainX.shape[0], trainX.shape[2]), trainY)
    LR_trainPredict = lr.predict(trainX.reshape(trainX.shape[0], trainX.shape[2]))
    LR_testPredict = lr.predict(testX.reshape(testX.shape[0], testX.shape[2]))
    LR_trainPredict = scaler.inverse_transform(LR_trainPredict.reshape(-1, 1))
    LR_testPredict = scaler.inverse_transform(LR_testPredict.reshape(-1, 1))
    train_error_pct = sum(abs(LR_trainPredict - scale_trainY)) / (len(scale_trainY) * np.mean(scale_trainY))
    test_error_pct = sum(abs(LR_testPredict - scale_testY)) / (len(scale_testY) * np.mean(scale_testY))
    ax[0].plot(LR_trainPredict, label='alpha={}'.format(alpha))
    ax[1].plot(LR_testPredict)
    print('train error: {} \n test error: {}'.format(train_error_pct, test_error_pct))
fig.legend()
plt.tight_layout(rect=(0, 0, 1, 0.9))
plt.savefig('linear_regression.png', dpi=500)

# Arima Model
arima_model = auto_arima(sales_month[0:-9], start_p=1, start_q=1, max_p=8, max_q=8,
                         start_P=1, start_Q=1, max_P=8, max_Q=8,
                         m=12, seasonal=True, trace=True, d=1, D=1,
                         suppress_warnings=True, error_action="ignore", stepwise=True, n_fits=30)
ARIMA_Predict = arima_model.predict(n_periods=9)
test_error_pct = sum(abs(ARIMA_Predict - sales_month.iloc[-9:, 0].values)) / (
        len(sales_month[-9:]) * np.mean(sales_month[-9:]))
print('test error: {}'.format(test_error_pct))

# Plotting
plt.plot(sales_month, label='actual data')
plt.plot(sales_month[-9:].index, ARIMA_Predict, label='ARIMA model')
plt.legend()
plt.tight_layout()
plt.savefig('arima_model.png', dpi=500)

# Exponential Smoothing
ES = ExponentialSmoothing(sales_month[0:-9], trend='add', seasonal_periods=12, seasonal='add', damped=True).fit()
ES_Predict = ES.forecast(9)
test_error_pct = sum(abs(ES_Predict - sales_month.iloc[-9:, 0].values)) / (
        len(sales_month[-9:]) * np.mean(sales_month[-9:]))
print('test error: {}'.format(test_error_pct))

# Plotting
plt.plot(sales_month, label='actual data')
plt.plot(sales_month[-9:].index, ES_Predict, label='Holt-Winters model')
plt.legend()
plt.tight_layout()
plt.savefig('Exponential_model.png', dpi=500)

# XGBOOST Model
cv_model = xgb.XGBRFRegressor(objective='reg:squarederror')
params = {'learning_rate': [0.01, 0.1, 0.2, 0.5, 1],
          'max_depth': [3, 4, 5, 8, 10],
          'n_estimator': [100, 200, 300, 500],
          'verbosity': [2]}
grid = GridSearchCV(estimator=cv_model, param_grid=params, cv=3)
grid.fit(trainX.reshape(trainX.shape[0], trainX.shape[2]), trainY)

xgb_model = xgb.XGBRFRegressor(learning_rate=1, max_depth=8, n_estimators=100, verbosity=2)
xgb_model.fit(trainX.reshape(trainX.shape[0], trainX.shape[2]), trainY)

# Make Prediction
XGB_trainPredict = xgb_model.predict(trainX.reshape(trainX.shape[0], trainX.shape[2]))
XGB_testPredict = xgb_model.predict(testX.reshape(testX.shape[0], testX.shape[2]))
XGB_trainPredict = scaler.inverse_transform(XGB_trainPredict.reshape(-1, 1))
XGB_testPredict = scaler.inverse_transform(XGB_testPredict.reshape(-1, 1))
train_error_pct = sum(abs(XGB_trainPredict - scale_trainY)) / (len(scale_trainY) * np.mean(scale_trainY))
test_error_pct = sum(abs(XGB_testPredict - scale_testY)) / (len(scale_testY) * np.mean(scale_testY))
print('train error: {} \n test error: {}'.format(train_error_pct, test_error_pct))

# Plotting
fig, ax = plt.subplots(2, 1)
ax[0].plot(scale_trainY, label='actual data')
ax[0].plot(XGB_trainPredict, label='XGBoost Model')
ax[0].set_title('training')
ax[1].plot(scale_testY)
ax[1].plot(XGB_testPredict)
ax[1].set_title('testing')
ax[1].set_ylim(0.5e8, 1.3e8)
fig.legend()
plt.tight_layout(rect=(0, 0, 1, 0.9))
plt.savefig('XGB_model.png', dpi=500)

# Compile All the forecast (predict 9 months)
predict_month = sales_month[-9:].index
XGB_RF_testPredict = (XGB_testPredict + RF_testPredict) / 2
XGB_RF_forecast = pd.DataFrame(np.hstack([XGB_RF_testPredict, XGB_RF_testPredict * 1.1, XGB_RF_testPredict * 0.9]),
                               columns=['model', 'model_upper', 'model_lower'],
                               index=predict_month).reset_index()
XGB_RF_forecast = XGB_RF_forecast.melt(id_vars='date', value_vars=['model', 'model_upper', 'model_lower'], var_name='model')
XGB_ARIMA_testPredict = (ARIMA_Predict.reshape(-1, 1) + XGB_testPredict) / 2
XGB_ARIMA_forecast = pd.DataFrame(np.hstack([XGB_ARIMA_testPredict, XGB_ARIMA_testPredict * 1.1, XGB_ARIMA_testPredict * 0.9]),
                                  columns=['model', 'model_upper', 'model_lower'],
                                  index=predict_month).reset_index()
XGB_ARIMA_forecast = XGB_ARIMA_forecast.melt(id_vars='date', value_vars=['model', 'model_upper', 'model_lower'], var_name='model')
RF_ARIMA_testPredict = (ARIMA_Predict.reshape(-1, 1) + RF_testPredict) / 2
RF_ARIMA_forecast = pd.DataFrame(np.hstack([RF_ARIMA_testPredict, RF_ARIMA_testPredict * 1.1, RF_ARIMA_testPredict * 0.9]),
                                 columns=['model', 'model_upper', 'model_lower'],
                                 index=predict_month).reset_index()
RF_ARIMA_forecast = RF_ARIMA_forecast.melt(id_vars='date', value_vars=['model', 'model_upper', 'model_lower'], var_name='model')
LSTM_forecast = pd.DataFrame(np.hstack([LSTM_testPredict, LSTM_testPredict * 1.1, LSTM_testPredict * 0.9]),
                             columns=['model', 'model_upper', 'model_lower'],
                             index=predict_month).reset_index()
LSTM_forecast = LSTM_forecast.melt(id_vars='date', value_vars=['model', 'model_upper', 'model_lower'], var_name='model')

# Compile Model Plot
fig, ax = plt.subplots(figsize=(7, 4))
sns.lineplot(sales_month[-24:].index, sales_month[-24:].revenue, ax=ax, label='Actual Sales')
sns.lineplot(predict_month, ARIMA_Predict, label='ARIMA alone', color='orange')
sns.lineplot(XGB_ARIMA_forecast.date, XGB_ARIMA_forecast.value, label='ARIMA + XGB', color='green')
sns.scatterplot(x='date', y='value', data=XGB_ARIMA_forecast[XGB_ARIMA_forecast.model == 'model'], color='green', s=80)
plt.tight_layout()
fig.savefig('ARIMA+XGB model.png', dpi=500)

fig, ax = plt.subplots(figsize=(7, 4))
sns.lineplot(sales_month[-24:].index, sales_month[-24:].revenue, ax=ax, label='Actual Sales')
sns.lineplot(predict_month, ARIMA_Predict, label='ARIMA alone', color='orange')
sns.lineplot(x='date', y='value', label='ARIMA + RF', data=RF_ARIMA_forecast, color='purple')
sns.scatterplot(x='date', y='value', data=RF_ARIMA_forecast[RF_ARIMA_forecast.model == 'model'], color='purple', s=80)
plt.tight_layout()
fig.savefig('ARIMA+RF model.png', dpi=500)

fig, ax = plt.subplots(figsize=(7, 4))
sns.lineplot(sales_month[-24:].index, sales_month[-24:].revenue, ax=ax, label='Actual Sales')
sns.lineplot(predict_month, ARIMA_Predict, label='ARIMA alone', color='orange')
sns.lineplot(XGB_RF_forecast.date, XGB_RF_forecast.value, label='XGB + RF', color='tomato')
sns.scatterplot(x='date', y='value', data=XGB_RF_forecast[XGB_RF_forecast.model == 'model'], color='tomato', s=80)
plt.tight_layout()
fig.savefig('XGB+RF model.png', dpi=500)

fig, ax = plt.subplots(figsize=(7, 4))
sns.lineplot(sales_month[-24:].index, sales_month[-24:].revenue, ax=ax, label='Actual Sales')
sns.lineplot(predict_month, ARIMA_Predict, label='ARIMA alone', color='orange')
sns.lineplot(LSTM_forecast.date, LSTM_forecast.value, label='Neural Network', color='red')
sns.scatterplot(x='date', y='value', data=LSTM_forecast[LSTM_forecast.model == 'model'], color='red', s=80)
plt.tight_layout()
fig.savefig('LSTM model.png', dpi=500)

# Compile Model Accuracy
ARIMA_accuracy = (sum(abs(ARIMA_Predict.reshape(-1, 1) - scale_testY)) / (len(scale_testY) * np.mean(scale_testY)))[0]
Holt_accuracy = (sum(abs(ES_Predict.values.reshape(-1, 1) - scale_testY)) / (len(scale_testY) * np.mean(scale_testY)))[0]
LSTM_accuracy = (sum(abs(LSTM_testPredict - scale_testY)) / (len(scale_testY) * np.mean(scale_testY)))[0]
XGB_RF_accuracy = (sum(abs(XGB_RF_testPredict - scale_testY)) / (len(scale_testY) * np.mean(scale_testY)))[0]
RF_ARIMA_accuracy = (sum(abs(RF_ARIMA_testPredict - scale_testY)) / (len(scale_testY) * np.mean(scale_testY)))[0]
XGB_ARIMA_accuracy = (sum(abs(XGB_ARIMA_testPredict - scale_testY)) / (len(scale_testY) * np.mean(scale_testY)))[0]
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=['ARIMA', 'Holt', 'Neural Network', 'XGB+RF', 'RF+ARIMA', 'XGB+ARIMA'],
            y=[ARIMA_accuracy, Holt_accuracy, LSTM_accuracy, XGB_RF_accuracy, RF_ARIMA_accuracy, XGB_ARIMA_accuracy])
plt.title('Model Accuracy Comparison')
plt.tight_layout()
fig.savefig('Model Comparison.png', dpi=500)



