import pypyodbc
import pyodbc
import pandas as pd
from datetime import date
from pmdarima import auto_arima
from sklearn.ensemble.forest import RandomForestRegressor
from functions import generate_lag, generate_monthscore, subtract_month, randomforest_predict
from urllib.parse import quote_plus
import sqlalchemy


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

# Generate Month_score (Current year and month)
dt = date.today()
Month_score = generate_monthscore(dt)

# Data Type Conversion
sales_data[sales_data.date < Month_score]
sales_data = sales_data.astype({'revenue': float})
sales_data['date'] = pd.to_datetime(sales_data['date'])
sales_data.set_index('date', inplace=True)
sales_month = sales_data.resample('M')[['revenue']].sum()

# ARIMA
model_arima = auto_arima(sales_month, start_p=1, start_q=1, max_p=8, max_q=8,
                         start_P=1, start_Q=1, max_P=8, max_Q=8,
                         m=12, seasonal=True, trace=True, d=1, D=1,
                         suppress_warnings=True, error_action="ignore", stepwise=True, n_fits=30)
revenue_arima = model_arima.predict(n_periods=3)

# Random Forest
trainX, trainY = generate_lag(inp=sales_month, lag=5)
model_rf = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=10, min_samples_split=2,
                                 max_features='auto', verbose=1)
model_rf.fit(trainX, trainY)
revenue_rf = randomforest_predict(sales_month, lag=5, model=model_rf, n_periods=3)

# Merge result
Month_SQN = pd.date_range(start=dt, periods=3, freq='M').map(generate_monthscore)
Month_SQN = [int(i[0:4]+i[5:7]) for i in Month_SQN]
Month_score = int(Month_score[0:4]+Month_score[5:7])
revenue = (revenue_arima + revenue_rf) / 2
sales_forecast = pd.DataFrame(
    {'Month_SQN': Month_SQN, 'Level_cat': 1, 'Div': '', 'Revenue': revenue, 'Month_score': Month_score})

# update sales forecast
params = quote_plus(r'DRIVER={SQL Server};SERVER=10.188.34.5;DATABASE=EDB;UID=hsproot;PWD=Pl@ceholder1')
engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
sales_forecast.to_sql(name='ForecastSales_WX', con=engine, schema='dbo', index=False, if_exists='append')

# Division Forecast
divs = ["Commercial AC", "Dishwasher", "Drum Washing Machine", "Freezer", "Home Air Conditioner", "Refrigerator",
        "Washing Machine"]
div_forecast = pd.DataFrame()
for div in divs:
    sales_div = sales_data[sales_data.division == div]
    sales_div.drop(columns='division', inplace=True)
    sales_div = sales_div.resample('M').sum()
    model_div = auto_arima(sales_div, start_p=1, start_q=1, max_p=8, max_q=8,
                           start_P=1, start_Q=1, max_P=8, max_Q=8,
                           m=12, seasonal=True, trace=True, d=1, D=1,
                           suppress_warnings=True, error_action="ignore", stepwise=True, n_fits=30)
    revenue_div = model_div.predict(n_periods=3)
    div_tmp = pd.DataFrame({'Month_SQN': Month_SQN, 'Level_cat': 2, 'Div': div, 'Revenue': revenue_div, 'Month_score': Month_score})
    div_forecast = pd.concat([div_forecast, div_tmp], ignore_index=True)

# update division forecast
params = quote_plus(r'DRIVER={SQL Server};SERVER=10.188.34.5;DATABASE=EDB;UID=hsproot;PWD=Pl@ceholder1')
engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
div_forecast.to_sql(name='ForecastSales_WX', con=engine, schema='dbo', index=False, if_exists='append')

# update YTD sales forecast
last_month = subtract_month(dt)
connection = pypyodbc.connect('Driver={SQL Server};'
                              'Server=10.188.34.5;'
                              'Database=EDB;'
                              'uid=hsproot;'
                              'pwd=Pl@ceholder1')
cur = connection.cursor()
cur.execute('SELECT * \
             FROM [EDB].[dbo].[ForecastSales_HSP_accumulative] \
             WHERE [MonthScore] = {}'.format(last_month))
record = cur.fetchall()
YTD_forecast = pd.DataFrame(record, columns=[x[0] for x in cur.description])
connection.close()

Month_score = str(Month_score)
Month_SQN = [str(i) for i in Month_SQN]
YTD_forecast = YTD_forecast[YTD_forecast.ym < Month_score]
YTD_forecast['monthscore'] = Month_score
YTD_forecast.loc[YTD_forecast.ym == last_month, 'salesactual'] = sales_month.values[-1, 0]
YTD_forecast_tmp = pd.DataFrame({'YM': Month_SQN, 'salesforcast': revenue, 'salesactual': revenue,
                                 'accsum': 0, 'monthscore': Month_score})
YTD_forecast = pd.concat([YTD_forecast, YTD_forecast_tmp], ignore_index=True)
YTD_forecast['year'] = YTD_forecast['YM'].map(lambda x: x[0:4])
YTD_forecast['accsum'] = YTD_forecast.groupby('year').salesactual.cumsum()
YTD_forecast = YTD_forecast.assign(accsum=YTD_forecast.accsum - YTD_forecast.salesactual + YTD_forecast.salesforcast)
YTD_forecast.drop(columns='year', inplace=True)

params = quote_plus(r'DRIVER={SQL Server};SERVER=10.188.34.5;DATABASE=EDB;UID=hsproot;PWD=Pl@ceholder1')
engine = sqlalchemy.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
YTD_forecast.to_sql(name='ForecastSales_HSP_accumulative', con=engine, schema='dbo', index=False, if_exists='append')


