from fbprophet import Prophet
import pandas as pd
import numpy as np
import os
df1 = pd.read_csv("retail_sales.csv")
model = Prophet()
model.fit(df1)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)
