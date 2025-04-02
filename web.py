from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
from prophet import Prophet

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/ss')
def stockselection():
    return render_template('ss.html')

@app.route('/ss/result',methods=['POST'])
def result():
    name = request.form['company']
    forecasts = ts_predict(name)
    table_html = forecasts.to_html(classes='table table-striped', index=False)
    return render_template('result.html',table = table_html, name= name)



def ts_predict(name):
    x = datetime.datetime.now()
    x = x.strftime("%Y-%m-%d")
    
    nifty50 = yf.Ticker(name)
    df = nifty50.history(start= '2010-01-01', end=x)

    df = df.reset_index()
    df['Date'] = df["Date"].dt.date

    df = df[['Date',"Close"]]

    # Prepare data for Prophet
    df_prophet = df.rename(columns={"Date": "ds", "Close": "y"})
    df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])  # Ensure datetime format

    # Initialize and fit Prophet model
    model = Prophet()
    model.fit(df_prophet)

    # Create future dataframe for the next 30 days
    future = model.make_future_dataframe(periods=30)

    # Make predictions
    forecast = model.predict(future)




    # Show the predicted values for the next 30 days
    r = forecast[['ds', 'yhat']].tail(30)
    r.columns =['Date', 'Price']
    return r

   
        


if __name__ == '__main__':
    app.run()


