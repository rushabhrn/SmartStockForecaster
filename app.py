import pandas as pd
from prophet import Prophet
import streamlit as st
import matplotlib.pyplot as plt

# Load your data (you can replace this with your actual file paths)
transactional_data = pd.concat([
    pd.read_csv('Transactional_data_retail_01.csv'),
    pd.read_csv('Transactional_data_retail_02.csv')
])

# Ensure the InvoiceDate is in datetime format
transactional_data['InvoiceDate'] = pd.to_datetime(transactional_data['InvoiceDate'], errors='coerce')

# Remove rows where InvoiceDate or Quantity is NaN
transactional_data.dropna(subset=['InvoiceDate', 'Quantity'], inplace=True)

# Streamlit app layout
st.title("Demand Forecasting App")
st.write("Enter Stock Code and the Number of Weeks to Forecast")

# Input for Stock Code
stock_code = st.text_input("Enter Stock Code:")
periods = st.number_input("Number of Weeks to Forecast:", min_value=1, max_value=52, value=15)

if st.button("Forecast"):
    # Filter the data for the selected stock code
    stock_data = transactional_data[transactional_data['StockCode'] == stock_code]

    if stock_data.empty:
        st.error("No data found for the given Stock Code.")
    else:
        # Prepare the data for Prophet
        prophet_df = stock_data[['InvoiceDate', 'Quantity']].rename(columns={'InvoiceDate': 'ds', 'Quantity': 'y'})

        # Remove any rows with NaN in ds or y
        prophet_df.dropna(subset=['ds', 'y'], inplace=True)

        # Check the number of remaining rows
        if prophet_df.shape[0] < 2:
            st.error("Not enough data points for the selected Stock Code. At least 2 non-NaN rows are required.")
        else:
            # Fit the Prophet model
            model = Prophet()
            model.fit(prophet_df)

            # Create a DataFrame for future dates (15 weeks forecast)
            future = model.make_future_dataframe(periods=periods, freq='W')

            # Make predictions
            forecast = model.predict(future)

            # Plot the results
            fig, ax = plt.subplots()
            model.plot(forecast, ax=ax)
            plt.title(f'Demand Forecast for Stock Code: {stock_code}')
            plt.xlabel('Date')
            plt.ylabel('Quantity')
            st.pyplot(fig)

            # Show forecast data
            st.write("Forecast Data:")
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods))
