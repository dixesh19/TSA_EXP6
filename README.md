# Ex.No: 6               HOLT WINTERS METHOD
### Date: 16-05-2026



### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd

file_path = "C:/Users/admin/OneDrive/Desktop/index_1.csv"

data = pd.read_csv(file_path)

data['datetime'] = pd.to_datetime(
    data['date'] + ' ' + data['datetime'],
    errors='coerce'
)

data = data.dropna(subset=['datetime'])

data = data.set_index('datetime')

data['money'] = pd.to_numeric(
    data['money'],
    errors='coerce'
)

data = data.dropna(subset=['money'])


monthly_data = data['money'].resample('MS').mean()

monthly_data = monthly_data.dropna()

train_size = int(0.9 * len(monthly_data))

train_data = monthly_data[:train_size]

test_data = monthly_data[train_size:]

fitted_model = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='add',
    seasonal_periods=4
).fit()

test_predictions = fitted_model.forecast(
    len(test_data)
)

plt.figure(figsize=(12, 8))

train_data.plot(
    legend=True,
    label='Train Data'
)

test_data.plot(
    legend=True,
    label='Test Data'
)

test_predictions.plot(
    legend=True,
    label='Predicted Data'
)

plt.title(
    'Train, Test and Predicted Coffee Sales using Holt-Winters'
)

plt.xlabel('Date')

plt.ylabel('Coffee Sales')

plt.show()

mae = mean_absolute_error(
    test_data,
    test_predictions
)

mse = mean_squared_error(
    test_data,
    test_predictions
)

print(f"Mean Absolute Error = {mae:.4f}")

print(f"Mean Squared Error = {mse:.4f}")

final_model = ExponentialSmoothing(
    monthly_data,
    trend='add',
    seasonal='add',
    seasonal_periods=4
).fit()

forecast_predictions = final_model.forecast(
    steps=12
)
plt.figure(figsize=(12, 8))

monthly_data.plot(
    legend=True,
    label='Original Data'
)

forecast_predictions.plot(
    legend=True,
    label='Forecasted Data',
    color='red'
)

plt.title(
    'Original and Forecasted Coffee Sales using Holt-Winters'
)

plt.xlabel('Date')

plt.ylabel('Coffee Sales')

plt.show()
```
### OUTPUT:


TEST_PREDICTION

<img width="1063" height="752" alt="image" src="https://github.com/user-attachments/assets/3798fc99-cf5c-4c17-aba1-f9bc76b7ba0d" />


FINAL_PREDICTION

<img width="1042" height="695" alt="image" src="https://github.com/user-attachments/assets/2a577686-7b68-4cd7-8837-8dc264ab2b80" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
