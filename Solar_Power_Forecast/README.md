# Solar Irradiance Forecasting using NREL API & Machine Learning

This project forecasts **Global Horizontal Irradiance (GHI)** one hour into the future using historical solar and weather data retrieved from the **NREL NSRDB API**. It uses a **Random Forest Regressor** model trained on environmental attributes like DNI, DHI, air temperature, wind speed, and relative humidity.

---

## Features

- Fetch real solar data using [NREL NSRDB API](https://developer.nrel.gov/docs/solar/nsrdb/psm3-download/)
- Forecast GHI 1 hour ahead with a machine learning model
- Visualize predicted vs actual GHI
- Support for 30-minute interval data
- Forecast specific datetime values

---

## Sample Output

```text
Requesting solar data for lat=41.5834, lon=-87.5, year=2020...
CSV Download complete.
Returned filename: solar_data.csv
DataFrame loaded. Columns: ['ghi', 'dni', 'dhi', 'air_temperature', 'wind_speed', 'relative_humidity']
Mean Absolute Error (MAE): 46.27
Forecasted GHI for 2020-11-05 11:00: 287.56

---

## Tech Stack

- Python 3
- pandas, numpy, requests
- scikit-learn
- matplotlib
- NREL NSRDB API
- dotenv for API key management

---

## Project Structure
solar-forecast-project/
│
├── solar_forecast.ipynb  # Notebook version of the pipeline
├── requirements.txt      # Required dependencies
├── .env.example          # Sample environment file
└── README.md             # Project documentation
