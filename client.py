import pandas as pd
import requests
from datasets import load_dataset
import json

# 1. Load the dataset from Hugging Face Hub
print("📥 Loading dataset from Hugging Face Hub...")
dataset = load_dataset("vitaliy-sharandin/energy-consumption-hourly-spain")
df = dataset["train"].to_pandas()
# print("Columns found in dataset:", df.columns.tolist())

# 2. Prepare the data
print("📊 Preparing data for request...")
# Use 'time' column as the timestamp, converting to UTC to handle mixed timezones.
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.sort_values("time").reset_index(drop=True)

# Select target and context
target_col = "total load actual"
context_length = 512
prediction_length = 96
# The server now handles data cleaning and windowing.
# We send a larger chunk of data and let the server select the context window.
# For demonstration, we'll send the last 1000 data points.
context_df = df.iloc[-1000:]

# Format data for the API request
request_data = {
    "data": {
        "time": context_df["time"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist(),
        target_col: context_df[target_col].tolist(),
    },
    "timestamp_col": "time",
    "target_cols": [target_col],
    "context_length": context_length,
    "prediction_length": prediction_length,
    "freq": "h",
}

# 3. Send request to the LitServe API
api_url = "http://localhost:8081/predict"
print(f"🚀 Sending request to {api_url}...")
try:
    # print(request_data)
    response = requests.post(api_url, json=request_data)
    response.raise_for_status()  # Raise an exception for bad status codes
    prediction = response.json()
    print("✅ Forecast received successfully:")
    # Print the first 5 predicted values for brevity
    predicted_values = prediction.get("prediction", [])
    print("Forecast result :\n", pd.DataFrame(predicted_values).head())
    # write prediction to a json file
    with open("forecast_output.json", "w") as f:
        json.dump(prediction, f, indent=2)
        print("Forecast saved to 'forecast_output.json'.")

except requests.exceptions.RequestException as e:
    print(f"🔥 An error occurred: {e}")
    print("Please ensure the LitServe server in 'main.py' is running.")
