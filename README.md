# AIS Time Series

IBM Time Series Analysis Suite.

This project provides a time-series forecasting service using the `ibm-granite/granite-timeseries-ttm-r2` model, served with [LitServe](https://github.com/Lightning-AI/litserve). View hugging face model card [here](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2).

## Features

- Exposes a simple REST API for time-series forecasting.
- Utilizes the `ibm-granite/granite-timeseries-ttm-r2` model for predictions.
- Built on the high-performance LitServe framework for serving machine learning models.
- Asynchronous API for efficient handling of requests.
- Request and response validation using Pydantic.

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended for environment management)

The required Python packages are listed in `pyproject.toml`.

## Getting Started

### Installation

1.  Clone the repository:
    ```bash
    git clone <your-repository-url>
    cd ais-time-series
    ```

2.  Create a virtual environment and install the dependencies:

    **Using `uv` (recommended):**
    ```bash
    uv venv
    source .venv/bin/activate
    uv sync
    ```

    **Using `pip` and `venv`:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -e .
    ```

### Running the Server

To start the forecasting server, run the following command from the project's root directory:

```bash
python main.py
```

The server will start on `http://127.0.0.1:8081`.

### Running the Example Client

> The client is based off the community notebook [Time_Series_Getting_Started](https://github.com/ibm-granite-community/granite-timeseries-cookbook/blob/main/recipes/Time_Series/Time_Series_Getting_Started.ipynb)

In a separate terminal, you can run the example client to send a sample request to the running server:

```bash
python client.py
```

## API Usage

The server is designed to be robust, handling data cleaning (forward/backward fill for missing values) and validation internally. You can send a `POST` request to the default `/predict` endpoint to get a forecast.

**Method:** `POST`

**URL:** `http://127.0.0.1:8081/predict`

### Request Body

The request body must be a JSON object with the following structure:

```json
{
  "data": {
    "timestamp": ["2023-01-01T00:00:00", "..."],
    "value1": [10.1, "..."],
    "value2": [20.5, "..."]
  },
  "timestamp_col": "timestamp",
  "target_cols": ["value1", "value2"],
  "freq": "h",
  "context_length": 512,
  "prediction_length": 96
}
```

-   `data`: A dictionary containing lists of equal length. One list must correspond to the `timestamp_col`.
-   `timestamp_col`: The name of the key in `data` that holds the timestamps.
-   `target_cols`: A list of keys in `data` to be forecasted.
-   `freq`: The frequency of the time series data, expressed as a pandas frequency string (e.g., `"h"` for hourly, `"D"` for daily). Defaults to `"h"`.
-   `context_length`: The number of past time steps to use as input for the model. The number of entries in your data must be at least this large.
-   `prediction_length`: The number of future time steps to forecast (the forecast horizon).

### Example with `curl`

Here is an example of how to send a request using `curl`. Note that `context_length` is set to `8`, so we provide 10 data points.

```bash
curl -X POST http://127.0.0.1:8081/predict \
-H "Content-Type: application/json" \
-d '{
    "data": {
        "timestamp": [
            "2023-01-01T00:00:00", "2023-01-01T01:00:00", "2023-01-01T02:00:00",
            "2023-01-01T03:00:00", "2023-01-01T04:00:00", "2023-01-01T05:00:00",
            "2023-01-01T06:00:00", "2023-01-01T07:00:00", "2023-01-01T08:00:00",
            "2023-01-01T09:00:00"
        ],
        "value1": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    },
    "timestamp_col": "timestamp",
    "target_cols": ["value1"],
    "freq": "h",
    "context_length": 8,
    "prediction_length": 2
}'
```

### Response

The API will return a JSON object containing the forecast.

```json
{
  "prediction": [
    {
      "timestamp": "2023-01-01T10:00:00",
      "value1": 20.123
    },
    {
      "timestamp": "2023-01-01T11:00:00",
      "value1": 21.456
    }
  ]
}
```
*(Note: The prediction values above are illustrative and not the actual model output.)*