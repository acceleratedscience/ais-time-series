import logging
import math
from typing import Dict, List

import litserve as ls
import pandas as pd
import torch
from litserve.mcp import MCP
from pydantic import BaseModel, Field
from tsfm_public import TinyTimeMixerForPrediction, TimeSeriesForecastingPipeline

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("TIME_SERIES_APP")


# Define the request schema using Pydantic
class TimeSeriesRequest(BaseModel):
    data: Dict[str, List] = Field(
        ...,
        description="Time-series data in a dictionary format, e.g., {'timestamp': [...], 'value1': [...], 'value2': [...]}",
    )
    timestamp_col: str = Field(..., description="Timestamp column name.")
    target_cols: List[str] = Field(..., description="List of target columns.")
    context_length: int = Field(
        512, description="Number of past timesteps to use as input."
    )
    prediction_length: int = Field(96, description="The forecast horizon.")


class TimeSeriesLitAPI(ls.LitAPI):
    """A LitServe API for time-series forecasting using ibm-granite/granite-timeseries-ttm-r2."""

    def _clean_nans(self, obj):
        if isinstance(obj, list):
            return [self._clean_nans(item) for item in obj]
        if isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    def setup(self, device):
        """
        Set up the model. This is called once for each worker process.
        """
        logger.info("Setting up the time-series model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "ibm-granite/granite-timeseries-ttm-r2"
        # Instantiate the model.
        self.model = TinyTimeMixerForPrediction.from_pretrained(
            model_name, num_input_channels=1
        )
        logger.info("Model setup complete.")

    async def decode_request(self, request: TimeSeriesRequest, **kwargs):
        """
        Converts the raw request into a pandas DataFrame.
        """
        df = pd.DataFrame(request.data)
        df[request.timestamp_col] = pd.to_datetime(df[request.timestamp_col])
        df = df.sort_values(request.timestamp_col).reset_index(drop=True)
        # print("Decoded DataFrame head:\n", df.head())
        return {
            "df": df,
            "timestamp_col": request.timestamp_col,
            "target_cols": request.target_cols,
            "context_length": request.context_length,
            "prediction_length": request.prediction_length,
        }

    async def predict(self, x: dict, **kwargs):
        """
        Perform inference on the decoded request.
        """
        print("\n==================== Starting prediction ====================\n")
        df = x["df"]
        timestamp_col = x["timestamp_col"]
        target_cols = x["target_cols"]
        context_length = x["context_length"]
        prediction_length = x["prediction_length"]

        # Create a pipeline.
        pipeline = TimeSeriesForecastingPipeline(
            model=self.model,
            timestamp_column=timestamp_col,
            id_columns=[],
            target_columns=target_cols,
            explode_forecasts=False,
            freq="h",
            device=self.device,
        )

        if len(df) < context_length:
            raise ValueError(
                f"Input data must contain at least {context_length} rows for context window."
            )

        window = df.iloc[-context_length:]
        forecast_df = pipeline.predict(window)
        if isinstance(forecast_df, pd.DataFrame):
            # print("Forecast columns:", forecast_df.columns.tolist())
            print("Forecast result:\n", forecast_df.tail())
        else:
            print("Forecast result:\n", forecast_df)
        return {"forecast_df": forecast_df, "timestamp_col": timestamp_col}

    async def encode_response(self, output: dict, **kwargs):
        """
        Converts the prediction output (DataFrame) into a JSON serializable format.
        """
        forecast_df = output["forecast_df"]
        timestamp_col = output["timestamp_col"]
        # Replace non-finite values (NaN, inf) with None for JSON compatibility
        forecast_df[timestamp_col] = forecast_df[timestamp_col].dt.strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        records = forecast_df.to_dict(orient="records")
        cleaned_records = [
            {
                col: self._clean_nans(value)
                for col, value in row.items()
            }
            for row in records
        ]

        return {"prediction": cleaned_records}

if __name__ == "__main__":
    logger.info("Initializing time-series LitServe server...")

    api = TimeSeriesLitAPI(
        enable_async=True,
        mcp=MCP(
            name="TimeSeriesForecaster",
            description="A time-series forecasting service using ibm-granite/granite-timeseries-ttm-r2.",
        )
    )

    server = ls.LitServer(api, accelerator="auto")
    server.run(port=8081, generate_client_file=False)
