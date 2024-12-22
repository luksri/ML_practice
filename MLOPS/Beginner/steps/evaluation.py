import logging
from zenml.steps import step
import pandas as pd

@step
def evaluate_model(df: pd.DataFrame) -> None:
    pass