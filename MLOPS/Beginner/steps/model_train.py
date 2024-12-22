import logging
from zenml.steps import step
import pandas as pd

@step
def train_model(df: pd.DataFrame) -> None:
    pass