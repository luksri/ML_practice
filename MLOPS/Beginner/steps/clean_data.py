import logging
from zenml.steps import step
import pandas as pd

@step
def clean_data(df: pd.DataFrame) -> None:
    pass