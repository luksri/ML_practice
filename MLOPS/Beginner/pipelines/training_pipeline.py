import logging
from zenml.pipelines import pipeline
from zenml.steps import step
import pandas as pd
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import train_model
from steps.evaluation import evaluate_model


@step
def data_path_provider() -> str:
    return "/Users/lakshmanv/PycharmProjects/ML/ML_practice/MLOPS/Beginner/data/train.csv"


@pipeline
def training_pipeline(data_path):
    df = ingest_data(data_path)
    clean_data(df)
    train_model(df)
    evaluate_model(df)
