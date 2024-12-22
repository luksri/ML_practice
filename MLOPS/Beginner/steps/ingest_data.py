import  logging

import pandas as pd
from zenml.steps import step

class IngestData:
    def __int__(self, data_path):
        self.data_path = data_path

    def get_data(self):
        logging.info(f"ingesting data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_data(data_path) -> pd.DataFrame:
    """
    ingesting the data from data path
    :type data_path: object
    :param data_path:
    :return:
    """
    try:
        # ingest_data = IngestData(data_path)
        # df = ingest_data.get_data()
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        logging.error(f"error while reading the data {e}")
        raise e