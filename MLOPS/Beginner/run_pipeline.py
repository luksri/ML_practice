from pipelines.training_pipeline import training_pipeline, data_path_provider

if __name__ == "__main__":
    data_path_step = data_path_provider()
    training_pipeline(data_path=data_path_step)

    # training_pipeline(data_loader=load_data.with_options(parameters={"data_path": "/Users/lakshmanv/PycharmProjects/ML/ML_practice/MLOPS/Beginner/data/example.csv"}))
