# Canonical_Attrition_Prediction

Below I describe how to reproduce my results and generate the model.

## Links of interest

- [Analysis and results report:](./Data_Analysis)
- Description of the creation process:

## Model Deployment

The code for the creation of the model, I present it as a [Python file]("./Model_Training/model_training.py") and as a [Jupiter Notebook]("./Model_Training/Model_Pipeline_Training.ipynb"), both inside the [Model_Training]("./Model_Training") folder.

Within any of the files the `test` flag indicates whether accuracy tests will be performed on the model and the data will be divided into validation and test groups. If the flag is set to `False` the model will be trained with all available data.

The data file paths can be changed in case you want to train the model with other data, or new paths can be added if the final model is to be tested.

## Model Load

The [random_forest.joblib]("/Model_Training/random_forest.joblib") file contains the trained model with all the data. To import it you need to use the `load` method of `joblib`:

`loaded_rf = joblib.load(<route_to_model>)`

This model can be used to predict data that have been preprocessed in the same way as described in my solution.