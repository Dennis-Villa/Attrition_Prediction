# Summary of Development Steps

## [In and Out Logs Data Transformation](./Notebooks/In_Out_Data_Function.ipynb)

My first step in this project was to analyze the data sets presented to me. The other datasets had their structure ready to be used in a DataFrame, but `in_data` and `out_data` needed previous steps in order to get relevant information from them.

After my analysis I determined that the best information I could extract was the number of days each employee missed from work, as well as the average daily and weekly hours worked. I structured this data as a pandas DataFrame using a function for this purpose.

## [Imputation and Encoding of the Data](./Notebooks/Imputation_Encoding.ipynb)

The rest of the datasets, not including `in_data` and `out_data` were ready to be processed. The first thing would be to determine the columns of numerical type and those of categorical type because they have a differentiated treatment.

In the case of numerical columns, use for the null imputation the median value of the column, except for `TotalWorkingYears`. For this one I decided to use the value of `YearsAtCompany` in the cases where its value was null.

The categorical columns I imputed by replacing them with the most frequent value. To encode them into numerical values I had to separate them in turn depending on whether their values were ordinal or not, using respectively an `OrdinalEncoder` or an `OneHotEncoder`.

To automate these processes I use Pipelines in conjunction with transformers created by me to have more flexibility and avoid losing the names of the columns of the Dataframe.

## [In-depth data analysis](../Data_Analysis/Data_Analysis.ipynb)

Having my functionalities ready to transform the data, I performed a deep analysis. The details of this analysis are in the [Analysis and Results Report](../Data_Analysis/README.md).

As can be seen in the correlation matrices section, many of the columns have a high level of correlation with each other, which can affect the model by causing overfitting.

To solve the problem, I applied a [Principal Component Analysis](./Notebooks/Dimensionality_Reduction.ipynb) (PCA) algorithm to reduce the dimensionality and correlation of the data.

![Correlation Matrix](<./Correlation_Matrix.png>)

As can be observed, the PCA algorithm reduced the correlation of the columns, ensuring that all had less than 0.4 correlation with each other.

## [Model Selection and Fitting](../Model_Training/Model_Pipeline_Training.ipynb)

With the previous steps, the data processing was ready, and I could begin with model selection. After several tests with different classification models and searches on Kaggle and Google Scholar, I decided on a `RandomForestClassifier` model.

Once the model was selected, I proceeded to tune the parameters, aiming for the best accuracy without overfitting the input data. The final adjustment of the model was `RandomForestClassifier(n_estimators = 300, max_features=0.4, max_depth=15, min_samples_leaf=2, random_state=0)`.

This model is capable of predicting positive attrition values with an accuracy greater than 80% and has been tuned in a way that it should respond correctly to new data different from the training set.