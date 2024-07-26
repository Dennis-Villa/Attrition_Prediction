## This flag indicates that the model will be tested.
## To train the model with all data set it to False.
test = True

## Path of the data files. 
## Change the path if you are looking to add other datasets
from pathlib import Path
import os 

currPath = os.path.dirname(os.path.realpath(__file__))

# route_employee_survey = '../Model_Training/employee_survey_data.csv'
route_employee_survey = Path(currPath).joinpath('datasets').joinpath('employee_survey_data.csv')
route_general_data = Path(currPath).joinpath('datasets').joinpath('general_data.csv')
route_manager_survey = Path(currPath).joinpath('datasets').joinpath('manager_survey_data.csv')
route_in_time = Path(currPath).joinpath('datasets').joinpath('in_time.csv')
route_out_time = Path(currPath).joinpath('datasets').joinpath('out_time.csv')


################################################################

import pandas as pd
import numpy as np

# Transform data into DataFrame
employee_survey = pd.read_csv(route_employee_survey, index_col="EmployeeID")
general_data = pd.read_csv(route_general_data, index_col="EmployeeID")
manager_survey = pd.read_csv(route_manager_survey, index_col="EmployeeID")
in_time = pd.read_csv(route_in_time, index_col=0)
out_time = pd.read_csv(route_out_time, index_col=0)

# Set the index name to EmployeeID
in_time.index.name = 'EmployeeID'
out_time.index.name = 'EmployeeID'

# Merge regular data sources into one dataframe
data = employee_survey.merge(general_data, on='EmployeeID')
data = data.merge(manager_survey, on='EmployeeID')

# Add a higer index name for regular columns
regular_cols = data.columns
new_cols = []
for col in regular_cols:
  new_cols.append(("Regular", col))

data.columns = pd.MultiIndex.from_tuples(new_cols)

# Add a higer index name for time logs columns
time_cols = in_time.columns
new_in_cols = []
new_out_cols = []

for col in time_cols:
  new_in_cols.append(("InTime", col))
  new_out_cols.append(("OutTime", col))

in_time.columns = pd.MultiIndex.from_tuples(new_in_cols)
out_time.columns = pd.MultiIndex.from_tuples(new_out_cols)

# Merge time logs data sources into dataframe
data = data.merge(in_time, on='EmployeeID')
data = data.merge(out_time, on='EmployeeID')

# Drop rows with null values of attrition
data.Regular = data.Regular.dropna(subset=['Attrition'])

# Separate target from predictors
y = data.Regular.Attrition
X = data.drop([('Regular', 'Attrition')], axis=1)

# Select numerical columns
columns_numerical = X.Regular.select_dtypes(include=[np.number]).columns

# Select categorical columns
columns_categorical = X.Regular.select_dtypes(include=["object"]).columns

# For Ordinal Encoding
columns_ordinal = ["BusinessTravel", "MaritalStatus", "Gender", "Over18"]

#For OneHot Encoding
columns_one_hot = columns_categorical.drop(columns_ordinal)

# Select numerical columns except TotalWorkingYears
data_int_cols = columns_numerical.drop("TotalWorkingYears")


################################################################

## Preprocesing Steps for Data

from sklearn.impute import SimpleImputer

class WorkingYearsTransformer:
    def __init__(self):
        self.gen_imputer = SimpleImputer(strategy='median')

    def fit(self, X, Y=None):
        self.gen_imputer.fit(X[data_int_cols])

        return self

    def transform(self, df):
        transformed_df = df.copy()

        transformed_df[data_int_cols] = self.gen_imputer.transform(df[data_int_cols])

        # Substitution of nulls
        transformed_df.loc[transformed_df.TotalWorkingYears.isna(), 
                           "TotalWorkingYears"] = transformed_df.loc[transformed_df.TotalWorkingYears.isna(), 
                                                                     "YearsAtCompany"]

        return transformed_df

numerical_transformer = WorkingYearsTransformer()



# Imputer for categorical columns
cat_imputer = SimpleImputer(strategy='most_frequent')

from sklearn.preprocessing import OrdinalEncoder

# Encoder for marital status, travel frequency and over 18.
ordinal_encoder_travel = OrdinalEncoder(
    categories=[np.array(['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'])],
    handle_unknown='use_encoded_value',
    unknown_value=-1
    )
ordinal_encoder_marital = OrdinalEncoder(
    categories=[np.array(['Single', 'Married', 'Divorced'])],
    handle_unknown='use_encoded_value',
    unknown_value=-1
    )
ordinal_encoder_18 = OrdinalEncoder(
    categories=[np.array(['N', 'Y'])],
    handle_unknown='use_encoded_value',
    unknown_value=-1
    )

class Categorical_Ordinal_Transformer:
    def __init__(self):
        self.temp_imputer = SimpleImputer(strategy='most_frequent')
        self.ordinal_encoder_travel = ordinal_encoder_travel
        self.ordinal_encoder_marital = ordinal_encoder_marital
        self.ordinal_encoder_gender = OrdinalEncoder()
        self.ordinal_encoder_18 = ordinal_encoder_18

    def fit(self, X, Y=None):
        imputed_X = X[columns_ordinal].copy()

        self.temp_imputer.fit(imputed_X)

        imputed_X.iloc[:, :] = self.temp_imputer.transform(imputed_X)

        self.ordinal_encoder_travel.fit(imputed_X[["BusinessTravel"]])
        self.ordinal_encoder_marital.fit(imputed_X[["MaritalStatus"]])
        self.ordinal_encoder_gender.fit(imputed_X[["Gender"]])
        self.ordinal_encoder_18.fit(imputed_X[["Over18"]])

        return self

    def transform(self, df):
        transformed_df = df[columns_ordinal].copy()

        transformed_df.iloc[:, :] = self.temp_imputer.transform(transformed_df)

        transformed_df.loc[:, "BusinessTravel"] = self.ordinal_encoder_travel.transform(transformed_df[["BusinessTravel"]])
        transformed_df.loc[:, "MaritalStatus"] = self.ordinal_encoder_marital.transform(transformed_df[["MaritalStatus"]])
        transformed_df.loc[:, "Gender"] = self.ordinal_encoder_gender.transform(transformed_df[["Gender"]])
        transformed_df.loc[:, "Over18"] = self.ordinal_encoder_18.transform(transformed_df[["Over18"]])

        return transformed_df

categorical_ordinal_transformer = Categorical_Ordinal_Transformer()


from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

onehot_encoder = OneHotEncoder(
    handle_unknown='infrequent_if_exist',
    sparse_output=False,
    )

categorical_oneHot_transformer = Pipeline(steps=[
    ('imputer', cat_imputer),
    ('oneHot', onehot_encoder)
])


class Preprocesor_Transformer:
    def __init__(self):
        self.categorical_ordinal_transformer = categorical_ordinal_transformer
        self.categorical_oneHot_transformer = categorical_oneHot_transformer

    def fit(self, X, Y=None):
        ordinal_df = X[columns_ordinal]
        oneHot_df = X[columns_one_hot]

        self.categorical_ordinal_transformer = self.categorical_ordinal_transformer.fit(ordinal_df)
        self.categorical_oneHot_transformer = self.categorical_oneHot_transformer.fit(oneHot_df)

        return self

    def transform(self, df):
        transformed_df = df.copy()

        transformed_df.loc[:, columns_ordinal] = self.categorical_ordinal_transformer.transform(transformed_df[columns_ordinal])

        hot_encoded_data = self.categorical_oneHot_transformer.transform(transformed_df[columns_one_hot])
        hot_encoded_columns = categorical_oneHot_transformer.get_feature_names_out()

        hot_encoded_df = pd.DataFrame(hot_encoded_data, columns=hot_encoded_columns, index=transformed_df.index)
        transformed_df = pd.concat([transformed_df[columns_ordinal], hot_encoded_df], axis=1)

        return transformed_df

categorical_transformer = Preprocesor_Transformer()



class Cat_Num_Separator:
    def __init__(self):
        self.numerical_transformer = numerical_transformer
        self.categorical_transformer = categorical_transformer

    def fit(self, X, Y=None):
        num_df = X[columns_numerical]
        cat_df = X[columns_categorical]

        self.numerical_transformer = self.numerical_transformer.fit(num_df)
        self.categorical_transformer = self.categorical_transformer.fit(cat_df)

        return self

    def transform(self, df):
        num_vals = self.numerical_transformer.transform(df[columns_numerical])
        cat_vals = self.categorical_transformer.transform(df[columns_categorical])

        return num_vals.merge(cat_vals, on="EmployeeID")

catNumSeparator = Cat_Num_Separator()



from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Scale_Reduce:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None

    def fit(self, df, Y=None):
        self.pca = PCA(n_components=(df.shape[1]-1))

        self.scaler.fit(df)
        scaled_df = pd.DataFrame(self.scaler.transform(df), columns=df.columns)

        self.pca.fit(scaled_df)

        return self

    def transform(self, df):

        return self.pca.transform(df)

scaleReduce = Scale_Reduce()

def workLogsData(df_time_logs):
  # Determines the number of days the worker was absent.
  working_days_out_df = pd.DataFrame(df_time_logs.InTime.isna().sum(axis=1), columns=['WorkingDaysOut'])

  # Transforms data to datetime type.
  in_time_datetimes = df_time_logs.InTime.apply(pd.to_datetime)
  out_time_datetimes = df_time_logs.OutTime.apply(pd.to_datetime)

  # Calculates the time between entry and exit.
  time_worked_datedif = out_time_datetimes - in_time_datetimes

  # Determine the hours worked each day.
  time_worked_hours_df = time_worked_datedif.map(lambda x: x.seconds / 3600)

  # Calculates the average hours worked.
  prom_day_hours_df = pd.DataFrame(time_worked_hours_df.apply(np.mean, axis=1), columns=['PromDayHours'])

  return prom_day_hours_df.merge(working_days_out_df, on='EmployeeID')


columns_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'DistanceFromHome',
                   'Education', 'JobLevel', 'StockOptionLevel', 'JobInvolvement',
                   'JobRole_Laboratory Technician', 'EducationField_Marketing',
                   'EducationField_Medical', 'JobRole_Healthcare Representative',
                   'JobRole_Human Resources', 'JobRole_Manager', 'JobRole_Sales Executive',
                   'JobRole_Sales Representative', 'Gender', 'Department_Research & Development',
                   'EducationField_Life Sciences']

class Preprocesor_Transformer:
    def __init__(self):
        self.separator = Cat_Num_Separator()
        self.pca_1 = Scale_Reduce()
        self.pca_2 = Scale_Reduce()
        self.pca_3 = Scale_Reduce()

    def fit(self, X, Y=None):
        self.separator.fit(X.Regular)
        procesed_df = self.separator.transform(X.Regular)

        # Columns that will reduce their dimensionality using PCA
        self.pca_1.fit(procesed_df[["PerformanceRating", "PercentSalaryHike"]])
        self.pca_2.fit(procesed_df[["TotalWorkingYears", "YearsAtCompany", "YearsSinceLastPromotion",
                                    "YearsWithCurrManager", "Age"]])
        self.pca_3.fit(procesed_df[["Department_Human Resources", "EducationField_Human Resources"]])

        return self

    def transform(self, df):
        procesed_df = self.separator.transform(df.Regular)

        pca_1 = self.pca_1.transform(procesed_df[["PerformanceRating", "PercentSalaryHike"]])
        pca_2 = self.pca_2.transform(procesed_df[["TotalWorkingYears", "YearsAtCompany",
                                                  "YearsSinceLastPromotion",  "YearsWithCurrManager",
                                                  "Age"]])
        pca_3 = self.pca_3.transform(procesed_df[["Department_Human Resources", "EducationField_Human Resources"]])

        # Removes unnecessary columns from the DataFrame.
        processed_df = procesed_df.drop(columns_to_drop, axis=1)

        # Removes reduced columns from the DataFrame.
        processed_df = processed_df.drop(["PerformanceRating", "PercentSalaryHike", "TotalWorkingYears",
                                          "YearsAtCompany", "YearsSinceLastPromotion", "YearsWithCurrManager",
                                          "Age", "Department_Human Resources", "EducationField_Human Resources"],
                                         axis=1)

        # Transform the in_time and out_time data
        work_logs = workLogsData(df)

        return np.concatenate((processed_df, work_logs, pca_1, pca_2, pca_3), axis=1)


preprocessor = Preprocesor_Transformer()


################################################################

## Model Training

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 300, max_features=0.4, max_depth=15,
                                 min_samples_leaf=2, random_state=0)

aplication = Pipeline(steps=[
    ("preprocesor", preprocessor),
    ("model", model, )
])


###################################
# Test Models

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if test:
  X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size= 0.80, random_state=0);
  aplication.fit(X_train, Y_train)
  predicts = aplication.predict(X_test)

  print(classification_report(Y_test, predicts))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

if test:
  conf_mat = confusion_matrix(Y_test, predicts)

  plt.figure(figsize=(6,4))
  sns.heatmap(conf_mat, annot=True, cmap=plt.cm.Greens, linewidths=0.2, fmt='g')

  # Add labels to the plot
  class_names = ["0", "1"]
  tick_marks = np.arange(len(class_names))
  tick_marks2 = tick_marks + 0.5
  plt.xticks(tick_marks2, class_names, rotation=0)
  plt.yticks(tick_marks2, class_names, rotation=0)
  plt.xlabel('Predicted label')
  plt.ylabel('True label')
  plt.title('Confusion Matrix for Random Forest Model')
  plt.show()


if test:

  from sklearn.model_selection import cross_val_score
  from sklearn.metrics import make_scorer, f1_score

  accuracy_scores = cross_val_score(aplication, X, y,
                                cv=5,
                                scoring='accuracy'
                          )
  f1_scores = cross_val_score(aplication, X, y,
                                cv=5,
                                scoring=make_scorer(f1_score, greater_is_better=True, pos_label='Yes')
                          )

  print("Accuracy Score:\n", accuracy_scores)
  print("F1 Score:\n", f1_scores)


###################################
# Prediction Model

if not test:
  aplication.fit(X, y)