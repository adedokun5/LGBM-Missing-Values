#LGBM Missing Values
#  -Using light gradient boosting to handle missing values


1) Implementation

The methodology involved the following steps:

1.1 Importing Necessary Libraries
   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import seaborn as sns
   from lightgbm import LGBMRegressor, LGBMClassifier
   ```

1.2 Read and Inspect Dataset 
   ```python
   df = pd.read_csv('sample_data/kidney_disease.csv')
   df.head()
   ```

1.3 Display the dimensions of the DataFrame (used to quickly check the size of a dataset)
   ```python
   df.shape
   ```

1.4 Get DataFrame Summary
   ```python
   df.info()
   ```
   provides a concise summary of the DataFrame df. It outputs key information about the structure and content of the dataset, which is helpful for understanding its characteristics before further analysis.

1.5 Get descriptive statistics for the numerical columns in the DataFrame df
   ```python
   df.describe()
   ```

1.6 Check the number of missing (NaN) values in each column of the DataFrame df
   ```python
   df.isna().sum()
   ```
   It returns the count of missing values for each column, helping to identify which features have incomplete data.

1.7 Drop id column
   ```python
   df.drop('id', axis=1, inplace=True)
   ```
   The id column has been removed from the dataset as it does not contribute to the analysis for this experiment.

1.8 Make copies of DataFrame
   ```python
   new_df = df.copy()
   pred_df = df.copy()
   new_df.head()
   ```
   The code new_df = df.copy() and pred_df = df.copy() creates two copies of the original DataFrame df. Both new_df and pred_df are independent of df, meaning that any changes made to new_df or pred_df will not affect the original DataFrame, and vice versa.
   The pred_df will be utilized for predicting the missing (NaN) values, with the predicted values being assigned to their corresponding indices in the new_df.

1.9 Identify categorical and continuous columns.
   ```python
   categorical_columns = new_df.select_dtypes(include=['object']).columns
   continuous_columns = new_df.select_dtypes(include=['int64', 'float64']).columns
   ```

   Categorical columns: are of data type object, which typically refers to categorical variables (e.g., strings, labels).
   Continuous columns: are of numerical data types (int and float), which are typically used for continuous variables.

1.10 Convert categorical columns into numeric representations
   ```python
   def convert_to_numeric( continuous_df, categorical_columns, categorical, pred_col ):
      # Iterate through the list of categorical columns
      for column in categorical_columns:
         # Check if the column is the one to be predicted and is categorical
         if column == pred_col and categorical == True:
         continue # Skip this column if it is the prediction column
         else:
         # Convert the categorical column to numeric using pd.factorize
         continuous_df[ column ] = pd.factorize( continuous_df[ column ] )[ 0 ]

      return continuous_df
   ```
   Parameters:
   continuous_df: The DataFrame containing both categorical and continuous columns.
   categorical_columns: A list of columns that are identified as categorical.
   categorical: A boolean flag indicating whether a categorical column is being predicted (True = predicting a categorical column, False = predicting a continuous column).
   pred_col: The column whose missing values are to be predicted (this column is excluded from the conversion if we are trying to predict missing values for a categorical feature).

   Function Logic:
   Iterate Through Categorical Columns: The function loops over all categorical columns provided in categorical_columns.
   Skip the Prediction Column: If the column is the one specified in pred_col (the column to be predicted for missing values), it is excluded from conversion.
   Convert Categorical to Numeric: For all other columns, the pd.factorize() method is used to convert the categorical data into numeric labels. pd.factorize() assigns a unique integer to each unique category in the column, and [0] selects the integer values (ignoring the index returned by factorize()).
   Return the Modified DataFrame: The function returns the updated DataFrame with the categorical columns converted to numeric values.

1.10 Model selection based on the type of prediction task
   ```python
   def set_model_type( categorical ):
      if categorical == True:
         # Predicting a categorical column
         return LGBMClassifier()
      else:
         # Predicting a continuous column
         return LGBMRegressor() # Returns LGBMRegressor
   ```

   The function set_model_type is designed to choose an appropriate LightGBM model based on the data type (whether the target variable is categorical or continuous). 
   Parameters:
   categorical: A boolean flag that indicates whether the target variable is categorical (True) or continuous (False).

   Logic:
   If categorical == True: The function returns an instance of LGBMClassifier(i.e., when the target variable is categorical).
   If categorical == False: The function returns an instance of LGBMRegressor(i.e., when the target variable is continuous).

1.11 Predict Missing Values

   ```python
   def predict_missing_values(pred_df, columns, new_df, categorical_data, categorical_columns ):
      for col in columns:

         # If more than 50% of the column data is missing, drop the column
         if pred_df[col].isna().sum() > pred_df.shape[ 0 ] / 1:
            pred_df.drop(col, axis=1, inplace=True)
         else:
            col_missing_index = np.where( pred_df[col].isna() == True )[0]

            # If no missing values in the column, continue to the next one
            if len( col_missing_index ) == 0:
               continue
            else:
               # Create an 'is_nan' column to mark missing values
               pred_df['is_nan'] = 0

               pred_df.loc[col_missing_index, 'is_nan'] = 1

               continuous_df = pred_df.copy()

               continuous_df = convert_to_numeric( continuous_df, categorical_columns, categorical_data, col )

               # Split data into train (non-missing) and test (missing) sets
               train = continuous_df[ continuous_df['is_nan'] == 0 ]
               test = continuous_df[ continuous_df['is_nan'] == 1 ]

               X_train = train.drop([col, 'is_nan'], axis=1)
               y_train = train[col]

               X_test = test.drop([col, 'is_nan'], axis=1)

               # Select the appropriate model based on the type of prediction (categorical or continuous)
               lgbm = set_model_type( categorical_data )
               lgbm.fit(X_train, y_train)

               # Predict missing values using the model
               y_pred = lgbm.predict(X_test)

               # Assign the predicted values to the missing entries in the original dataframe
               new_df.loc[col_missing_index, col] = y_pred
   ```

   The function predict_missing_values is designed to handle missing values in a dataset by predicting them using a machine learning model. This approach involves training a model on non-missing data and using the trained model to predict the missing values in the dataset.
   Function Parameters:
   pred_df: The dataframe containing the data with missing values.
   columns: List of columns to predict missing values for.
   new_df: The dataframe where the predicted values will be stored.
   categorical_data: A boolean flag indicating whether the target column is categorical (True) or continuous (False).
   categorical_columns: List of columns that are categorical.

   Function Logic:
   Missing Value Threshold:

   If more than 50% of the data in a column is missing, the column is dropped from the dataset.

   2. Handling Missing Data:
   For each column, the missing values are identified.
   An is_nan column is created, marking rows with missing values as 1 and others as 0.

   3. Data Conversion:
   The data is converted to numeric format using the convert_to_numeric function to handle categorical data.

   4. Data Splitting:
   The dataset is split into two parts:
   Train Data: Contains rows with non-missing values for the target column.
   Test Data: Contains rows with missing values for the target column.

   5. Model Training:
   The appropriate model (either LGBMClassifier or LGBMRegressor) is selected using the set_model_type function based on whether the target column is categorical or continuous.
   The model is trained using the train data.

   6. Prediction:
   The trained model is used to predict the missing values in the test data (rows with missing values for the target column).

   7. Updating the DataFrame:
   The predicted values are assigned to the missing entries in the new_df dataframe.

1.12 Predict missing values for continuous features

   ```python
   predict_missing_values( pred_df, continuous_columns, new_df, False, categorical_columns )
   ```

   By running this code, the missing continuous values in pred_df are predicted and stored in new_df, ensuring that the imputed values are based on the patterns observed in the available data.

1.13 Predict missing values for categorical features

   ```python
   predict_missing_values( pred_df, categorical_columns, new_df, True, categorical_columns )
   ```

   By running this code, the missing categorical values in pred_df are predicted and stored in new_df, ensuring that the imputed values are based on the patterns observed in the available data.

2) Results
Upon executing the above code, the results can be verified by running the following blocks of code

2.1 Display first few rows
   ```python
   new_df.head()
   ```

   Display the first few rows of the new_df dataframe, allowing you to verify that the missing values in the dataset have been successfully imputed. This function helps to quickly inspect the changes made to the data, ensuring that the predicted values are correctly filled in the columns with missing entries.

3.2 Display dimension of the DataFrame
   ```python
   new_df.shape
   ```

   To confirm the integrity of the dataset after imputation, the dimensions of the new_df dataframe can be checked using the command new_df.shape. This will ensure that the number of rows and columns remain consistent after handling missing values.

3.3 Get DataFrame Summary
   ```python
   new_df.describe()
   ```

   To examine the statistical summary of the dataset after imputation, the command new_df.describe() can be used. This will provide key summary statistics, including measures of central tendency and spread, for each numerical feature in the new_df dataframe, helping to evaluate the overall impact of the imputation process.

3.4 Check the number of missing (NaN) values in each column of the DataFrame
   n```python
   ew_df.isna().sum()
   ```

   To verify the success of the imputation process, the command new_df.isna().sum() can be executed. This will provide the count of missing values for each feature, helping to confirm that all missing values have been imputed and that no NaN values remain in the dataset.


To gain a deeper understanding and enhance your reading experience, feel free to explore more in my Medium article by clicking [Here](https://medium.com/@adedokunjuliusayobami/handling-missing-values-with-light-gbm-4a222d8af31b)