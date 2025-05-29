import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def report_missing_values(df: pd.DataFrame):
    """
    Prints a report of missing values per column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    """
    print("Missing values report:")
    missing_counts = df.isnull().sum()
    missing_percentages = (df.isnull().sum() / len(df)) * 100
    missing_report = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Percentage': missing_percentages
    })
    print(missing_report[missing_report['Missing Count'] > 0])
    if missing_report[missing_report['Missing Count'] > 0].empty:
        print("No missing values found.")

def impute_numerical(df: pd.DataFrame, column_name: str, strategy: str = 'mean') -> pd.DataFrame:
    """
    Imputes missing values in a numerical column using the specified strategy.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the numerical column to impute.
    strategy (str): The imputation strategy ('mean' or 'median'). Default is 'mean'.

    Returns:
    pd.DataFrame: The DataFrame with the imputed column.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        raise TypeError(f"Column '{column_name}' is not numerical. Imputation strategy '{strategy}' can only be applied to numerical columns.")
    if strategy not in ['mean', 'median']:
        raise ValueError(f"Invalid imputation strategy '{strategy}'. Choose 'mean' or 'median'.")

    imputer = SimpleImputer(strategy=strategy)
    df[column_name] = imputer.fit_transform(df[[column_name]])
    print(f"Imputed missing values in '{column_name}' using strategy '{strategy}'.")
    return df

def encode_categorical(df: pd.DataFrame, column_name: str, prefix: str = None) -> pd.DataFrame:
    """
    Performs one-hot encoding on a specified categorical column using pandas.get_dummies.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the categorical column to encode.
    prefix (str, optional): Prefix for new column names. Defaults to the original column name.

    Returns:
    pd.DataFrame: The DataFrame with the original column dropped and new encoded columns added.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")
    if pd.api.types.is_numeric_dtype(df[column_name]):
        # Allow encoding if column is numerical but clearly categorical (e.g. integer codes)
        # However, user should be aware. For now, let's proceed but a warning could be added.
        print(f"Warning: Column '{column_name}' is numerical. Proceeding with one-hot encoding. Ensure this is intended for categorical representation.")

    if prefix is None:
        prefix = column_name
        
    encoded_df = pd.get_dummies(df[column_name], prefix=prefix, dummy_na=False) # dummy_na=False to not create a column for NaN
    df = pd.concat([df.drop(columns=[column_name]), encoded_df], axis=1)
    print(f"One-hot encoded column '{column_name}' with prefix '{prefix}'. Original column dropped.")
    return df

def scale_numerical_features(df: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    """
    Scales specified numerical columns using StandardScaler from scikit-learn.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_names (list[str]): A list of numerical column names to scale.

    Returns:
    pd.DataFrame: The DataFrame with the scaled columns.
    """
    scaler = StandardScaler()
    
    for col in column_names:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Column '{col}' is not numerical and cannot be scaled.")
            
    # Fit and transform the specified columns
    df[column_names] = scaler.fit_transform(df[column_names])
    print(f"Scaled numerical features: {', '.join(column_names)} using StandardScaler.")
    return df

if __name__ == '__main__':
    print("--- Basic Preprocessor Demonstration ---")

    # Create a sample DataFrame
    data = {
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [10.0, 11.5, 12.0, np.nan, 14.0],
        'category_feature': ['A', 'B', 'A', 'C', np.nan], # Added NaN to test categorical imputation/encoding handling
        'feature_to_scale1': [100, 200, 300, 400, 500],
        'feature_to_scale2': [0.1, 0.2, 0.3, 0.4, 0.5]
    }
    sample_df = pd.DataFrame(data)
    print("\n--- Original Sample DataFrame ---")
    print(sample_df)
    report_missing_values(sample_df)

    # 1. Impute numerical features
    print("\n--- Imputing Numerical Features ---")
    try:
        sample_df = impute_numerical(sample_df, 'feature1', strategy='mean')
        sample_df = impute_numerical(sample_df, 'feature2', strategy='median')
        # Example of error handling:
        # sample_df = impute_numerical(sample_df, 'category_feature', strategy='mean') # Should raise TypeError
        # sample_df = impute_numerical(sample_df, 'feature1', strategy='mode') # Should raise ValueError
    except (ValueError, TypeError) as e:
        print(f"Error during numerical imputation: {e}")
    print(sample_df)
    report_missing_values(sample_df)

    # 2. Handle missing categorical values (e.g., by filling with a placeholder like 'Unknown')
    # One-hot encoding (pd.get_dummies) can handle NaNs by default if dummy_na=True, 
    # or they can be explicitly filled. For this demo, let's fill them.
    print("\n--- Handling Missing Categorical Values ---")
    if 'category_feature' in sample_df.columns: # Check if column exists
        sample_df['category_feature'] = sample_df['category_feature'].fillna('Unknown')
        print(sample_df)
        report_missing_values(sample_df)

        # 3. Encode categorical features
        print("\n--- Encoding Categorical Features ---")
        try:
            sample_df = encode_categorical(sample_df, 'category_feature', prefix='cat')
            # Example of error handling (though our current function allows numerical if user insists):
            # sample_df = encode_categorical(sample_df, 'feature1') # Might give unexpected results if not truly categorical
        except ValueError as e:
            print(f"Error during categorical encoding: {e}")
        print(sample_df.head()) # Display head to see new columns

    # 4. Scale numerical features
    print("\n--- Scaling Numerical Features ---")
    numerical_cols_to_scale = ['feature_to_scale1', 'feature_to_scale2']
    # Also include the imputed numerical features if they exist and are not one-hot encoded prefixes
    if 'feature1' in sample_df.columns and pd.api.types.is_numeric_dtype(sample_df['feature1']):
        numerical_cols_to_scale.append('feature1')
    if 'feature2' in sample_df.columns and pd.api.types.is_numeric_dtype(sample_df['feature2']):
        numerical_cols_to_scale.append('feature2')
    
    # Ensure all columns to scale actually exist in the dataframe
    numerical_cols_to_scale = [col for col in numerical_cols_to_scale if col in sample_df.columns]

    if numerical_cols_to_scale:
        try:
            sample_df = scale_numerical_features(sample_df, numerical_cols_to_scale)
        except (ValueError, TypeError) as e:
            print(f"Error during numerical scaling: {e}")
        print(sample_df.head())
    else:
        print("No numerical columns found or specified for scaling.")

    print("\n--- Final Processed DataFrame (Head) ---")
    print(sample_df.head())
    report_missing_values(sample_df)
    print("\n--- Basic Preprocessor Demonstration Finished ---")
