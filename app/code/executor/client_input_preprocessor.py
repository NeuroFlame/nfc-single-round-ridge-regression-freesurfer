import logging
import numpy as np
import pandas as pd

from typing import Dict, Any

def validate_and_get_inputs(covariates_path: str, data_path: str, computation_parameters: Dict[str, Any], log_path: str) -> bool:
    try:
        # Extract covariates and dependent headers from computation parameters
        #expected_covariates = computation_parameters["Covariates"]
        #expected_dependents = computation_parameters["Dependents"]

        #If given as covariate:datatype as input format
        expected_covariates_info = computation_parameters["Covariates"]
        expected_dependents_info = computation_parameters["Dependents"]
        expected_covariates = list(expected_covariates_info.keys())
        expected_dependents = list(expected_dependents_info.keys())
        #X_types = list(X_vars.values())
        #y_types = list(y_vars.values())

        # Load the data
        covariates = pd.read_csv(covariates_path)
        data = pd.read_csv(data_path)

        # Validate covariates headers
        covariates_headers = set(covariates.columns)
        if not set(expected_covariates).issubset(covariates_headers):
            error_message = f"Covariates headers do not contain all expected headers. Expected at least {expected_covariates}, but got {covariates_headers}."
            _log_message(error_message, log_path, "info")
            return False, None, None
        
        # Validate data headers
        data_headers = set(data.columns)
        if not set(expected_dependents).issubset(data_headers):
            error_message = f"Data headers do not contain all expected headers. Expected at least {expected_dependents}, but got {data_headers}."
            _log_message(error_message, log_path, "info")
            return False, None, None

        _log_message(f'-- Checking covariate file : {str(covariates_path)}', log_path, "info")
        X = convert_data_to_given_type(covariates, expected_covariates_info, log_path)
        #dummy encoding categorical variables
        X = pd.get_dummies(X, drop_first=True)

        _log_message(f'-- Checking dependents file : {str(data_path)}', log_path, "info")
        y = convert_data_to_given_type(data, expected_dependents_info, log_path)

        # If all checks pass
        return True, X, y

    except Exception as e:
        error_message = f"An error occurred during validation: {str(e)}"
        _log_message(error_message, log_path, "error")
        return False, None, None



def convert_data_to_given_type(data_df : pd.DataFrame, column_info : dict, log_path : str):
    expected_column_names = column_info.keys()

    all_rows_to_ignore = _validate_data_datatypes(data_df, column_info, log_path)
    if len(all_rows_to_ignore) > 0:
        _log_message(f'-- Ignored following rows with incorrect column values: '
                     f'{str(all_rows_to_ignore)}', log_path, "info")

        data_df.drop(data_df.index[all_rows_to_ignore], inplace=True)

    else:
        _log_message(f' Data validation passed for all the columns: {str(expected_column_names)}', log_path, "info")

    # All the potential
    try:
        for column_name, column_datatype in column_info.items():
            _log_message(f'Casting datatype of column: {column_name} to the requested datatype '
                             f': {column_datatype}', log_path, "info")
            if column_datatype.strip().lower() == "int":
                data_df[column_name] = pd.to_numeric(data_df[column_name], errors='coerce').astype('int') #or .astype('Int64')
            elif column_datatype.strip().lower() == "float":
                data_df[column_name] = pd.to_numeric(data_df[column_name], errors='coerce').astype('float')
            elif column_datatype.strip().lower() == "str":
                data_df[column_name] = data_df[column_name].astype('object')
            elif column_datatype.strip().lower() == "bool":
                data_df[column_name] = pd.to_numeric(data_df[column_name], errors='coerce').astype('bool')
            else:
                raise Exception(f'Invalid datatype provided in the input for column : {column_name}'
                                f' and datatype: {column_datatype}. Allowed datatypes are int, float, str, bool.')
        # Check for null or NaNs in the converted data
        curr_rows_to_ignore = data_df[data_df.isnull().any(axis=1)].index.tolist()
        data_df.drop(data_df.index[curr_rows_to_ignore], inplace=True)

        data_df = data_df[expected_column_names]

    except Exception as e:
        error_message = f"An error occurred during type conversion for data: {str(e)}"
        _log_message(error_message, log_path, "error")
        raise (e)

    return data_df

def _validate_data_datatypes(data_df : pd.DataFrame, column_info : dict, log_path : str) -> list:
    all_rows_to_ignore=set()
    try:
        for column_name, column_datatype in column_info.items():
            _log_message(f'Validating column: {column_name} with requested datatype '
                             f': {column_datatype}', log_path, "info")
            if column_datatype.strip().lower() == "int":
                temp = pd.to_numeric(data_df[column_name], errors='coerce').astype('int') #or .astype('Int64')
            elif column_datatype.strip().lower() == "float":
                temp = pd.to_numeric(data_df[column_name], errors='coerce').astype('float')
            elif column_datatype.strip().lower() == "str":
                temp = data_df[column_name].astype('object')
            elif column_datatype.strip().lower() == "bool":
                #Converting to int first to make sure all the possible values are converted correctly
                temp = pd.to_numeric(data_df[column_name], errors='coerce').astype('int') #or .astype('Int64')
            else:
                raise Exception(f'Invalid datatype provided in the input for column : {column_name}'
                                f' and datatype: {column_datatype}. Allowed datatypes are int, float, str, bool.')

            #Check for null or NaNs in the data
            rows_to_ignore = data_df[temp.isnull()].index.tolist()

            #Check for emtpy values in the data
            if column_datatype.strip().lower() == "str":
                rows_to_ignore = data_df[temp.str.strip() == ''].index

            all_rows_to_ignore = all_rows_to_ignore.union(rows_to_ignore)

            if len(rows_to_ignore) > 0:
                _log_message(f' Ignoring rows with incorrect values for column {column_name} : '
                         f'{str(rows_to_ignore)}', log_path, "info")
            else:
                _log_message(f' Data validation passed for column: {column_name} to the requested datatype '
                             f': {column_datatype}', log_path, "info")

    except Exception as e:
        error_message = f"An error occurred during validation: {str(e)}"
        _log_message(error_message, log_path, "error")
        raise (e)

    return list(all_rows_to_ignore)


def _log_message(message: str, log_path: str, message_level : str ) -> None:
    """
    Logs details to log file.

    Args:
        log_path: path of the log file
        message_level: "error" or "info". Default level is info

    Returns:

    """
    import datetime
    time_prefix = datetime.datetime.now().astimezone().strftime("%m/%d/%Y %H:%M:%S") + ' : '
    message = time_prefix + message

    if message_level.strip().lower() == "error":
        logging.error(message)
    else:
        logging.info(message)

    try:
        with open(log_path, 'a') as f:
            f.write(f"{message}\n")
            f.flush()  # Ensure data is written to the file
    except IOError as e:
        logging.error(f"Failed to write to log file {log_path}: {e}")


def ignore_nans(X, y):
    """Removing rows containing NaN's in X and y"""

    if type(X) is pd.DataFrame:
        X_ = X.values.astype('float64')
    else:
        X_ = X

    if type(y) is pd.Series:
        y_ = y.values.astype('float64')
    else:
        y_ = y

    finite_x_idx = np.isfinite(X_).all(axis=1)
    finite_y_idx = np.isfinite(y_)

    finite_idx = finite_y_idx & finite_x_idx

    y_ = y_[finite_idx]
    X_ = X_[finite_idx, :]

    return X_, y_
