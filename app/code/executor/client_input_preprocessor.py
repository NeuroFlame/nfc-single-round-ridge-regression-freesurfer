from distutils.util import strtobool
from typing import Dict, Any

import pandas as pd
from utils.logger import NFCLogger

from . import client_constants


def validate_and_get_inputs(covariates_path: str, data_path: str, computation_parameters: Dict[str, Any],
                            logger: NFCLogger) -> bool:
    """
       Performs validation on the covariates and data files against provided computation parameters
    """
    try:
        # Extract covariates and dependent headers from computation parameters
        # If given as covariate:datatype as input format
        expected_covariates_info = computation_parameters["Covariates"]
        expected_dependents_info = computation_parameters["Dependents"]
        expected_covariates = list(expected_covariates_info.keys())
        expected_dependents = list(expected_dependents_info.keys())

        ignore_subjects_with_missing_entries = computation_parameters.get("IgnoreSubjectsWithMissingData",
                                                                          client_constants.DEFAULT_IgnoreSubjectsWithMissingData)
        ignore_subjects_with_missing_entries = bool(strtobool(str(ignore_subjects_with_missing_entries)))
        logger.info(f' ignore_subjects_with_missing_entries = {ignore_subjects_with_missing_entries}')

        # Load the data
        covariates = pd.read_csv(covariates_path)
        data = pd.read_csv(data_path)

        # Validate covariates headers
        covariates_headers = set(covariates.columns)
        if not set(expected_covariates).issubset(covariates_headers):
            error_message = (f"Covariates headers do not contain all expected headers. Expected at least "
                             f"{expected_covariates}, but got {covariates_headers}.")
            logger.info(error_message)
            return False, None, None

        # Validate data headers
        data_headers = set(data.columns)
        if not set(expected_dependents).issubset(data_headers):
            error_message = (f"Data headers do not contain all expected headers. Expected at least "
                             f"{expected_dependents}, but got {data_headers}.")
            logger.info(error_message)
            return False, None, None

        logger.info(f'-- Checking covariate file : {str(covariates_path)}')

        X = _convert_data_to_given_type(covariates, expected_covariates_info, logger,
                                        ignore_subjects_with_missing_entries)
        # dummy encoding categorical variables
        X = pd.get_dummies(X, drop_first=True)

        logger.info(f'-- Checking dependents file : {str(data_path)}')

        y = _convert_data_to_given_type(data, expected_dependents_info, logger, ignore_subjects_with_missing_entries)

        # If all checks pass
        return True, X, y

    except Exception as e:
        error_message = f"An error occurred during validation: {str(e)}"
        logger.error(error_message)
        return False, None, None


def _convert_data_to_given_type(data_df: pd.DataFrame, column_info: dict, logger: NFCLogger,
                                ignore_subjects_with_missing_entries: bool):
    """
      Converts each dataframe column to its type specified in computation parameters. If
      ignore_subjects_with_missing_entries is true, then the subjects with missing data will be ignored, otherwise
      it gives errors.
    """
    expected_column_names = column_info.keys()

    all_rows_to_ignore = _validate_data_datatypes(data_df, column_info, logger)
    if len(all_rows_to_ignore) > 0:
        if ignore_subjects_with_missing_entries:
            logger.info(f'-- Ignored following rows with incorrect column values: {str(all_rows_to_ignore)}')
            data_df.drop(data_df.index[all_rows_to_ignore], inplace=True)
        else:
            err_msg = (f'Following rows have empty or invalid entries for columns. Either choose to ignore these rows '
                       f'or correct the data and try again. See log file for details: {str(all_rows_to_ignore)}')
            logger.error(err_msg)
            raise Exception(err_msg)

    else:
        logger.info(f' Data validation passed for all the columns: {str(expected_column_names)}')

    # All the potential
    try:
        for column_name, column_datatype in column_info.items():
            logger.info(f'Casting datatype of column: {column_name} to the requested datatype : {column_datatype}')
            if column_datatype.strip().lower() == "int":
                data_df[column_name] = pd.to_numeric(data_df[column_name], errors='coerce').astype(
                    'int')  # or .astype('Int64')
            elif column_datatype.strip().lower() == "float":
                data_df[column_name] = pd.to_numeric(data_df[column_name], errors='coerce').astype('float')
            elif column_datatype.strip().lower() == "str":
                data_df[column_name] = data_df[column_name].astype('object')
            elif column_datatype.strip().lower() == "bool":
                data_df[column_name] = pd.to_numeric(data_df[column_name], errors='coerce').astype('bool')
            else:
                err_msg = (f'Invalid datatype provided in the input for column : {column_name} and datatype: '
                           f'{column_datatype}. Allowed datatypes are int, float, str, bool.')
                logger.error(err_msg)
                raise Exception(err_msg)

        # Check for null or NaNs in the converted data
        curr_rows_to_ignore = data_df[data_df.isnull().any(axis=1)].index.tolist()
        if len(curr_rows_to_ignore) > 0:
            if ignore_subjects_with_missing_entries:
                logger.info(f'-- Ignored following rows with incorrect column values: {str(all_rows_to_ignore)}')
                data_df.drop(data_df.index[curr_rows_to_ignore], inplace=True)
            else:
                err_msg = (f'Following rows have empty or invalid entries for columns after converting to their '
                           f'respective datatypes. Either choose to ignore these rows or correct the data and'
                           f' try again. See log file for details: {str(all_rows_to_ignore)}')
                logger.error(err_msg)
                raise Exception(err_msg)

        data_df = data_df[expected_column_names]

    except Exception as e:
        error_message = f"An error occurred during type conversion for data: {str(e)}"
        logger.error(error_message)
        raise (e)

    return data_df


def _validate_data_datatypes(data_df: pd.DataFrame, column_info: dict, logger: NFCLogger) -> list:
    """
     Validates if each dataframe column is compatible to the type specified in computation parameters.
    """
    all_rows_to_ignore = set()
    try:
        for column_name, column_datatype in column_info.items():
            logger.info(f'Validating column: {column_name} with requested datatype : {column_datatype}')
            if column_datatype.strip().lower() == "int":
                temp = pd.to_numeric(data_df[column_name], errors='coerce').astype('int')  # or .astype('Int64')
            elif column_datatype.strip().lower() == "float":
                temp = pd.to_numeric(data_df[column_name], errors='coerce').astype('float')
            elif column_datatype.strip().lower() == "str":
                temp = data_df[column_name].astype('object')
            elif column_datatype.strip().lower() == "bool":
                # Converting first to 'int' type to make sure all the possible values are converted correctly
                temp = pd.to_numeric(data_df[column_name], errors='coerce').astype('int')  # or .astype('Int64')
            else:
                err_msg = (f'Invalid datatype provided in the input for column : {column_name} and datatype: '
                           f'{column_datatype}. Allowed datatypes are int, float, str, bool.')
                logger.error(err_msg)
                raise Exception(err_msg)

            # Check for null or NaNs in the data
            rows_to_ignore = data_df[temp.isnull()].index.tolist()

            # Check for emtpy values in the data
            if column_datatype.strip().lower() == "str":
                rows_to_ignore = data_df[temp.str.strip() == ''].index

            all_rows_to_ignore = all_rows_to_ignore.union(rows_to_ignore)

            if len(rows_to_ignore) > 0:
                logger.info(f'Rows with incorrect values for column {column_name} : {str(rows_to_ignore)}')

            else:
                logger.info(
                    f'Data validation passed for column: {column_name} to the requested datatype : {column_datatype}')

    except Exception as e:
        error_message = f"An error occurred during validation: {str(e)}"
        logger.error(error_message)
        raise (e)

    return list(all_rows_to_ignore)
