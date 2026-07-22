from distutils.util import strtobool
from logging import Logger
from typing import Dict, Any

import pandas as pd

from . import constants


def validate_and_get_inputs(covariates_path: str, data_path: str, computation_parameters: Dict[str, Any],
                            logger: Logger) -> bool:
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
                                                                          constants.DEFAULT_IgnoreSubjectsWithMissingData)
        ignore_subjects_with_missing_entries = bool(strtobool(str(ignore_subjects_with_missing_entries)))
        logger.info(f' ignore_subjects_with_missing_entries = {ignore_subjects_with_missing_entries}')

        strict_type_checking = computation_parameters.get(
            "StrictTypeChecking",
            constants.DEFAULT_StrictTypeChecking,
        )
        strict_type_checking = bool(strtobool(str(strict_type_checking)))
        logger.info(f' strict_type_checking = {strict_type_checking}')

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


        covariates = covariates[expected_covariates]
        data = data[expected_dependents]

        logger.info(f'-- Checking covariate and dependent files : {str(covariates_path)}, {str(data_path)}')

        X, y = _convert_data_to_given_type(covariates, expected_covariates_info, data, expected_dependents_info,
                                           logger, ignore_subjects_with_missing_entries, strict_type_checking)
        # dummy encoding categorical variables
        X = pd.get_dummies(X, drop_first=True)

        # If all checks pass
        return True, X, y

    except Exception as e:
        error_message = f"An error occurred during validation: {str(e)}"
        logger.error(error_message)
        return False, None, None


def _convert_data_to_given_type(covariates: pd.DataFrame, covariate_info: dict, data: pd.DataFrame,
                                dependent_info: dict, logger: Logger,
                                ignore_subjects_with_missing_entries: bool,
                                strict_type_checking: bool = False):
    """
      Converts each dataframe column to its type specified in computation parameters. If
      ignore_subjects_with_missing_entries is true, then the subjects with missing data will be ignored, otherwise
      it gives errors. If strict_type_checking is true, columns are also rejected if the raw cell value
      is not already the expected type (e.g. a boolean in a str column is flagged); otherwise only values
      that cannot be coerced at all are rejected (lenient/legacy behavior).
    """
    column_info = dict(covariate_info)
    column_info.update(dependent_info)

    expected_column_names = column_info.keys()

    assert len(covariates) == len(data), ("Covariates and Data have different number of rows. Please make sure both "
                                          "of them have the same number of rows.")

    #Combine data frames and
    combined_df = pd.concat([covariates, data], axis=1)
    combined_df = combined_df[list(expected_column_names)]

    all_rows_to_ignore = _validate_data_datatypes(combined_df, column_info, logger, strict_type_checking)
    if len(all_rows_to_ignore) > 0:
        if ignore_subjects_with_missing_entries:
            logger.info(f'-- Ignored following rows with incorrect column values: {str(_get_user_row_numbers(all_rows_to_ignore))}')
            combined_df.drop(all_rows_to_ignore, inplace=True)
        else:
            err_msg = (f'Following rows have empty or invalid entries for columns. Either choose to ignore these rows '
                       f'or correct the data and try again. See log file for details: {str(_get_user_row_numbers(all_rows_to_ignore))}')
            logger.error(err_msg)
            raise Exception(err_msg)

    else:
        logger.info(f' Data validation passed for all the columns: {str(expected_column_names)}')

    # All the potential
    try:
        for column_name, column_datatype in column_info.items():
            logger.info(f'Casting datatype of column: {column_name} to the requested datatype : {column_datatype}')
            if column_datatype.strip().lower() == "int":
                combined_df[column_name] = pd.to_numeric(combined_df[column_name], errors='coerce').astype(
                    'int')  # or .astype('Int64')
            elif column_datatype.strip().lower() == "float":
                combined_df[column_name] = pd.to_numeric(combined_df[column_name], errors='coerce').astype('float')
            elif column_datatype.strip().lower() == "str":
                combined_df[column_name] = combined_df[column_name].astype('object')
            elif column_datatype.strip().lower() == "bool":
                combined_df[column_name] = pd.to_numeric(combined_df[column_name], errors='coerce').astype('bool')
            else:
                err_msg = (f'Invalid datatype provided in the input for column : {column_name} and datatype: '
                           f'{column_datatype}. Allowed datatypes are int, float, str, bool.')
                logger.error(err_msg)
                raise Exception(err_msg)

        # Check for null or NaNs in the converted data
        curr_rows_to_ignore = combined_df[combined_df.isnull().any(axis=1)].index.tolist()
        if len(curr_rows_to_ignore) > 0:
            if ignore_subjects_with_missing_entries:
                logger.info(f'-- Ignored following rows with incorrect column values: {str(_get_user_row_numbers(curr_rows_to_ignore))}')
                combined_df.drop(curr_rows_to_ignore, inplace=True)
            else:
                err_msg = (f'Following rows have empty or invalid entries for columns after converting to their '
                           f'respective datatypes. Either choose to ignore these rows or correct the data and'
                           f' try again. See log file for details: {str(_get_user_row_numbers(curr_rows_to_ignore))}')
                logger.error(err_msg)
                raise Exception(err_msg)

        combined_df = combined_df[expected_column_names]

    except Exception as e:
        error_message = f"An error occurred during type conversion for data: {str(e)}"
        logger.error(error_message)
        raise (e)

    #Separate covariates and data into separate data frames
    covariates_X_df = combined_df[list(covariate_info.keys())]
    data_y_df = combined_df[list(dependent_info.keys())]

    return covariates_X_df, data_y_df


def _validate_data_datatypes(data_df: pd.DataFrame, column_info: dict, logger: Logger,
                             strict_type_checking: bool = False) -> list:
    """
     Validates if each dataframe column is compatible with the type specified in computation parameters.

     Lenient mode (strict_type_checking=False, default): flags rows where the value cannot be coerced
     to the target type at all (e.g. "banana" for a float column). This preserves legacy behavior.

     Strict mode (strict_type_checking=True): additionally flags rows where the raw cell value is not
     already the expected Python type — e.g. a boolean (True/False) in a str column, or a string in
     an int column. Use this to catch type mismatches that would otherwise be silently coerced.
    """
    all_rows_to_ignore = set()

    # Strict mode: map each expected type string to the Python types considered valid for that column.
    # bool must be checked before int/float since bool is a subclass of int in Python.
    _STRICT_ALLOWED_TYPES = {
        "str":   (str,),
        "int":   (int,),
        "float": (float, int),   # int is acceptable in a float column
        "bool":  (bool,),
    }

    try:
        for column_name, column_datatype in column_info.items():
            logger.info(f'\nValidating column: {column_name} with requested datatype : {column_datatype}')
            dtype_key = column_datatype.strip().lower()

            if dtype_key not in ("int", "float", "str", "bool"):
                err_msg = (f'Invalid datatype provided in the input for column : {column_name} and datatype: '
                           f'{column_datatype}. Allowed datatypes are int, float, str, bool.')
                logger.error(err_msg)
                raise Exception(err_msg)

            rows_to_ignore = []

            if strict_type_checking:
                allowed_types = _STRICT_ALLOWED_TYPES[dtype_key]

                def _is_wrong_type(val):
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        return False  # defer to post-cast null check
                    # For bool columns, only accept actual bools (not 0/1 ints)
                    if dtype_key == "bool":
                        return not isinstance(val, bool)
                    # For int/float columns, reject bools even though bool is a subclass of int
                    if dtype_key in ("int", "float"):
                        return isinstance(val, bool) or not isinstance(val, allowed_types)
                    return not isinstance(val, allowed_types)

                mask = data_df[column_name].map(_is_wrong_type)
                rows_to_ignore = data_df[mask].index.tolist()

            else:
                # Lenient mode: attempt coercion and flag rows that produce NaN
                if dtype_key == "int":
                    temp = pd.to_numeric(data_df[column_name], errors='coerce').astype('int')
                elif dtype_key == "float":
                    temp = pd.to_numeric(data_df[column_name], errors='coerce').astype('float')
                elif dtype_key == "str":
                    temp = data_df[column_name].astype('object')
                elif dtype_key == "bool":
                    # Converting first to 'int' type to make sure all the possible values are converted correctly
                    temp = pd.to_numeric(data_df[column_name], errors='coerce').astype('Int64')

                # Check for null or NaNs in the data
                rows_to_ignore = data_df[temp.isnull()].index.tolist()

                # Check for empty values in str columns
                if dtype_key == "str":
                    rows_to_ignore = data_df[temp.str.strip() == ''].index.tolist()

            all_rows_to_ignore = all_rows_to_ignore.union(rows_to_ignore)

            if len(rows_to_ignore) > 0:
                logger.info(f'Rows with incorrect values for column {column_name} : {str(_get_user_row_numbers(rows_to_ignore))}')
            else:
                logger.info(f'Data validation passed for column: {column_name} to the requested datatype : {column_datatype}')

    except Exception as e:
        error_message = f"An error occurred during validation: {str(e)}"
        logger.error(error_message)
        raise (e)

    return list(all_rows_to_ignore)


def _get_user_row_numbers(df_index_list):
    return [ri + 1 for ri in df_index_list]
