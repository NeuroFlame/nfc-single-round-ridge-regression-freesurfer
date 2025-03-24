import os
from sklearn.model_selection import StratifiedKFold
import pandas as pd

def merge_existing_data(input_dir, covariates_file_name, data_file_name):
    '''
    Combine covariate and data from all the sites and return them as dataframes
    '''
    site_dirs = [os.path.join(input_dir, name) for name in os.listdir(input_dir) if
                    name.startswith("site") and os.path.isdir(os.path.join(input_dir, name))]
    covariates_df=None
    data_df=None

    for site_dir in site_dirs:
        temp_covar_df = pd.read_csv(os.path.join(site_dir, covariates_file_name), header=0)
        temp_data_df = pd.read_csv(os.path.join(site_dir, data_file_name))
        if covariates_df is None or data_df is None:
            covariates_df = temp_covar_df
            data_df = temp_data_df
        else:
            covariates_df = pd.concat([covariates_df, temp_covar_df], ignore_index=True)
            data_df = pd.concat([data_df, temp_data_df], ignore_index=True)

    return covariates_df, data_df

def partition_data(num_splits, stratify_covariate_column_name, covariates_df, data_df, output_dir):
    '''
    Partitions a dataset into the specified number of splits based on stratified sampling.
    num_splits: number of paritions needed for the data
    stratify_covariate_column_name: column name in the covariate file based on which data needs to be sampled
    covariates_df: covariates dataframe object
    data_df: freesurfer dataframe object
    output_dir: output directory path (needs a new directory name)
    '''

    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    splits = []
    for train_index, test_index in skf.split(covariates_df, covariates_df[stratify_covariate_column_name]):
        splits.append(test_index.tolist())

    has_issue=False
    for i in range(len(splits)):
        for j in range(i+1, len(splits)):
           if set(splits[i]).intersection(splits[j]):
               err_msg=f"Has common elements between split {i}, {j}"
               print(err_msg)
               has_issue=True
               break;
    if has_issue:
        raise Exception(err_msg)

    for i in range(len(splits)):
        X = covariates_df.iloc[splits[i]]
        y = data_df.iloc[splits[i]]

        dir_name=os.path.join(output_dir, f'site{i+1}')
        os.makedirs(dir_name)

        X.to_csv(os.path.join(dir_name, f'covariates.csv'), index=False)
        y.to_csv(os.path.join(dir_name, f'data.csv'), index=False)


if __name__ == "__main__":
    # test-1
    # base_dir = '../vault_data/test_data/SSR_test_data_from_cobre_vault'
    # covariates_df = pd.read_csv(os.path.join(base_dir, 'Covariates.csv'), header=0)
    # data_df = pd.read_csv(os.path.join(base_dir, 'data.csv'))
    # partition_data(4, covariates_df, data_df, output_dir)

    # test-2
    covar_df, data_df = merge_existing_data("../test_data", "covariates.csv", "data.csv")
    partition_data(4, "isControl", covar_df, data_df, "../temp_test_data")

