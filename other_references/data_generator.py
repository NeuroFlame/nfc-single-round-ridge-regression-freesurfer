import os
import pandas as pd


def generate_data_csv(aseg_file_names, data_dir, output_file_name, freesurfer_labels=[]):
    '''
    Generates freesurfer data as csv from a directory with aseg freesurfer files for a list of subjects.
    aseg_file_names: files names to parse to be included (subjects)
    data_dir: directory containing aseg files
    output_file_name: name of the output file with extension (.csv)
    freesurfer_labels: freesurfer labels that needs to be in the output file
    '''
    log_list=[]
    if len(aseg_file_names) <=0:
        print("Please provide aseg file names.")

    if not freesurfer_labels:
        #Read the first file to load all the possible freesurfer labels
        temp_df = pd.read_csv(
            os.path.join(data_dir, aseg_file_names[0]),
            sep='\t',
            header=None,
            names=['Measure:volume', aseg_file_names[0]],
            index_col=0)
        freesurfer_labels=sorted(temp_df.index.unique().tolist())

    freesurfer_label_set=set(freesurfer_labels)
    y = pd.DataFrame(index=freesurfer_labels)

    log_list.append(f'ROIs provided {freesurfer_labels}')
    remove_aseg_file_names=[]
    for file_name in aseg_file_names:
        if file_name:
            try:
                y_ = pd.read_csv(
                    os.path.join(data_dir, file_name), sep='\t', header=None, names=['Measure:volume', file_name],
                    index_col=0)
                y_ = y_[~y_.index.str.contains("Measure:volume")]
                if not freesurfer_label_set.issubset(set(y_.index)):
                    err_msg = (f"ERROR: Freesurfer Areas of interest provided in the input are not present in the aseg "
                               f"files. \n Provided areas of interest: {str(freesurfer_label_set)} "
                               f"\n Missing areas of interest: {str(freesurfer_label_set.difference(set(y_.index)))}"
                               f"\n Areas of interest in aseg files: {str(y_.index)}")
                    log_list.append(err_msg)
                    raise Exception(err_msg)

                # skipping files with repeated brain regions
                repeated_brain_regions=freesurfer_label_set.intersection(y_[y_.index.duplicated()].index.tolist())
                if any(repeated_brain_regions):
                    remove_aseg_file_names.append(file_name)
                    log_list.append(f'SKIPPING file {os.path.join(data_dir, file_name)} which has repeated '
                                    f'brain region measures for {str(repeated_brain_regions)}')
                    continue
                y_ = y_.apply(pd.to_numeric)
                y = pd.merge(
                    y, y_, how='left', left_index=True, right_index=True)
                log_list.append(f'Processed file {os.path.join(data_dir, file_name)}')
            except pd.errors.EmptyDataError:
                log_list.append(f'Empty content in the file {os.path.join(data_dir, file_name)}')
                continue
            except FileNotFoundError:
                log_list.append(f'File not found{os.path.join(data_dir, file_name)}')
                continue

    y = y.T

    # Save to csv
    y.to_csv(os.path.join(data_dir, output_file_name), index_label='freesurferfile')

    return remove_aseg_file_names, log_list

def _get_freesurfer_file_names_from_covariates(covariates_file):
    temp_df=pd.read_csv(covariates_file, header=0)
    return temp_df['freesurferfile'].tolist()

def get_data_csv_from_covariates(data_dir, covariates_file_name, freesurfer_labels=[], output_data_file_name='data.csv'):
    '''
    Generates freesurfer data as csv from a directory with covariates file containing the aseg freesurfer files
                    for a list of subjects.
    data_dir: directory containing aseg files and covariate file
    covariates_file_name: covariates file name
    output_data_file_name: name of the output file with extension (.csv)
    freesurfer_labels: freesurfer labels that needs to be in the output file. Considers all labels if no value is given.
    '''

    freesurfer_file_names = _get_freesurfer_file_names_from_covariates(os.path.join(data_dir, covariates_file_name))
    aseg_file_names_to_remove, logs = generate_data_csv(freesurfer_file_names, data_dir, output_data_file_name,
                                                        freesurfer_labels)

    if len(aseg_file_names_to_remove) > 0:
        new_covariates_file_name=os.path.join(data_dir, "updated_"+covariates_file_name)
        logs.append(f"Updating and creating a new covariates file skipping rows with the "
                    f"following aseg files: {str(aseg_file_names_to_remove)}")
        temp_df = pd.read_csv(os.path.join(data_dir, covariates_file_name), header=0, index_col="freesurferfile")
        temp_df.drop(aseg_file_names_to_remove, inplace=True)
        temp_df.to_csv(new_covariates_file_name,  index_label = 'freesurferfile')
        logs.append(f"Updated covariates file saved at: {new_covariates_file_name}")

    # Saving log file
    log_file_name=os.path.join(data_dir, "data_generator_log.txt" )
    with open(log_file_name, "a") as file:
        for line in logs:
            file.write(line + "\n")

    return logs


# Example usage
# if __name__ == "__main__":
#     # test1
#     # data_dir1='../test_data2/SSR_test_data_from_cobre_vault'
#     # freesurfer_file_names = _get_freesurfer_file_names_from_covariates(os.path.join(data_dir1, 'Covariates.csv'))
#     # generate_data_csv(aseg_file_names, data_dir1, output_file_name='data.csv', freesurfer_labels=[])
#
#     # test2
#     # Generates the data.csv in "data_dir1" folder itself
#     data_dir1='../vault_data/test_data/SSR_test_data_from_cobre_vault'
#     get_data_csv_from_covariates(data_dir1, "Covariates.csv", "data.csv")