import os
import shutil

from nilearn import datasets

from DataTools import DataTools

# Input data variables
root_folder = '/Data_SecondFastestSpeed/cth/MMGL/miccai_ABIDE/data'
txt_path = '/home/cth/pythonProject/MMGL/data/ABIDE/subject_IDs.txt'
download_mode = False

pipeline = 'cpac'
files = ['rois_ho']
num_subjects = 871
filemapping = {'func_preproc': 'func_preproc.nii.gz',
               'rois_ho': 'rois_ho.1D'}
data_folder = os.path.join(root_folder, 'ABIDE_pcp/cpac/filt_noglobal')

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

data_tools = DataTools(root_folder, txt_path)
subject_IDs = data_tools.subject_IDs

if download_mode:
    print("Downloading files...")
    # Download database files
    datasets.fetch_abide_pcp(data_dir=root_folder, n_subjects=num_subjects, pipeline=pipeline,
                             band_pass_filtering=True, global_signal_regression=False, derivatives=files)

    print("Download done!")
    print("Moving files to target folders...")

    for s, fname in zip(subject_IDs, data_tools.fetch_filenames(subject_IDs, files[0])):
        subject_folder = os.path.join(data_folder, s)
        if not os.path.exists(subject_folder):
            os.mkdir(subject_folder)

        # Get the base filename for each subject
        base = fname.split(files[0])[0]

        # Move each subject file to the subject folder
        for fl in files:
            if not os.path.exists(os.path.join(subject_folder, base + filemapping[fl])):
                shutil.move(base + filemapping[fl], subject_folder)

    print("Moving files done!")

time_series = data_tools.get_timeseries(subject_IDs, 'ho')

# Compute and save connectivity matrices
for i in range(len(subject_IDs)):
    # subject_connectivity func will generate .mat file as output
    data_tools.subject_connectivity(time_series[i], subject_IDs[i], 'ho', 'correlation')
