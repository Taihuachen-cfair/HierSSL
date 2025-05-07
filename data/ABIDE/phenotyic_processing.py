import os

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

from DataTools import DataTools

# Input data variables
root_folder = '/Data_SecondFastestSpeed/cth/MMGL/miccai_ABIDE/data'
txt_path = '/home/cth/pythonProject/MMGL/data/ABIDE/subject_IDs.txt'
save_path = '~/pythonProject/MMGL/data/ABIDE/'

data_tools = DataTools(root_folder, txt_path)
subject_IDs = data_tools.subject_IDs
save_path = os.path.expanduser(save_path)

LABEL = data_tools.get_subject_score(subject_IDs, score='DX_GROUP')
num_nodes = len(subject_IDs)

# Initialise variables for class labels
y = np.zeros([num_nodes, 1])

# Get class labels and acquisition site for all subjects
for i in range(num_nodes):
    y[i] = int(LABEL[subject_IDs[i]])

label = y
label = label.reshape(-1)

# Compute feature vectors (vectorised connectivity networks)
features = data_tools.get_networks(subject_IDs, kind='correlation', atlas_name='ho')

skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
feat_256_1 = None

for train_index, test_index in skf.split(features, label):
    feat_256_1 = data_tools.feature_selection(features, label, train_index, 256)

feat_256_1_pd = pd.DataFrame(feat_256_1)
feat_256_1_pd['label'] = label
# processed_standard_data_256_1.csv is a temp file
feat_256_1_pd.to_csv(os.path.join(save_path, "processed_standard_data_256_1.csv"), index=False)

input_df = pd.read_csv(data_tools.phenotype_path, low_memory=False)

initial_data = input_df[input_df.SUB_ID.isin(subject_IDs)]

anat = ['anat_cnr', 'anat_efc', 'anat_fber', 'anat_fwhm', 'anat_qi1', 'anat_snr']
func = ['func_efc', 'func_fber', 'func_fwhm', 'func_dvars', 'func_outlier', 'func_quality', 'func_mean_fd',
        'func_num_fd', 'func_perc_fd', 'func_gsr']

data_func = initial_data[func]
data_anat = initial_data[anat]

data = initial_data.copy()

min_age, max_age = data.AGE_AT_SCAN.min(), data.AGE_AT_SCAN.max()
step, bins, block = 2, [min_age], min_age
while block < max_age:
    block += 2
    bins.append(block)
data.loc[:, 'AGE_AT_SCAN'] = pd.cut(data.AGE_AT_SCAN, bins, right=False)

data = pd.get_dummies(data, columns=['AGE_AT_SCAN'])
data = pd.get_dummies(data, columns=['SITE_ID'])
data = pd.get_dummies(data, columns=['SEX'])

pheno_list = list(data.columns[data.columns.str.contains('^SITE|^AGE_AT_SCAN|^SEX')])
anat_list = anat
func_list = func
feat_list = pheno_list + anat_list + func_list

ABIDE_modal_list = {'PHENO': pheno_list,
                    'ANAT': anat_list,
                    'FUNC': func_list}

select_data = data[feat_list]

standard_list = anat_list + func_list

scaler = preprocessing.StandardScaler()
standard_data = scaler.fit_transform(data[standard_list])

select_data[standard_list] = standard_data

# processed_data_modal_three.csv is a temp file
select_data.to_csv(os.path.join(save_path, "processed_data_modal_three.csv"), index=False)

feat_256_1 = pd.read_csv(os.path.join(save_path, "processed_standard_data_256_1.csv"), low_memory=False)
pheno_256 = pd.read_csv(os.path.join(save_path, "processed_data_modal_three.csv"), low_memory=False)

feat_256_1_list = list(feat_256_1.columns)[:-1]
ABIDE_modal_list['Correlation'] = feat_256_1_list
data_256_1 = pd.concat([pheno_256, feat_256_1], axis=1)

np.save(os.path.join(save_path, 'modal_feat_dict.npy'), ABIDE_modal_list)
data_256_1.to_csv("processed_standard_data.csv", index=False)

# TEMP FILE DELETE
os.remove(os.path.join(save_path, "processed_standard_data_256_1.csv"))
os.remove(os.path.join(save_path, "processed_data_modal_three.csv"))
