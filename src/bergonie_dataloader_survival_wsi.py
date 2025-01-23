import os
import torch
import numpy as np
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
#from imblearn.over_sampling import SMOTE
from collections import Counter
import pickle as pl
import pandas as pd

class Patient(data_utils.Dataset):
    def _setup_bag(self, pname, n_tiles = 1000):
        list_files = sorted(list(os.listdir(self.root_folder)))
        list_wsis = [pfile for pfile in list_files if pname in pfile]
        all_wsi = []
        for wsi in list_wsis:
            wsi_path = os.path.join(self.root_folder, wsi)
            c_files, _, _ = self._read_folder(wsi_path)
            all_wsi.append(c_files)
        wsi_file = np.concatenate(all_wsi, axis = 0)
        wsi_file= shuffle(wsi_file)#, random_state = self.seed)
        selected_tiles = self.random_select_tiles_wsi(pname, wsi_file, num_tiles = n_tiles)
        return selected_tiles

    def __init__(self, root_folder, patient_list, labels_list, transf = None, n_tiles = 0, seed = 2452):
        super(Patient, self).__init__()
        self.root_folder = root_folder
        self.patient_list = patient_list
        self.outcome_list = labels_list
        self.transforms = transf
        self.seed = seed
        self.n_tiles = n_tiles
        print(self.patient_list)
    
    def _read_folder(self, npz_file, label = 0):
        npz_array = np.load(str(npz_file), allow_pickle=True)['arr_0']
        list_labels = [label] * npz_array.shape[0]
        return npz_array, np.array(list_labels), npz_array.shape[0]
        
    def random_select_tiles_wsi(self, pname, np_array, num_tiles = 10000):
        n_rows = np_array.shape[0]
        r_state = np.random.RandomState(self.seed)
        if num_tiles == -1:
            return np_array
        rd_indices = np.random.choice(n_rows, size = num_tiles, replace = True)
        selected_array = np_array[rd_indices,:]
        return selected_array
        
    def get_bag_patient_idx(self, idx):
        patient = self.patient_list[idx]
        status, surv_time = self.outcome_list[idx]
        
        wsi_file = self._setup_bag(patient, self.n_tiles)
        status = torch.Tensor([status])
        surv_time_tensor = torch.Tensor([surv_time])
        input_tensor = torch.from_numpy(wsi_file)
        
        sample = {'itensor': input_tensor, 
					'istatus': status,
					'isurvtime':surv_time_tensor}
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample['itensor'], sample['istatus'], sample['isurvtime']
    
    def __len__(self):
        return len(self.patient_list)
        
    def __getitem__(self, index):
        return self.get_bag_patient_idx(index)

def read_csv(csv_file):
	# # Read WSI folder
	# list_files = sorted(list(os.listdir(wsi_folder)))
	# Read CSV file
	file = open(csv_file)
	csvreader = csv.reader(file)
	header = next(csvreader)
	
	labels = []
	surv_times = []
	patients = []
	patient_outcomes = {}
	for row in csvreader:
		name, label, time = row
		labels.append(int(label))
		surv_times.append(float(time))
		patients.append(name)

		patient_outcomes[name] = (int(label), float(time))

	assert len(labels) == len(patients)
	file.close()

	return patient_outcomes, patients, labels, surv_times

def weighted_sampler(target_labels):
	labels_count = np.unique(target_labels, return_counts = True)[1]
	class_weight = 1./ labels_count
	
	samples_weight = class_weight[target_labels]
	samples_weight = torch.from_numpy(samples_weight)
	samples_weight = samples_weight.double()
	sampler = data_utils.WeightedRandomSampler(samples_weight, len(samples_weight))
	return sampler

def get_data_by_patient(root_folder, 
					csv_labels,
					n_tiles = 10000,
					batch_size = 256,
					val_per = 0.2,
					seed = 2334):
	patients_outcomes, patients, status, surv_times = read_csv(csv_labels)
	events = np.array(status)
	times = np.array(surv_times)
	patients = np.array(patients)
	time_bins = pd.qcut(times, q = 4, labels = False)
	stratify_labels = events * 10 + time_bins
	kf = StratifiedShuffleSplit(n_splits = 1, test_size = val_per, random_state = seed)
	
	train_patients_dict = {}
	valid_patients_dict = {}
	for train_index, valid_index in kf.split(X = np.zeros(len(times)), y = events):
		train_patients, valid_patients = patients[train_index], patients[valid_index]
		train_outcomes = [patients_outcomes[patient]for patient in train_patients]
		valid_outcomes = [patients_outcomes[patient]for patient in valid_patients]
		train_patients_dict['fold_0'] = (list(train_patients), train_outcomes)
		valid_patients_dict['fold_0'] = (list(valid_patients), valid_outcomes)
	
	train_patients, train_outcomes = train_patients_dict['fold_0'][0], train_patients_dict['fold_0'][1]
	train_transf = None
	train_dataset = Patient(root_folder, 
							train_patients, 
							train_outcomes,
							transf = train_transf, 
							n_tiles = n_tiles, 
							seed = seed)
	train_dataloader = torch.utils.data.DataLoader(train_dataset, 
													batch_size = batch_size, 
													num_workers=1,
													shuffle = True,)

	valid_patients, valid_outcomes = valid_patients_dict['fold_0'][0], valid_patients_dict['fold_0'][1]
	valid_transf = None
	valid_dataset = Patient(root_folder, 
							valid_patients, 
							valid_outcomes, 
							transf = valid_transf, 
							n_tiles = n_tiles, 
							seed = seed)
	valid_dataloader = torch.utils.data.DataLoader(valid_dataset, 
													batch_size = batch_size, 
													num_workers = 1, 
													shuffle = False,)
													#sampler = valid_weighted_sampler)
	
	print("Total training tiles....: {}".format(len(train_dataset)))
	print("Total validation tiles....: {}".format(len(valid_dataset)))
	
	return train_dataloader, valid_dataloader, train_patients_dict, valid_patients_dict

########################################### Cross-Validation ####################################################
def get_data_cross_validation_SKFold(csv_labels, n_folds = 5, seed = 2452):
    patients_outcomes, patients, status, surv_times = read_csv(csv_labels)
    events = np.array(status)
    times = np.array(surv_times)
    patients = np.array(patients)
    # Combine survival times and event status into a stratification group
    # Discretize survival time (e.g., into quartiles) to reduce the number of unique values
    time_bins = pd.qcut(times, q = 4, labels = False)
    stratify_labels = events * 10 + time_bins

    kf = StratifiedKFold(n_splits = n_folds, shuffle=True, random_state = seed)
    train_patients_dict = {}
    valid_patients_dict = {}
    idx = 0
    for train_index, valid_index in kf.split(X = np.zeros(len(times)), y = stratify_labels):
        train_patients, valid_patients = patients[train_index], patients[valid_index]
        train_outcomes = [patients_outcomes[patient]for patient in train_patients]
        valid_outcomes = [patients_outcomes[patient]for patient in valid_patients]

        train_patients_dict['fold_' + str(idx)] = (list(train_patients), train_outcomes)
        valid_patients_dict['fold_' + str(idx)] = (list(valid_patients), valid_outcomes)
        print("Fold ", idx,  "Train:{}/{}".format(len(train_patients), len(train_outcomes)), "Valid: {}/{}".format(len(valid_patients), len(valid_outcomes)))
        idx += 1
        
    return train_patients_dict, valid_patients_dict

# def get_data_cross_validation_SKFold(csv_labels, 
# 									k_folds = 5, 
# 									seed = 2452):
# 	all_classes_dict, list_labels, patients_wsi_dict = split_patients_per_fold_CV(root_folder,
# 																					csv_labels,
# 																					k_folds,
# 																					seed)
# 	train_patients_dict = {}
# 	valid_patients_dict = {}
# 	for idx in range(k_folds):
# 		all_train_patients = []
# 		all_train_labels = []
# 		all_train_clinicals = []
# 		all_train_outcomes = []
# 		all_valid_patients = []
# 		all_valid_labels = []
# 		all_valid_clinicals = []
# 		all_valid_outcomes = []
# 		for lbl in list_labels:
# 			train_data, valid_data = all_classes_dict[str(lbl)]
# 			# Get train data
# 			train_data_idx = train_data[idx]
# 			train_p, train_l, train_clin, train_out = train_data_idx[0], train_data_idx[1], train_data_idx[2], train_data_idx[3]
# 			#Get valid data
# 			valid_data_idx = valid_data[idx]
# 			valid_p, valid_l, valid_clin, valid_out = valid_data_idx[0], valid_data_idx[1], valid_data_idx[2], valid_data_idx[3]

# 			all_train_patients += list(train_p)
# 			all_train_labels += list(train_l)
# 			all_train_clinicals += list(train_clin)
# 			all_train_outcomes += list(train_out)

# 			all_valid_patients += list(valid_p)
# 			all_valid_labels += list(valid_l)
# 			all_valid_clinicals += list(valid_clin)
# 			all_valid_outcomes += list(valid_out)

# 		print("Fold ", idx, 
# 				"Train:{}/{}".format(len(all_train_patients), len(all_train_labels)), 
# 				"Valid: {}/{}".format(len(all_valid_patients), len(all_valid_labels)))

# 		train_patients_dict['fold_' + str(idx)] = (all_train_patients, all_train_labels, all_train_clinicals, all_train_outcomes)
# 		valid_patients_dict['fold_' + str(idx)] = (all_valid_patients, all_valid_labels, all_valid_clinicals, all_valid_outcomes)

# 	return train_patients_dict, valid_patients_dict, patients_wsi_dict

def get_data_from_cross_validation(root_folder, 
	train_dict,
	valid_dict, 
	n_tiles = 10000,
	batch_size = 1, 
	seed = 2334):
	
	# Read sample files and split
	# print("Load the loaders .........")
	train_patients, train_outcomes = train_dict[0], train_dict[1]
	valid_patients, valid_outcomes = valid_dict[0], valid_dict[1]

	train_transf = None
	train_dataset = Patient(root_folder, 
							train_patients, 
							train_outcomes,
							transf = train_transf, 
							n_tiles = n_tiles, 
							seed = seed)

	#train_status = list(list(zip(*train_labels))[0])
	#train_status = list(list(zip(*train_labels)))
	#train_status = list(map(int, train_status)) # convert list of float number to int
	#train_status = train_labels
	#train_weighted_sampler = weighted_sampler(train_status)
	train_dataloader = torch.utils.data.DataLoader(train_dataset, 
													batch_size = batch_size, 
													num_workers=1,
													shuffle = True,)

	#valid_transf = transforms.Compose([RandomNormalize(prob = 1.0)])
	valid_transf = None
	valid_dataset = Patient(root_folder, 
							valid_patients, 
							valid_outcomes, 
							transf = valid_transf, 
							n_tiles = n_tiles, 
							seed = seed)
	#valid_status = list(list(zip(*valid_labels))[0])
	#valid_status = list(map(int, valid_status))
	#valid_weighted_sampler = weighted_sampler(valid_status)
	valid_dataloader = torch.utils.data.DataLoader(valid_dataset, 
													batch_size = batch_size, 
													num_workers = 1, 
													shuffle = False,)
													#sampler = valid_weighted_sampler)

	print("Total training patients....: {}".format(len(train_dataset)))
	print("Total validation patients....: {}".format(len(valid_dataset)))

	return train_dataloader, valid_dataloader

########################################### End Cross-Validation ####################################################
