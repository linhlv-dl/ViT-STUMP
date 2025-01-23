import sys
import os
import torch.nn
import torch 
from torchvision import transforms

from model_attention_survival import NN_Model2aplus_Clinical, NN_Model2aplus
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import auc, RocCurveDisplay
from sklearn.utils import shuffle
from PIL import Image
from datetime import datetime
import pickle
import csv
import sklearn.metrics as sm
from loss_cox import concordance_index

from predictor import Predictor
from tiles_extraction import *
from ViT import ViT
from metrics import c_index_sksurv


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def prediction(model, patient_tensor, batch_size = 512):
    """Predict the risk for one patient by consindering all tiles of patients. 
       Due to the limitation of GPU capacity, we consider small batch_size and
       compute the average of all prediction.

    Args:
        model : Trained model
        patient_tensor (_type_): Inputted patient tensor
        batch_size (int, optional): The number of selected tiles for one time process. Defaults to 512.

    Returns:
        float: The risk score equals to the average risk scores of all batch_sizes.
    """
    model.to(device)
    all_predictions = []
    n_tiles = patient_tensor.shape[0]
    for idx in range(0, n_tiles, batch_size):
        input_tensor = patient_tensor[idx:idx + batch_size]
        input_tensor = input_tensor.unsqueeze(0)
        pred = model(input_tensor.to(device))
        all_predictions.append(pred.cpu().detach().numpy())

    all_predictions = np.concatenate(all_predictions, axis = 0)
    avg_pred = np.average(all_predictions)
    return avg_pred

def inference_validation(model, root_folder, 
                            labels_dict, 
                            patient_wsi_dict,
                            n_tiles = -1):
    patient_list = list(labels_dict)
    #print(FOLD_IDX + ' = ',patient_list)

    predictions = []
    with torch.no_grad():
        model.eval()
        for ptx in range(len(patient_list)):
            patient = patient_list[ptx]
            
            wsi_file, rd_indices = _setup_bag_2(root_folder, patient_wsi_dict, patient)
            inf_patient = torch.from_numpy(wsi_file)
    
            # predict
            if n_tiles == -1:
                #print("Get all tiles !", inf_patient.shape)
                inf_patient = inf_patient.unsqueeze(0)
                # Move to the GPU
                # We can run the test on CPU (by disable 2 following rows) but it is slow
                #model.to(device)
                #inf_patient= inf_patient.to(device)

                risk_pred = model(inf_patient)
                print(risk_pred)
                predictions.append(risk_pred.cpu().detach().item())
            else:
                #print("Not enough memory, process on batch and take average.")
                risk_pred = prediction(model, inf_patient, batch_size = n_tiles)
                predictions.append(risk_pred)
    print(predictions)
    return predictions

def read_row(row, nb_clinicals = 1):
    if nb_clinicals == 0:
        name, label, time = row
        cli_list = []
    elif nb_clinicals == 1:
        name, label, time, mit = row
        cli_list = [float(mit)]
    elif nb_clinicals == 2:
        name, label, time, mit, tc = row
        cli_list = [float(mit), float(tc)]
    elif nb_clinicals == 3:
        name, label, time, mit, tc, at = row
        cli_list = [float(mit), float(tc), float(at)]
    elif nb_clinicals == 4:
        name, label, time, mit, tc, at, lvi = row
        cli_list = [float(mit), float(tc), float(at), float(lvi)]
    return name, label, time, cli_list

def read_row_noGT(row, nb_clinicals = 1):
    if nb_clinicals == 0:
        name= row[0]
        cli_list = []
    elif nb_clinicals == 1:
        name, mit = row
        cli_list = [float(mit)]
    elif nb_clinicals == 2:
        name, mit, tc = row
        cli_list = [float(mit), float(tc)]
    elif nb_clinicals == 3:
        name, mit, tc, at = row
        cli_list = [float(mit), float(tc), float(at)]
    elif nb_clinicals == 4:
        name, mit, tc, at, lvi = row
        cli_list = [float(mit), float(tc), float(at), float(lvi)]
    return name, cli_list, 0, 0.0

def read_csv(csv_file, wsi_folder, with_GT = False):
    print("Testing patients from CSV")
    # Read WSI folder
    list_files = sorted(list(os.listdir(wsi_folder)))

    # Read CSV file
    file = open(csv_file)
    csvreader = csv.reader(file)
    header = next(csvreader)
    print(header)
    if with_GT:
        nclinical = len(header) - 3
    else:
        nclinical = len(header) - 1
    print("Number of clinicals: ", nclinical)
    
    label_patients_dict = {}
    patients_wsi_dict = {}
    patients_clinical = {}
    surv_times = []
    patients = []
    patient_outcomes = {}
    for row in csvreader:
        if with_GT:
            name, label, time, cli_list = read_row(row, nb_clinicals = nclinical)
        else:
            name, cli_list, label, time = read_row_noGT(row, nb_clinicals = nclinical)
        
        surv_times.append(float(time))
        patients.append(name)
        #patients[name] = int(label)
        wsi_files = [f_name for f_name in list_files if name in f_name]
        patients_wsi_dict[name] = wsi_files
        
        patients_clinical[name] = np.array(cli_list)
        #label = 1.0 if label == 'True' or label == 1 else 0.0
        patient_outcomes[name] = (int(label), float(time))

    file.close()

    return patients, patients_wsi_dict, patients_clinical, patient_outcomes

def _read_npz(npz_file, label = 0):
    npz_array = np.load(npz_file)['arr_0']
    return npz_array

def _load_wsis(root_folder, list_wsis, seed = 2452):
    all_wsi = []
    for wsi in list_wsis:
        wsi_path = os.path.join(root_folder, wsi)
        c_files= _read_npz(wsi_path)
        all_wsi.append(c_files)
    all_wsi = np.concatenate(all_wsi, axis = 0)
    return all_wsi

def _setup_bag_2(root, patient_wsi_dict, pname, n_tiles = -1, seed = 2452): 
    #print("Get bag of {}".format(self.region))
    list_wsis = patient_wsi_dict[pname]
    wsi_file = _load_wsis(root, list_wsis)
    return wsi_file, np.arange(wsi_file.shape[0])
    #selected_tiles, rd_indices = random_select_tiles_wsi(wsi_file, num_tiles = n_tiles)
    #return selected_tiles, rd_indices

def random_select_tiles_wsi(np_array, num_tiles = 10000):
    n_rows = np_array.shape[0]
    if n_rows < num_tiles or num_tiles == -1:
        return np_array, np.arange(n_rows)

    rd_indices = np.random.choice(n_rows, size = num_tiles, replace = True)
    selected_array = np_array[rd_indices,:]

    return selected_array, rd_indices

def load_model(ckpt_path):
    # Load model and create the predictor
    checkpoint = torch.load(ckpt_path)
    for key in list(checkpoint['state_dict'].keys()):
        new_key = key[6:]
        checkpoint['state_dict'][new_key] = checkpoint['state_dict'].pop(key)
    
    model = ViT(num_classes=1, input_dim=512, pool='cls')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model

def export_to_csv(filename, patients, outcomes, predictions):
    with open(filename, 'w') as f:
        write = csv.writer(f)
        head = ['Patient', 'Status', 'Time', 'Risk']
        write.writerow(head)

        for idx in range(len(patients)):
            patient = patients[idx]
            pred = predictions[idx]
            outcome = list(outcomes[patient])
            row = [patient] + outcome + [pred]
            write.writerow(row)


def load_patient_dict_and_inference(ckpt_path, 
                                    root_folder, 
                                    csv_patient, 
                                    nb_tile,
                                    fold_idx, 
                                    save_folder):
    ext = csv_patient.split('.')[-1]
    if ext == 'csv':
       patients, patients_wsi_dict, _, patient_outcomes = read_csv(csv_patient, root_folder, with_GT = True)
    print("Fold ", fold_idx, ": ", list(patients))
    model = load_model(ckpt_path)

    basename = os.path.basename(csv_patient)
    y_test_preds  = inference_validation(model,
                                    root_folder,
                                    patients,
                                    patients_wsi_dict,
                                    n_tiles = nb_tile)
    save_test = os.path.join(save_folder, basename)
    export_to_csv(save_test, patients, patient_outcomes,  y_test_preds)
    
    # Compute the c-index
    y_status = [patient_outcomes[p][0] for p in patients]
    y_status = np.array(y_status)
    y_survtimes = [patient_outcomes[p][1] for p in patients]
    y_survtimes = np.array(y_survtimes)
    y_test_preds = np.array(y_test_preds)
    print(y_status.shape, y_survtimes.shape, y_test_preds.shape)

    y_status = torch.from_numpy(y_status)
    y_survtimes = torch.from_numpy(y_survtimes)
    y_test_preds = torch.from_numpy(y_test_preds)
    cindex = concordance_index(y_survtimes, y_status, -y_test_preds)
    #cindex = 0.0
    return cindex.item()

root = '/beegfs/vle/Sabrina_Croce/GSMT_Survival/lightning_logs_Transformer_112024_fine_tuning/STUMP_relapse_at_10y/default/'
version = 'version_2'
ckpt_path = root + f'{version}/checkpoints/epoch=3-step=11.ckpt'

data_in = 'rfs_STUMP_relapse_at_10y_0.4'
testing_patients = f'/beegfs/vle/Sabrina_Croce/GSMT_Survival/exports_stumps_112024_fine_tuning/STUMP_relapse_at_10y/{data_in}/valid.csv'
#testing_patients = '/beegfs/vle/Sabrina_Croce/GSMT_data/v10_112024/rfs_STUMP_IGR_ORH.csv'

nb_tile = 1000
root_folder = '/beegfs/vle/data/uterus_tumor_20X/CONCH_features'
save_folder = f'/beegfs/vle/Sabrina_Croce/GSMT_Survival/exports_csv_v7_stumps_112024_fine_tuning/STUMP_relapse_at_10y/{data_in}'
os.makedirs(save_folder, exist_ok = True)

test_cindex = load_patient_dict_and_inference(ckpt_path,
                                    root_folder, 
                                    testing_patients, 
                                    nb_tile,
                                    0,
                                    save_folder)
print("Test C-index: ", test_cindex)
print("Finish !!!!")

