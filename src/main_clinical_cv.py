import sys
import os
print(sys.path)

import math
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import numpy as np
import socket
import torch.distributed as dist
import time
import random
from datetime import datetime
import helpers as help
from model_pl_clinical import DeepNN_Model

#import bergonie_dataloader_survival_wsi_clinical as pdata
import bergonie_dataloader_survival_wsi as pdata

def cross_validation(hparams):

    config = "CONCH_STUMP_patients_{}CV_{}_clinical_MIDL_10_heads".format(hparams.n_folds, hparams.nb_clinical)
    if hparams.m_get_data == 'SKFold':
        print("Get data by SKFold CV.")
        train_patients_dict, valid_patients_dict = pdata.get_data_cross_validation_SKFold(hparams.npz_labels, 
                                                                                                    #hparams.npz_train,
                                                                                                    n_folds = hparams.n_folds,
                                                                                                    seed = hparams.seed)
    
    #print(valid_patients_dict)
    assert(len(train_patients_dict) == len(valid_patients_dict))
    
    for idx in range(hparams.n_folds):
        # if idx < 2:
        # 	continue
        print("Load the loaders fold {} .........".format(idx))

        # Cross-validation normal
        train_dict = train_patients_dict['fold_' + str(idx)]
        valid_dict = valid_patients_dict['fold_' + str(idx)]

        # Save the patients for training and validation
        root_export_csv = os.path.join('exports_stumps_012025', config, f'Fold_{idx}')
        help.export_to_csv_from_dict(train_dict, output_dir = root_export_csv, file_name = 'train.csv')
        help.export_to_csv_from_dict(valid_dict, output_dir = root_export_csv, file_name = 'valid.csv')
        
        # Load dataloaders
        train_loader_idx, valid_loader_idx = pdata.get_data_from_cross_validation(hparams.npz_train,
                                                                          train_dict,
                                                                          valid_dict, 
                                                                          #patients_wsi,
                                                                          n_tiles = hparams.n_tiles, 
                                                                          batch_size = hparams.batch_size, 
                                                                          seed = hparams.seed)
        
        # Cross-validation modification: valid -> test/ used 20% of train as validation
        # train_dict = train_patients_dict['fold_' + str(idx)]
        # train_loader_idx, valid_loader_idx= pdata.split_train_data(hparams.npz_train, 
        #      train_dict,
        #      patients_wsi, 
        #      n_tiles = hparams.n_tiles,
        #      batch_size = hparams.batch_size,  
        #      val_per = 0.2, 
        #      seed = hparams.seed)
        
        model = DeepNN_Model(hparams, train_loader_idx, valid_loader_idx)
        estop_callback = EarlyStopping(monitor = 'val_loss', mode ='min', min_delta = 0.0000, patience = 10, verbose = True)
        chp_callback = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min",)
        
        tb_logger = pl_loggers.TensorBoardLogger(save_dir = 'lightning_logs_Transformer_012025/{}/'.format(config))
        trainer = pl.Trainer(max_epochs=hparams.epochs, \
                            weights_summary='top', \
                            gpus = hparams.n_gpus, \
                            #accelerator = 'gpu', \
                            #strategy= "ddp", \
                            #amp_level = "O1", \
                            precision = 16, \
                            callbacks = [chp_callback, estop_callback],
                            num_sanity_val_steps = 0,
                            #replace_sampler_ddp=False,
                            logger = tb_logger,)
        trainer.fit(model, train_loader_idx, valid_loader_idx)
        
def main(hparams):
    # use a random seed
    if hparams.seed == -1:
        hparams.seed = random.randint(0,5000)
        print('The SEED number was randomly set to {}'.format(hparams.seed))

    torch.manual_seed(hparams.seed)
    np.random.seed(hparams.seed)
    random.seed(hparams.seed)

    # print(hparams)
    # torch.cuda.empty_cache()
    # cross_validation(hparams)
    
    # To test different learning rates
    n_folds_list = [5]
    n_clins = [0]
    for f in n_folds_list:
        hparams.n_folds = f
        for nc in n_clins:
            hparams.nb_clinical = nc
            print(hparams)
            torch.cuda.empty_cache()
            cross_validation(hparams)

if __name__ == '__main__':
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    main_arg_parser = argparse.ArgumentParser(description="parser for observation generator", add_help=False)
    main_arg_parser.add_argument("--log-interval", type=int, default=500,
                                  help="number of images after which the training loss is logged, default is 500")
    main_arg_parser.add_argument("--checkpoint-interval", type=int, default=500,
                                  help="number of batches after which a checkpoint of the trained model will be created")

    main_arg_parser.add_argument('--epochs', default = 100, type=int)
    main_arg_parser.add_argument('--n_gpus', default = 1, type=int)
    main_arg_parser.add_argument('--learning_rate', default=0.00001, type=float) # 0.0023
    main_arg_parser.add_argument('--w_decay', default=1e-4, type=float)
    main_arg_parser.add_argument('--batch_size', default=64, type=int)
    main_arg_parser.add_argument('--n_folds', default= 5, type=int)

    #main_arg_parser.add_argument('--npz_train', default = '/media/monc/LaCie/Sabrina/Sabrina_STUMPs/Tiles_STUMPs/40X_102024/CONCH_features', type=str)
    main_arg_parser.add_argument('--npz_train', default = '/beegfs/vle/Sabrina_Croce/GSMT_data/STUMP_data/CONCH_features', type=str) 
    #main_arg_parser.add_argument('--npz_train', default = '/beegfs/vle/data/uterus_tumor_20X/CONCH_features', type=str) # Clusters_features Centroids_77 selection
    #
    # cls_os_STUMP_with_clinical.csv
    # cls_relapse_STUMP_with_clinical.csv
    # cls_relapse_STUMP_with_clinical_no_IGR.csv
    #main_arg_parser.add_argument('--npz_labels', default = '/beegfs/vle/Sabrina_Croce/GSMT_data/v10_112024/rfs_LMS_STUMP.csv', type=str)
    main_arg_parser.add_argument('--npz_labels', default = '/beegfs/vle/Sabrina_Croce/GSMT_data/STUMP_data/csv_102024/cls_relapse_STUMP_without_clinical.csv', type=str)
    main_arg_parser.add_argument('--init_features', default = 512, type=int)
    main_arg_parser.add_argument('--m_get_data', default = 'SKFold', type=str) # KFold or SKFold

    main_arg_parser.add_argument('--seed', default = 2452, type=int)
    main_arg_parser.add_argument('--n_tiles', default = 1000, type=int)

    main_arg_parser.add_argument('--fc_1', default = 256, type=int)
    main_arg_parser.add_argument('--fc_2', default = 128, type=int)
    main_arg_parser.add_argument('--num_classes', default = 1, type=int) 
    
    # add model specific args i
    parser = DeepNN_Model.add_model_specific_args(main_arg_parser, os.getcwd())
    hyperparams = parser.parse_args()

    main(hyperparams)

