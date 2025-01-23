import os
import torch
import torch.utils.data as data_utils
from torch.nn import functional as F
import pytorch_lightning as pl
from argparse import ArgumentParser
import sklearn.metrics as sm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

from ViT import ViT

import bergonie_dataloader_survival_wsi as pdata
from loss_cox import cox_loss_custom, concordance_index

import helpers as help

class DeepNN_Model(pl.LightningModule):
    def __init__(self, hparams, train_loader = None, valid_loader = None):
        super().__init__()
        
        self.save_hyperparameters(hparams)
        print("global seed: ", os.environ.get("PL_GLOBAL_SEED"))
        
        # Create the model
        self.model = ViT(num_classes=hparams.num_classes, input_dim=hparams.init_features, heads = 10,  pool='cls')
            
        # For fine-tuning
        # hparams.pretrained = '/beegfs/vle/Sabrina_Croce/GSMT_Survival/lightning_logs_STUMPs_Transformer/PFS/UNI_batch_noIGR_lr_00001_bs_64_25102024/PFS_20K_5CV_0_clinical_stop_by_loss/default/version_4/checkpoints/epoch=18-step=18.ckpt'
        # hparams.frozen = False
        # if hparams.pretrained != None:
        #     print("Load the model from a pre-trained model")
        #     self.model = self.load_model(self.model, hparams.pretrained)

        # if hparams.frozen:
        #     print("Freeze the layers")
        #     print(self.model)
        #     self.model = self.freeze_model()
        #     #self.freeze_core()
        # else:
        #     print("No freeze any layer.")

        ## DataLoader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        if self.train_loader == None and self.valid_loader == None:
            self.train_loader, self.valid_loader, train_dict, valid_dict = pdata.get_data_by_patient(hparams.npz_train, 
                                                            hparams.npz_labels, 
                                                            n_tiles = hparams.n_tiles,
                                                            batch_size = hparams.batch_size, 
                                                            val_per = hparams.val_per,
                                                            seed = hparams.seed)
            # Save the patients for training and validation
            folder_export = os.path.basename(hparams.npz_labels).split(".")[0] + f"_{hparams.val_per}"
            root_export_csv = os.path.join('exports_stumps_112024_fine_tuning', 'STUMP_relapse_at_10y/' + folder_export)
            help.export_to_csv_from_dict(train_dict['fold_0'], output_dir = root_export_csv, file_name = 'train.csv')
            help.export_to_csv_from_dict(valid_dict['fold_0'], output_dir = root_export_csv, file_name = 'valid.csv')

        self.loss_f = cox_loss_custom
        self.iter = 0
        self.lbl_pred_each = []
        self.survtime_all = []
        self.status_all = []
        #self.automatic_optimization = False

    def load_model(self, model, pretrained_chkpoint):
        checkpoint = torch.load(pretrained_chkpoint)
        print(checkpoint['state_dict'].keys())
        for key in list(checkpoint['state_dict'].keys()):
            new_key = key[6:]
            checkpoint['state_dict'][new_key] = checkpoint['state_dict'].pop(key)    
        
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def freeze_model(self):
        for name, param in self.model.named_parameters():
            if 'mlp_head' not in name:
                param.requires_grad = False
        return self.model

    def freeze_core(self):
        self.model.conv_extractor.requires_grad_(False)
        for i in range(4):
            self.model.classifier[i].requires_grad_(False)

    # Delegate forward to underlying model
    def forward(self, x):
        y_prob = self.model(x)
        return y_prob

    # Train on one batch
    def training_step(self, batch, batch_idx):
        x, y_status, y_survtime = batch
        y_prob = self.forward(x)

        loss = self.loss_f(y_survtime, y_status, y_prob)
        tcindex = concordance_index(y_survtime, y_status, -y_prob.detach())
        #tcindex = c_index_sksurv(y_prob.detach(), y_survtime, y_status)
        #lcindex = c_index_lifelines(y_prob.detach(), y_survtime, y_status)

        tensorboard_logs = {'train_loss':loss,
                            'train_cindex': tcindex,}
                            #'train_cindex_ll':lcindex}
        self.log('train_loss', loss)
        self.log('train_cindex', tcindex)
        #self.log('train_cindex_lifelines', lcindex)

        return {'loss': loss, 
                'log': tensorboard_logs,
                'time': y_survtime,
                'status':y_status,
                'y_pred': y_prob, 
                'batch_idx': batch_idx}

    
    # Validate on one batch
    
    def validation_step(self, batch, batch_idx):
        x, y_status, y_survtime = batch
        y_prob = self.forward(x)
        val_loss = self.loss_f(y_survtime, y_status, y_prob)
        #print(y_prob.size())
        cindex = concordance_index(y_survtime, y_status, -y_prob.detach())
        #cindex = c_index_sksurv(y_prob.detach(), y_survtime, y_status)
        #lcindex = c_index_lifelines(y_prob.detach(), y_survtime, y_status)

        return {'val_loss': val_loss,
                'cindex': cindex,}
                #'lcindex': lcindex}   

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_cindex = torch.stack([x['cindex'] for x in outputs]).mean()
        #avg_lcindex = torch.stack([x['lcindex'] for x in outputs]).mean()
       
        tensorboard_logs = {'loss': avg_loss, 
                            'c-index':avg_cindex,}
                            #'c-index-life':avg_lcindex,}

        self.log('val_loss',avg_loss)
        self.log('val_cindex',avg_cindex)
        #self.log('val_cindex_lifelines',avg_lcindex)

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    # Setup optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr = self.hparams.learning_rate,
                                    weight_decay = self.hparams.w_decay)
        #scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 20, verbose = True)
        lr_schedulers = {"scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                                mode = 'min', 
                                                                                patience = 5, 
                                                                                factor = 0.1, 
                                                                                min_lr = 1e-7, 
                                                                                verbose = True), 
                        "monitor": "val_loss"}
        return [optimizer],[lr_schedulers]
    
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader


    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--name', default='Deep NN model for classification', type=str)
        return parser


