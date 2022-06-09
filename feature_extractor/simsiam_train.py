import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import grad_scaler, autocast_mode
import torch.nn.functional as F

import os
import math
import shutil
from datetime import datetime
from functools import partial

import simsiam.loader
import simsiam.builder

import vits

import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

torch.manual_seed(0)


# apex_support = False
# try:
#     sys.path.append('./apex')
#     from apex import amp

#     apex_support = True
# except:
#     print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
#     apex_support = False
    
def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        # copy and put config.yaml to the model_checkpoints_folder
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

class SimSiam_train(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device() # set to cuda if available
        self.writer = SummaryWriter() # use tensorboard
        self.dataset = dataset

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device

    def adjust_learning_rate(self, optimizer, init_lr, epoch, epochs):
    # Decay the learning rate based on schedule
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
        for param_group in optimizer.param_groups:
            if 'fix_lr' in param_group and param_group['fix_lr']:
                param_group['lr'] = init_lr
            else:
                param_group['lr'] = cur_lr

    def train(self):

        start_time = datetime.now()
        print("Start Time: {start_time}")

        print("train")
        # get train and valid loader from function inside of dataset
        train_loader, valid_loader = self.dataset.get_data_loaders()

        model = simsiam.builder.SimSiam(
            models.__dict__[self.config['simsiam_arch']],
            self.config['simsiam_dim'], self.config['simsiam_pred_dim'])
        
        # encoder = models.resnet18()
        # model = simsiam.builder.SimSiam(encoder)

        if self.config['n_gpu'] > 1:
            model = torch.nn.DataParallel(model, device_ids=eval(self.config['gpu_ids']))

        # check if there is any pretrained weights
        model = self._load_pre_trained_weights(model)
        model = model.to(self.device)
        
        criterion = nn.CosineSimilarity(dim=1).cuda()

        init_lr = self.config['learning_rate'] * self.config['batch_size'] / 256

        # if args.fix_pred_lr:
        #     optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
        #                     {'params': model.module.predictor.parameters(), 'fix_lr': True}]
        # else:
        #     optim_params = model.parameters()

        # change the model parameters depend on fix_pred_lr
        optimizer = torch.optim.SGD([{'params': model.encoder.parameters(), 'fix_lr': False},
                             {'params': model.predictor.parameters(), 'fix_lr': True}], init_lr,
                                momentum=0.9,
                                weight_decay=1e-4)

        model_checkpoints_folder = os.path.join('../../graph_transformer/runs/simsiam', self.writer.log_dir)

        # save config file
        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        valid_n_iter = 0
        # initialize at first to infinity
        best_valid_loss = np.inf
        scaler = grad_scaler.GradScaler()
        epochs = self.config['epochs']
        # epochs:20_
        for epoch_counter in range(self.config['epochs']):
            self.adjust_learning_rate(optimizer, init_lr, epoch_counter, epochs)
            train_loss = 0
            for batch_idx, (xis, xjs) in enumerate(train_loader):
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)
                
                # compute output and loss
                p1, p2, z1, z2 = model(x1=xis, x2=xjs)
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                train_loss += loss.item() * xis.size(0)

                # z1 = model(xis)
                # z2 = model(xjs)
                # loss = -1 * cosine_similarity(z1, z2)
                # train_loss += loss.item()

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print("[%d/%d] step: %d train_loss: %.3f" % (epoch_counter, self.config['epochs'], n_iter, loss.item() * xis.size(0)))
                n_iter += 1
                
            print("TRAIN LOSS:", train_loss / len(train_loader))

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader, criterion)
                print("[%d/%d] val_loss: %.3f" % (epoch_counter, self.config['epochs'], valid_loss))
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                    print('saved at {}'.format(os.path.join(model_checkpoints_folder, 'model.pth' )))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

        print(f"Training Execution time: {datetime.now() - start_time}")
        
    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('../../graph_transformer/runs/simsiam/runs', self.config['fine_tune_from'])
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader, criterion):

        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0

            for (xis, xjs) in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                # SimSiam
                # compute output and loss
                p1, p2, z1, z2 = model(x1=xis, x2=xjs)
                loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                valid_loss += loss.item() * xis.size(0) #loss.item()
                counter += 1
            valid_loss /= counter
        return valid_loss