import random
import math
import time
import pandas as pd
import numpy as np
import os
import argparse
import yaml

import torch
import torch.utils.data as data
import torch.nn as nn
#import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

from models.pspnet import PSPNet

DATASET_NCLASS_ADE = 150
DATASET_NCLASS_VOC = 21

prm_rand_seed = 1234
torch.manual_seed(prm_rand_seed)
np.random.seed(prm_rand_seed)
random.seed(prm_rand_seed)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config/cfg_sample_pspnet.yaml', help='config file')
    return parser

def train(cfg):
    cfg['model'] = 'pspnet_' + cfg['padding_mode']

    print('[cfg.padding mode] ' + cfg['padding_mode'])
    print('[cfg.model] ' + cfg['model'])
    print('[cfg.outputs] ' + cfg['outputs'])

    #----- DataLoader ------#

    # DataTransform_2: input image is expanded to expanded_size before cropping into input_size
    #from utils.dataloader import make_datapath_list, DataTransform, VOCDataset
    from utils.dataloader import make_datapath_list, DataTransform_2, VOCDataset

    # filepath list
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
        rootpath=cfg['dataset'])

    # Dataset
    color_mean = cfg['color_mean']
    color_std = cfg['color_std']
    input_size = cfg['input_size']
    expanded_size = cfg['expanded_size']
    train_dataset = VOCDataset(
        train_img_list, train_anno_list, phase="train", 
        #transform=DataTransform(input_size=input_size, color_mean=color_mean, color_std=color_std)
        transform=DataTransform_2(input_size=input_size, expanded_size=expanded_size, color_mean=color_mean, color_std=color_std)
        )
    val_dataset = VOCDataset(
        val_img_list, val_anno_list, phase="val", 
        #transform=DataTransform(input_size=input_size, color_mean=color_mean, color_std=color_std)
        transform=DataTransform_2(input_size=input_size, expanded_size=expanded_size, color_mean=color_mean, color_std=color_std)
        )

    # DataLoader
    batch_size = cfg['batch_size'] #6
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

    #----- Network Model ------#

    # PSPNet
    [dataset_name, weights_file_path] = cfg['pretrain']
    if dataset_name == 'ADE':
        # Pretrained model with ADE20K dataset (150 Classes)
        net = PSPNet(n_classes=DATASET_NCLASS_ADE, padding_mode=cfg['padding_mode'])
        state_dict = torch.load(weights_file_path)
        net.load_state_dict(state_dict, strict=False)

        # Replace last layers for classification into the layers with 21 classes
        net.decode_feature.classification = nn.Conv2d(in_channels=512, out_channels=DATASET_NCLASS_VOC, kernel_size=1, stride=1, padding=0)
        net.aux.classification = nn.Conv2d(in_channels=256, out_channels=DATASET_NCLASS_VOC, kernel_size=1, stride=1, padding=0)

        # Initialize replaced convolution layers
        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias != None:  # with bias
                    nn.init.constant_(m.bias, 0.0)

        # Apply the classification layers in the model
        net.decode_feature.classification.apply(weights_init)
        net.aux.classification.apply(weights_init)

    elif dataset_name == 'VOC':
        net = PSPNet(n_classes=DATASET_NCLASS_VOC, padding_mode=cfg['padding_mode'])
        state_dict = torch.load(weights_file_path)
        net.load_state_dict(state_dict, strict=False)

    else:
        print('Pretrain model should be ADE or VOC.')
        return

    # only for padding_mode = CAP
    # Fix network weights except for CAP full-connection layers in cap_pretrain, and vice versa in cap_train
    if cfg['padding_mode'] in ('cap_pretrain', 'cap_train'):
        lock_list = ['CAP_up','CAP_down','CAP_left','CAP_right']
        for name , param in net.named_parameters():
            name_type = name.split('.')
            if len(name_type) >=4:
                for layer_name in lock_list:
                    if layer_name == name_type[-3]:
                        param.requires_grad = True if cfg['padding_mode'] == 'cap_pretrain' else False
                        break
                    else:
                        param.requires_grad = False if cfg['padding_mode'] == 'cap_pretrain' else True

    print('[Network model] Pretrained weights were loaded.')
    print(cfg['pretrain'])
    net

    #----- Loss Function ------#

    class PSPLoss(nn.Module):
        def __init__(self, aux_weight=0.4):
            super(PSPLoss, self).__init__()
            self.aux_weight = aux_weight  # Weight for aux_loss

        def forward(self, outputs, targets):
            """
            Calculate Loss Function

            Parameters
            ----------
            outputs: output of PSPNet (tuple)
                (output=torch.Size([num_batch, 21, 475, 475]), output_aux=torch.Size([num_batch, 21, 475, 475]))。
            targets : Annotations [num_batch, 475, 475]

            Returns
            -------
            loss: value of loss function
            """

            loss = F.cross_entropy(outputs[0], targets, reduction='mean')
            loss_aux = F.cross_entropy(outputs[1], targets, reduction='mean')

            return loss+self.aux_weight*loss_aux

    criterion = PSPLoss(aux_weight=0.4)

    #----- Optimizer ------#

    if cfg['optimizer'] == 'sgd':
        optimizer = optim.SGD([
            {'params': net.feature_conv.parameters(), 'lr': 1e-3},
            {'params': net.feature_res_1.parameters(), 'lr': 1e-3},
            {'params': net.feature_res_2.parameters(), 'lr': 1e-3},
            {'params': net.feature_dilated_res_1.parameters(), 'lr': 1e-3},
            {'params': net.feature_dilated_res_2.parameters(), 'lr': 1e-3},
            {'params': net.pyramid_pooling.parameters(), 'lr': 1e-3},
            {'params': net.decode_feature.parameters(), 'lr': 1e-2},
            {'params': net.aux.parameters(), 'lr': 1e-2},
        ], momentum=0.9, weight_decay=0.0001)

    elif cfg['optimizer'] == 'adam':
        optimizer = optim.Adam([
            {'params': net.feature_conv.parameters(), 'lr': 1e-4},
            {'params': net.feature_res_1.parameters(), 'lr': 1e-3},
            {'params': net.feature_res_2.parameters(), 'lr': 1e-3},
            {'params': net.feature_dilated_res_1.parameters(), 'lr': 1e-3},
            {'params': net.feature_dilated_res_2.parameters(), 'lr': 1e-3},
            {'params': net.pyramid_pooling.parameters(), 'lr': 1e-3},
            {'params': net.decode_feature.parameters(), 'lr': 1e-2},
            {'params': net.aux.parameters(), 'lr': 1e-2},
        ], betas=(0.9, 0.999), weight_decay=0.0001)

    # Scheduler
    def lambda_epoch(epoch):
        return math.pow((1-epoch/cfg['num_epochs']), 0.9)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)

    #----- train/validation ------#

    def train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Device: ', device)

        # Send network to GPU
        net.to(device)

        # Auto-tuning of network structure
        torch.backends.cudnn.benchmark = True

        # Number of Batches
        num_train_imgs = len(dataloaders_dict["train"].dataset)
        num_val_imgs = len(dataloaders_dict["val"].dataset)
        batch_size = dataloaders_dict["train"].batch_size

        # Initialization of iteration counter
        iteration = 1
        logs = []

        # Update network weights after multiple minibatches
        batch_multiplier = cfg['batch_multiplier'] #4

        # Output folder
        if not os.path.exists(cfg['outputs']):
            os.makedirs(cfg['outputs'])

        # Loop for epoch
        #for epoch in tqdm(range(num_epochs)):
        for epoch in range(num_epochs+1):
            # Starting time
            t_epoch_start = time.time()
            t_iter_start = time.time()
            epoch_train_loss = 0.0  # Train loss for this epoch
            epoch_val_loss = 0.0  # Val loss for this epoch

            print('-------------')
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-------------')

            # train / val
            for phase in ['train', 'val']:
                if phase == 'train':
                    if epoch == 0:
                        # only val for 0th loop
                        continue
                    else:
                        net.train()  # Train mode
                        #scheduler.step()  # Update Scheduler for optimizer
                        optimizer.zero_grad()
                        print('（train）')
                else: # validation
                    if (epoch % cfg['val_output_interval'] == 0):
                        net.eval()   # Validation mode
                        print('-------------')
                        print('（val）')
                    else: 
                        # Calculate validation every 5 epoches
                        continue

                count = batch_multiplier #0 # multiple minibatch
                for images, anno_class_images in dataloaders_dict[phase]:
                    # To avoid mini-batch with minibatch size = 1, due to the error in batch normalization
                    if images.size()[0] == 1:
                        continue

                    # Send data to GPU if possible
                    images = images.to(device)
                    #print(type(images))
                    anno_class_images = anno_class_images.to(device)
                    
                    # Update paramters after multiple minibatch (count = batch_multiplier)
                    if (phase == 'train') and (count == 0):
                        optimizer.step()
                        optimizer.zero_grad() # Reset optimizer
                        count = batch_multiplier

                    # Forward calculation
                    with torch.set_grad_enabled(phase == 'train'):
                        if cfg['padding_mode'] == 'cap_pretrain':
                            [output, output_aux, cap_loss] = net(images)
                            loss = cap_loss / batch_multiplier
                        else:
                            outputs = net(images)
                            loss = criterion(outputs, anno_class_images.long()) / batch_multiplier

                        # Back propagation in 'train'
                        if phase == 'train':
                            loss.backward()  # calculate grad with back propagation
                            count -= 1  # multiple minibatch

                            if (iteration % 10 == 0):  # Display loss every 10iter
                                t_iter_finish = time.time()
                                duration = t_iter_finish - t_iter_start
                                print('Iteration {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                                    iteration, loss.item()/batch_size*batch_multiplier, duration))
                                t_iter_start = time.time()

                            epoch_train_loss += loss.item() * batch_multiplier
                            iteration += 1
                        else:
                            # Validation
                            epoch_val_loss += loss.item() * batch_multiplier
                
                if phase == 'train':
                    scheduler.step()  # Update Scheduler

            # Calculate loss and accuracy for train and val each epoch
            log_epoch_train_loss = epoch_train_loss/num_train_imgs
            log_epoch_val_loss = epoch_val_loss/num_val_imgs

            t_epoch_finish = time.time()
            print('-------------')

            def save_network_weights(net, cfg):
                weights_path = cfg['outputs'] + cfg['model'] + '_best.pth'
                torch.save(net.state_dict(), weights_path)
                print('network weights were saved: ' + weights_path)
                return weights_path

            # check val_loss and save network weights if smaller val_loss
            if epoch == 0:
                # save initial model
                min_val_loss = log_epoch_val_loss
                min_epoch = 0
                weights_path = save_network_weights(net, cfg)
                print('epoch {} || Epoch_TRAIN_Loss:{:.4f} || Epoch_VAL_Loss:{:.4f}'.format(epoch, log_epoch_train_loss, log_epoch_val_loss))
            elif (epoch % cfg['val_output_interval'] == 0):
                if (log_epoch_val_loss < min_val_loss):
                    min_val_loss = log_epoch_val_loss
                    min_epoch = epoch
                    weights_path = save_network_weights(net, cfg)
                print('epoch {} || Epoch_TRAIN_Loss:{:.4f} || Epoch_VAL_Loss:{:.4f}'.format(epoch, log_epoch_train_loss, log_epoch_val_loss))
            else:
                print('epoch {} || Epoch_TRAIN_Loss:{:.4f}'.format(epoch, log_epoch_train_loss))

            print('min_epoch {} || min_val_loss:{:.4f}'.format(min_epoch, min_val_loss))
            print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))

            # Save log file
            log_epoch = {'epoch': epoch, 'train_loss': log_epoch_train_loss, 'val_loss': log_epoch_val_loss}
            logs.append(log_epoch)
            df = pd.DataFrame(logs)
            df.to_csv(cfg['outputs'] + cfg['model'] + '_log_output.csv')

            t_epoch_start = time.time()

        print('-------------')
        print('min_epoch: ' + str(min_epoch))
        print('weights_path: ' + str(weights_path))

        # save final model. Since val data is also used in the evaluation, the final model should be used in the evaluation, instead of the best model based on the val data.
        weights_path = cfg['outputs'] + cfg['model'] + '_' + str(epoch + cfg['num_epoch_offset']) +'.pth'
        torch.save(net.state_dict(), weights_path)

        print('weights_path: ' + str(weights_path))
        print('network weights were saved: ' + weights_path)

    train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs=cfg['num_epochs'])

    return

if __name__ == '__main__':
    args = get_parser().parse_args()
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    train(cfg)
