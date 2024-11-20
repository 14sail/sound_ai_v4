import torch
import torch.nn as nn

import torch
from torch.nn import Module
from torch import nn, einsum, Tensor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import time;
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim;

import utils_aud as U
import calculator as calc;

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from model import att_Model
from trainer import TLTrainer 

from th.resources.pruning_tools import filter_pruning, filter_pruner;


class PruningTrainer:
    # global logObj
    def __init__(self, opt):
        self.opt = opt;
        self.opt.channels_to_prune_per_iteration = 1;
        self.opt.finetune_epoch_per_iteration = 2;
        self.opt.prune_targets = [(1,1)]
        self.opt.prune_type = 2 #determine the prunning algo, 1: Magnitude Pruning ;2: tylor-pruning
        self.opt.device = 'cuda:0'
        self.pruner = None;
        self.iterations = 0;
        self.cur_acc = 0.0;
        self.cur_iter = 1;
        self.cur_lr = self.opt.lr;
        self.net = None;
        self.criterion = torch.nn.KLDivLoss(reduction='batchmean');
        self.trainer = TLTrainer(opt)  
        self.trainGen = TLTrainer.getTrainGen_choose(self)

        self.opt.valX = None;
        self.opt.valY = None;
        self.trainer.load_choose_val_data()
    

    def PruneAndTrain(self):
        self.net = att_Model(self.opt);
        trained_model = self.opt.base_model_path
        self.net.load_state_dict(torch.load(trained_model, map_location=self.opt.device)['weight'] ,strict=False);
        print(f" --- chonfig{torch.load(trained_model, map_location=self.opt.device)['config']}")
        # print("self.net",self.net)
        self.net = self.net.to('cuda:0');#at home use apple m2
        # self.net = self.net.to(self.opt.device);
        self.pruner = filter_pruning.Magnitude(self.net, self.opt) if self.opt.prune_type == 1 else filter_pruning.Taylor(self.net, self.opt);
        print(f"pruning algorithm is {self.pruner}");
        self.__validate();
        # calc.summary(self.net, (1, 1, self.opt.inputLength), brief=False); # shape of one sample for inferenceing
        # exit();
        #Make sure all the layers are trainable
        for param in self.net.parameters():
            param.requires_grad = True
        self.iterations = self.__estimate_pruning_iterations();
        # exit();
        for i in range(1, self.iterations):
            self.cur_iter = i;
            iter_start = time.time();
            print("\nIteration {} of {} starts..".format(i, self.iterations-1), flush=True);
            print("Ranking channels.. ", flush=True);
            prune_targets = self.__get_candidates_to_prune(self.opt.channels_to_prune_per_iteration);
            self.net = filter_pruner.prune_layers(self.net, prune_targets , self.opt.prune_all, self.opt.device);
            
            
            # calc.summary(self.net, (1, 1, self.opt.inputLength), brief=True);
            self.__validate();
            print("Fine tuning {} epochs to recover from prunning iteration.".format(self.opt.finetune_epoch_per_iteration), flush=True);
            print(f"config{self.__get_channel_list()}")
            print("self.net",self.net)
            if self.cur_iter in list(map(int, np.array(self.iterations)*self.opt.schedule)):
                self.cur_lr *= 0.1;
            optimizer = optim.SGD(self.net.parameters(), lr=self.cur_lr, momentum=0.9);

            if (i + 1) % 25 == 0:
                self.trainer.plot_confusion_matrix(self.net, i)
                
            self.__train(optimizer, epoches = self.opt.finetune_epoch_per_iteration);
            print("Iteration {}/{} finished in {}".format(self.cur_iter, self.iterations+1, U.to_hms(time.time()-iter_start)), flush=True);
            print("Total channels prunned so far: {}".format(i*self.opt.channels_to_prune_per_iteration), flush=True);

            self.__save_model(self.net)
                
        calc.summary(self.net, (1, 1, self.opt.inputLength)); # shape of one sample for inferenceing
        print("save the final model!")
        print("self.net",self.net)
        self.__save_model(self.net);
    
    def __get_candidates_to_prune(self, num_filters_to_prune):
        self.pruner.reset();
        if self.opt.prune_type == 1:
            self.pruner.compute_filter_magnitude();
        else:
            self.__train_epoch(rank_filters = True);
            self.pruner.normalize_ranks_per_layer();

        return self.pruner.get_prunning_plan(num_filters_to_prune);

    def __estimate_pruning_iterations(self):
        # get total number of variables from all conv2d featuremaps
        prunable_count = sum(self.__get_channel_list(self.opt.prune_all));
        total_count= sum(self.__get_channel_list());
        #iterations_reqired = int((prunable_count * self.opt.prune_ratio) / self.opt.channels_to_prune_per_iteration);
        #prune_ratio works with the total number of channels, not only with the prunable channels. i.e. 80% or total will be pruned from total or from only features
        iterations_reqired = int((total_count * self.opt.prune_ratio) / self.opt.channels_to_prune_per_iteration);
        print('Total Channels: {}, Prunable: {}, Non-Prunable: {}'.format(total_count, prunable_count, total_count - prunable_count), flush=True);
        print('No. of Channels to prune per iteration: {}'.format(self.opt.channels_to_prune_per_iteration), flush=True);
        print('Total Channels to prune ({}%): {}'.format(int(self.opt.prune_ratio*100), int(total_count * self.opt.prune_ratio)-1), flush=True);
        print('Total iterations required: {}'.format(iterations_reqired-1), flush=True);
        return iterations_reqired;

    def __get_channel_list(self, prune_all=True):
        ch_conf = [];
        # if prune_all:
        for name, module in enumerate(self.net.sfeb):
            if issubclass(type(module), torch.nn.Conv2d):
                ch_conf.append(module.out_channels);

        for name, module in enumerate(self.net.tfeb):
            if issubclass(type(module), torch.nn.Conv2d):
                ch_conf.append(module.out_channels);

        return ch_conf;

    def __train(self, optimizer = None, epoches=10):
        for i in range(epoches):
            # print("Epoch: ", i);
            self.__train_epoch(optimizer);
            self.__validate();
        print("Finished fine tuning.", flush=True);

    def __train_batch(self, optimizer, batch, label, rank_filters):
        self.net.zero_grad()
        if rank_filters:
            output = self.pruner.forward(batch);
            
            if self.opt.device == "cuda":
                label = label.cpu() #use apple m2, in office use cuda
                output = output.cpu() #use apple m2, in office use cuda
            self.criterion(output.log(), label).backward();
        else:
            self.criterion(self.net(batch), label).backward();
            optimizer.step();

    def __train_epoch(self, optimizer = None, rank_filters = False):
        if rank_filters is False and optimizer is None:
            print('Please provide optimizer to train_epoch', flush=True);
            exit();
        n_batches = math.ceil(len(self.trainGen.data)/self.opt.batchSize);
        for b_idx in range(n_batches):
            x,y = self.trainGen.getitem(b_idx)
            # dataX = x.reshape(x.shape[0],1,1,x.shape[1]).astype(np.float32);
            # x = torch.tensor(dataX).to(self.opt.device);
            x = torch.tensor(np.moveaxis(x, 3, 1)).to(self.opt.device);
            y = torch.tensor(y).to(self.opt.device);
            self.__train_batch(optimizer, x, y, rank_filters);

    def __validate(self):
        
        self.net.eval();
        with torch.no_grad():
            y_pred = None;
            batch_size = (self.opt.batchSize//self.opt.nCrops)*self.opt.nCrops;
            for idx in range(math.ceil(len(self.opt.valX)/batch_size)):
                x = self.opt.valX[idx*batch_size : (idx+1)*batch_size];

                x = x.type(torch.cuda.FloatTensor);
                # print("In val ")
                # calc.summary(self.net, (1, 1, self.opt.inputLength), brief=False); # shape of one sample for inferenceing

                scores = self.net(x);
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data));

            acc, loss = self.__compute_accuracy(y_pred, self.opt.valY);
        print('Current Testing Performance - Val: Loss {:.3f}  Acc(top1) {:.3f}%'.format(loss, acc), flush=True);
        self.cur_acc = acc;
        self.net.train();
        return acc, loss;

    def __save_model(self, net):
        net.ch_config = self.__get_channel_list();
        dir = os.getcwd();
        fname = self.opt.model_name;
        if os.path.isfile(fname):
            os.remove(fname);
        torch.save({'weight':net.state_dict(), 'config':net.ch_config}, fname);
        print(f" --- chonfig{net.ch_config}")
        print(f" --- save model at {self.opt.model_name} --- ")

    def __compute_accuracy(self, y_pred, y_target):
        with torch.no_grad():
            #Reshape to shape theme like each sample comtains 10 samples, calculate mean and find the indices that has highest average value for each sample
            y_pred = (y_pred.reshape(y_pred.shape[0]//self.opt.nCrops, self.opt.nCrops, y_pred.shape[1])).mean(dim=1).argmax(dim=1);
            y_target = (y_target.reshape(y_target.shape[0]//self.opt.nCrops, self.opt.nCrops, y_target.shape[1])).mean(dim=1).argmax(dim=1);
            # if self.opt.device == "mps":
            #     y_target = y_target.cpu() #use apple m2, in office use cuda
            acc = (((y_pred==y_target)*1).float().mean()*100).item();
            # valLossFunc = torch.nn.KLDivLoss();
            loss = self.criterion(y_pred.float().log(), y_target.float()).item();
            # loss = 0.0;
        return acc, loss;