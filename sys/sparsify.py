import torch
import torch.nn as nn

import torch
from torch.nn import Module
from torch import nn, einsum, Tensor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import time;
import math
import sys
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

import th.resources.pruning_tools.weight_pruning as weight_pruner;

class PruningTrainer:
    global logObj
    def __init__(self, opt):
        self.opt = opt;
        self.opt.prune_algo = 'l0norm';
        self.opt.prune_interval = 1;
        self.bestAcc = 0.0; 
        self.bestAcc_tr = 0.0; 
        self.bestAcc_all = self.bestAcc+self.bestAcc_tr; 
        self.bestAccEpoch = 0;
        self.bestAccEpoch_tr= 0;
        self.opt.best_save_sum = 0.0
        self.sum_acc = 0.0
        # self.trainGen = TLGenerator(opt)#train_generator.setup(self.opt, self.opt.split);
        self.trainer = TLTrainer(opt)  
        self.trainGen = TLTrainer.getTrainGen_choose(self)

        if torch.cuda.is_available():
            self.opt.device = 'cuda:0'
        else:
            self.opt.device = 'cpu'
        print(f"!!! In PruningTrainer:: current used device:{self.opt.device}")

        self.start_time = time.time();

    def PruneAndTrain(self):
        self.trainGen;
        self.trainer;
        print(self.opt.device);
        tr_loss_list, tr_acc_list, val_loss_list, val_acc_list = [],[],[],[]
        loss_func = torch.nn.KLDivLoss(reduction='batchmean');

        #Load saved model dict net.load_state_dict(torch.load(base_model_path, map_location=self.opt.device)['weight'], strict=False)
        net = att_Model(self.opt).to(self.opt.device)#GetACDNetModel()
        
        # base_model_path = use_model_in_step1 #！！！！！！！！！！！！！！！！！！！！！

        net.load_state_dict(torch.load(self.opt.base_model_path, map_location=self.opt.device)['weight'] ,strict=False);
        calc.summary(net, (1,1,self.opt.inputLength))
        net.eval();
        val_acc, val_loss = self.trainer.validate(net, loss_func);
        print('Testing - Val: Loss {:.3f}  Acc(top1) {:.3f}%'.format(val_loss, val_acc));
        net.train();

        optimizer = optim.SGD(net.parameters(), lr=self.opt.lr, weight_decay=self.opt.weightDecay, momentum=self.opt.momentum, nesterov=True)

        weight_name = ["weight"]# if not self.opt.factorize else ["weightA", "weightB", "weightC"]
        layers_n = weight_pruner.layers_n(net, param_name=["weight"])[1];
        all_num = sum(layers_n.values());
        print("\t TOTAL PRUNABLE PARAMS: {}".format(all_num));
        print("\t PRUNE RATIO :{}".format(self.opt.prune_ratio));
        sparse_factor = int(all_num * (1-self.opt.prune_ratio));
        print("\t SPARSE FACTOR: {}".format(sparse_factor));
        model_size = (sparse_factor * 4)/1024**2;
        print("\t MODEL SIZE: {:.2f} MB".format(model_size));
        prune_algo = getattr(weight_pruner, self.opt.prune_algo);
        prune_func = lambda m: prune_algo(m, sparse_factor, param_name=weight_name);

        for epoch_idx in range(self.opt.nEpochs):
            epoch_start_time = time.time();
            optimizer.param_groups[0]['lr'] = self.__get_lr__(epoch_idx+1);
            cur_lr = optimizer.param_groups[0]['lr'];
            running_loss = 0.0;
            running_acc = 0.0;
            n_batches = math.ceil(len(self.trainGen.data)/self.opt.batchSize);
            net.train();
           
            for batch_idx in range(n_batches):
                x, y = self.trainGen.getitem(batch_idx)
                x = torch.tensor(np.moveaxis(x, 3, 1)).to(self.opt.device)
                y = torch.tensor(y).to(self.opt.device)

                # Ensure x and y are moved to the correct device directly as Float
                x = x.to(torch.float32)
                y = y.to(torch.float32)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = net(x)
                res_y = y.argmax(dim=1).type(torch.FloatTensor).to(self.opt.device)

                running_acc += ((( outputs.data.argmax(dim=1) == res_y)*1).float().mean()).item();

                # Ensure consistency of device and types
                # Calculate loss, making sure both inputs are on the same device
                outputs_log = F.log_softmax(outputs, dim=1)  # Outputs remain on the same device
                loss = loss_func(outputs_log, y)  # Both `outputs_log` and `y` should be on the same device

                # Backward pass
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                with torch.no_grad():
                    prune_func(net)

            prune_func(net)

            tr_acc = (running_acc / n_batches)*100;
            tr_loss = running_loss / n_batches;

            #Epoch wise validation Validation
            epoch_train_time = time.time() - epoch_start_time;
            net.eval();
            val_acc, val_loss = self.trainer.validate(net, loss_func);
            
            #Save best model
            self.__save_model(val_acc, tr_acc, epoch_idx, net);

            self.trainer.on_epoch_end(epoch_start_time, epoch_train_time, epoch_idx, cur_lr, tr_loss, tr_acc, val_loss, val_acc);

            val_loss_list.append(val_loss); val_acc_list.append(val_acc);
            tr_loss_list.append(tr_loss); tr_acc_list.append(tr_acc);
            if (epoch_idx + 1) % 50 == 0:
                self.trainer.plot_confusion_matrix(net, epoch_idx)
                self.trainer.plot_loss_acc_epoch(tr_loss_list, tr_acc_list, val_loss_list, val_acc_list, epoch_idx)

            
            running_loss = 0;
            running_acc = 0;
            net.train();

        total_time_taken = time.time() - self.start_time;
        print("Execution finished in: {}".format(U.to_hms(total_time_taken)));
        self.__save_model(val_acc, tr_acc, epoch_idx, net);
    
        # return net
    
    def __get_lr__(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule]);
        decay = sum(epoch > divide_epoch);
        if epoch <= self.opt.warmup:
            decay = 1;
        return self.opt.lr * np.power(0.1, decay);

    def __save_model(self, acc, train_acc, epochIdx, net):
        self.sum_acc = acc+train_acc
        if acc > self.bestAcc and acc > self.opt.first_save_acc:
            self.bestAcc = acc;
            self.bestAccEpoch = epochIdx +1;
            self.__do_save_model(acc, train_acc, epochIdx+1, net);
        elif train_acc >= self.bestAcc_tr and train_acc > self.opt.first_save_acc and epochIdx>self.opt.least_save_epoch:
            self.bestAcc_tr =train_acc
            self.bestAccEpoch_tr = epochIdx +1;
            self.__do_save_model(acc, train_acc, epochIdx+1, net);
        
        else:
            if acc > self.opt.save_val_acc and train_acc > self.opt.save_train_acc: 
                self.__do_save_model(acc, train_acc, epochIdx+1, net);
            elif train_acc >= self.opt.save_train_acc and epochIdx>self.opt.least_save_epoch:
                self.__do_save_model(acc, train_acc, epochIdx+1, net);
            else:
                pass

    def __do_save_model(self, acc, tr_acc, epochIdx, net):
        self.opt.for_save_sum = self.bestAcc+acc+self.bestAcc_tr+tr_acc
        save_model_name_ = self.opt.model_name.format(self.bestAcc, acc, tr_acc, epochIdx);
        save_model_fullpath = self.opt.save_dir + save_model_name_;
        print(f"save model to {save_model_fullpath}")
        torch.save({'weight':net.state_dict(), 'config':net.ch_config}, save_model_fullpath);
        self.opt.logObj.write(f"save model:{self.opt.model_name}, bestAcc:{self.bestAcc}, valAcc:{acc}, trainAcc:{tr_acc}, record@{epochIdx}-epoch");
        self.opt.logObj.write("\n");
        self.opt.logObj.flush();
        if self.opt.for_save_sum>=self.opt.best_save_sum:
            self.opt.best_save_sum = self.opt.for_save_sum
            self.opt.best_save_name =  save_model_fullpath  
