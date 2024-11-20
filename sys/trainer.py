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
from gen_trainData import TLGenerator


logObj = None;

class TLTrainer:
    global logObj;
    def __init__(self, opt):
        self.opt = opt;
        self.valX = None;
        self.valY = None;
        self.bestAcc = 0.0; 
        self.bestAcc_tr = 0.0; 
        self.bestAcc_all = self.bestAcc+self.bestAcc_tr; 
        self.bestAccEpoch = 0;
        self.bestAccEpoch_tr= 0;
        self.opt.best_save_sum = 0.0
        self.sum_acc = 0.0
        # self.trainGen = getTrainGen(opt,classes_dict=classes_dict) 
        self.trainGen = self.getTrainGen_choose()

    def getTrainGen_choose(self):
        dataset = np.load(self.opt.Data_npz_path, allow_pickle=True);

        selected_indices = [i for i, label in enumerate(dataset['labels_train']) if label in self.opt.choose_class]

        train_sounds = dataset['sounds_train'][selected_indices]
        labels_train = dataset['labels_train'][selected_indices]
        
        if self.opt.choose_or_not:
            unique_sorted = sorted(set(labels_train))
            mapping = {original: consecutive for consecutive, original in enumerate(unique_sorted)}
            
            labels_train = [mapping[num] for num in labels_train]


        trainGen = TLGenerator(train_sounds, labels_train, self.opt, preprocess_type=None);
        return trainGen


    def Train(self):
        train_start_time = time.time();
        tr_loss_list, tr_acc_list, val_loss_list, val_acc_list = [],[],[],[]

        net = att_Model(self.opt).to(self.opt.device)#models.GetACDNetModel().to(self.opt.device);
        #print networks parameters' require_grade value
        for k_, v_ in net.named_parameters():
            print(f"{k_}:{v_.requires_grad}")
        print('ACDNet model has been prepared for training');

        calc.summary(net, (1,1,self.opt.inputLength));

        # training_text = "Re-Training" if self.opt.retrain else "Training from Scratch";
        # print("{} has been started. You will see update after finishing every training epoch and validation".format(training_text));

        lossFunc = torch.nn.KLDivLoss(reduction='batchmean');
        optimizer = optim.SGD(net.parameters(), lr=self.opt.lr, weight_decay=self.opt.weightDecay, momentum=self.opt.momentum, nesterov=True);

        # self.opt.nEpochs = 1957 if self.opt.split == 4 else 2000;
        for epochIdx in range(self.opt.nEpochs):
            epoch_start_time = time.time();
            optimizer.param_groups[0]['lr'] = self.__get_lr__(epochIdx+1);
            cur_lr = optimizer.param_groups[0]['lr'];
            running_loss = 0.0;
            running_acc = 0.0;
            n_batches = math.ceil(len(self.trainGen.data)/self.opt.batch_size);
            for batchIdx in range(n_batches):
                # with torch.no_grad():
                x,y = self.trainGen.getitem(batchIdx)
                x = torch.tensor(np.moveaxis(x, 3, 1)).to(self.opt.device);



                y = torch.tensor(y).to(self.opt.device);

                # # zero the parameter gradients    # Rick版本
                # optimizer.zero_grad();

                # # forward + backward + optimize
                # outputs = net(x);
                # running_acc += (((outputs.data.argmax(dim=1) == y.argmax(dim=1))*1).float().mean()).item();
                # loss = lossFunc(outputs.log(), y);
                # loss.backward();
                # optimizer.step();

                # running_loss += loss.item();

                # 
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(x)

                # Convert outputs to log probabilities
                outputs_log_prob = F.log_softmax(outputs, dim=1)
                
                # Ensure outputs_log_prob shape is consistent with y
                if outputs_log_prob.shape[1] != y.shape[1]:
                    raise ValueError("Output shape and target shape mismatch")

                running_acc += (((outputs_log_prob.argmax(dim=1) == y.argmax(dim=1)) * 1).float().mean()).item()
                loss = lossFunc(outputs_log_prob, y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                #

            tr_acc = (running_acc / n_batches)*100;
            tr_loss = running_loss / n_batches;

            #Epoch wise validation Validation
            epoch_train_time = time.time() - epoch_start_time;

            net.eval();
            val_acc, val_loss = self.validate(net, lossFunc);
            #Save best model
            # self.save_model(val_acc, epochIdx, net);
            self.save_model_refined(val_acc, tr_acc, epochIdx, net);
            self.on_epoch_end(epoch_start_time, epoch_train_time, epochIdx, cur_lr, tr_loss, tr_acc, val_loss, val_acc);

            val_loss_list.append(val_loss); val_acc_list.append(val_acc);
            tr_loss_list.append(tr_loss); tr_acc_list.append(tr_acc);            

            if (epochIdx + 1) % 50 == 0:
                # self.save_model_refined(val_acc, tr_acc, epochIdx, net);
                self.plot_confusion_matrix(net, epochIdx)
                self.plot_loss_acc_epoch(tr_loss_list, tr_acc_list, val_loss_list, val_acc_list, epochIdx)


            running_loss = 0;
            running_acc = 0;
            net.train();

        total_time_taken = time.time() - train_start_time;
        print("Execution finished in: {}".format(U.to_hms(total_time_taken)));
        self.save_model_refined(val_acc, tr_acc, epochIdx, net);
        print(self.opt.best_save_sum)
        return net
    

    def one_hot_encode(self, data):
        order=self.opt.choose_class
        index_map = {number: index for index, number in enumerate(order)}
        one_hot_list = []

        for num in data:
            one_hot = [0] * len(order)
            if num in index_map:
                one_hot[index_map[num]] = 1
            one_hot_list.append(one_hot)

        return one_hot_list

    def plot_confusion_matrix(self, net, epochIdx):
        # Ensure validation data is loaded
        if self.opt.valX is None:
            self.load_choose_val_data()
            # self.load_val_data()
        
        net.eval()
        with torch.no_grad():
            y_pred = net(self.opt.valX).argmax(dim=1).cpu().numpy()
            y_true = self.opt.valY.argmax(dim=1).cpu().numpy()
        
        cm = confusion_matrix(y_true, y_pred, labels=range(self.opt.ch_n_class))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(self.opt.ch_n_class))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix at Epoch {epochIdx + 1}")
        plt.savefig(f'{self.opt.logSaveDir}/Confusion_at_epoch{epochIdx + 1}.png', bbox_inches='tight')
        plt.show()
    def plot_confusion_matrix_cpu(self, net, epochIdx):
        # Ensure validation data is loaded
        if self.opt.valX is None:
            self.load_choose_val_data()
            # self.load_val_data()
        
        net.eval()
        with torch.no_grad():
            # net_ = net.to('cpu')
            y_pred = net(self.opt.valX.to('cpu')).argmax(dim=1).cpu().numpy()
            y_true = self.opt.valY.argmax(dim=1).cpu().numpy()
        
        cm = confusion_matrix(y_true, y_pred, labels=range(self.opt.ch_n_class))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(self.opt.ch_n_class))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix at Epoch {epochIdx + 1}")
        plt.savefig(f'{self.opt.logSaveDir}/Confusion_at_epoch{epochIdx + 1}.png', bbox_inches='tight')
        plt.show()

    def load_choose_val_data(self):
        data = np.load(self.opt.Data_npz_path, allow_pickle=True)
        print(f"device is: {self.opt.device}")

        sounds_val, labels_val = data['sounds_val'], data['labels_val']

        selected_indices = [i for i, label in enumerate(labels_val) if label in self.opt.choose_class]
        
        filtered_sounds_val = sounds_val[selected_indices]
        filtered_labels_val = labels_val[selected_indices]

        dataX = filtered_sounds_val.reshape(filtered_sounds_val.shape[0], 1, 1, filtered_sounds_val.shape[1]).astype(np.float32)
        # self.valX = torch.tensor(dataX).to(self.opt.device)
        self.valY = torch.tensor(self.one_hot_encode(filtered_labels_val)).type(torch.float32).to(self.opt.device)
        # print(dataX.max(),dataX.max())
        sounds = []
        for sound in dataX:
            sound = TLGenerator.preprocess(self,sound)
            sounds.append(sound)
            # print(f"---VAL---{sound.max()},{sound.min()}")
        self.valX = torch.tensor(np.asarray(sounds)).to(self.opt.device)
        self.opt.valX = self.valX
        self.opt.valY = self.valY
        # print("valX",self.opt.valX.shape,"valY",self.opt.valY.shape)


    # def load_val_data(self): # *if you don't want to choose the class
    #     data = np.load(self.opt.Data_npz_path, allow_pickle=True);
    #     # print("load validation data")
    #     print(f"device is :{self.opt.device}")
    #     print(f"len of Y:{len(data['labels_val'])}")
    #     dataX = data['sounds_val'].reshape(data['sounds_val'].shape[0],1,1,data['sounds_val'].shape[1]).astype(np.float32);
    #     self.valX = torch.tensor(dataX).to(self.opt.device);
    #     self.valY = torch.tensor(self.one_hot_encode(data['labels_val'])).type(torch.float32).to(self.opt.device);
    #     # print(self.valX.max(),self.valX.max())
        
    def __get_lr__(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule]);
        decay = sum(epoch > divide_epoch);
        if epoch <= self.opt.warmup:
            decay = 1;
        return self.opt.lr * np.power(0.1, decay);

    def get_batch(self, index):
        x = self.trainX[index*self.opt.batch_size : (index+1)*self.opt.batch_size];
        y = self.trainY[index*self.opt.batch_size : (index+1)*self.opt.batch_size];
        return x.to(self.opt.device), y.to(self.opt.device);

    def validate(self, net, lossFunc):
        if self.valX is None:
            self.load_choose_val_data()
        
        net.eval()
        with torch.no_grad():
            y_pred = None
            batch_size = len(self.valX)

            x = self.valX[:]
            scores = net(x)
            scores_log_prob = F.log_softmax(scores, dim=1)  # Ensure scores are in log-probability format
            
            # Ensure valY is a probability distribution
            valY_prob = self.valY.float() / torch.sum(self.valY, dim=1, keepdim=True)

            acc, loss = self.compute_accuracy(scores_log_prob, valY_prob, lossFunc)
        
        net.train()

        return acc, loss    


    #Calculating average prediction (10 crops) and final accuracy
    def compute_accuracy(self, y_pred, y_target, lossFunc):

        with torch.no_grad():
            #Reshape to shape theme like each sample comtains 10 samples, calculate mean and find theindices that has highest average value for each sample
            # print("y_pred",y_pred)
            # print("y_target",y_target)
            if self.opt.nCrops == 1:
                y_pred = y_pred.argmax(dim=1);
                y_target = y_target.argmax(dim=1);
            else:
                y_pred = (y_pred.reshape(y_pred.shape[0]//self.opt.nCrops, self.opt.nCrops, y_pred.shape[1])).mean(dim=1).argmax(dim=1);
                y_target = (y_target.reshape(y_target.shape[0]//self.opt.nCrops, self.opt.nCrops, y_target.shape[1])).mean(dim=1).argmax(dim=1);
                print(f"after: len of y_pred:{len(y_pred)}, len of y_target:{len(y_target)}")

            acc = (((y_pred==y_target)*1).float().mean()*100).item();
            # valLossFunc = torch.nn.KLDivLoss();
            loss = lossFunc(y_pred.float().log(), y_target.float()).item();
            # loss = 0.0;
        return acc, loss;


    def on_epoch_end(self, start_time, train_time, epochIdx, lr, tr_loss, tr_acc, val_loss, val_acc):
        self.sum_acc = val_acc+tr_acc
        epoch_time = time.time() - start_time;
        val_time = epoch_time - train_time;
        # line = 'SP-{} Epoch: {}/{} | Time: {} (Train {}  Val {}) | Train: LR {}  Loss {:.2f}  Acc {:.2f}% | Val: Loss {:.2f}  Acc(top1) {:.2f}% | HA {:.2f}@{}\n'.format(
        #     self.opt.splits, epochIdx+1, self.opt.nEpochs, U.to_hms(epoch_time), U.to_hms(train_time), U.to_hms(val_time),
        #     lr, tr_loss, tr_acc, val_loss, val_acc, self.bestAcc, self.bestAccEpoch);
        line = (
            f"SP-{self.opt.splits} Epoch: {epochIdx + 1}/{self.opt.nEpochs} | Time: {U.to_hms(epoch_time)} "
            f"(Train {U.to_hms(train_time)}  Val {U.to_hms(val_time)}) | Train: lr {lr}  Loss {tr_loss:.4f}  "
            f"Acc {tr_acc:.2f}% | HA {self.bestAcc_tr:.2f}@{self.bestAccEpoch_tr} | Val: Loss {val_loss:.4f}  Acc {val_acc:.2f}% | "
            f"HA {self.bestAcc:.2f}@{self.bestAccEpoch} | Sum {self.sum_acc:.2f} HA {self.bestAcc_tr+self.bestAcc:.2f}\n"
        )

        sys.stdout.write(line);
        sys.stdout.flush();
        self.opt.logObj.write(line);
        self.opt.logObj.write("\n");
        self.opt.logObj.flush();


    def save_model_refined(self, acc, train_acc, epochIdx, net):  # self.bestAcc_tr
        self.sum_acc = acc+train_acc
        if epochIdx == self.opt.nEpochs:
            print("train finished")
            self.do_save_model(acc, train_acc, self.bestAccEpoch_tr, net);

        if train_acc >= self.bestAcc_tr and train_acc > self.opt.first_save_acc and epochIdx>self.opt.least_save_epoch:
            self.bestAcc_tr =train_acc
            self.bestAccEpoch_tr = epochIdx +1;
            self.do_save_model(acc, train_acc, self.bestAccEpoch_tr, net);
        

        elif acc >= self.bestAcc and acc > self.opt.first_save_acc and epochIdx>self.opt.least_save_epoch:
            self.bestAcc = acc;
            self.bestAccEpoch = epochIdx +1;
            self.do_save_model(acc, train_acc, self.bestAccEpoch, net);

        else:
            if acc >= self.opt.save_val_acc and epochIdx>self.opt.least_save_epoch:
                self.do_save_model(acc, train_acc, epochIdx, net);
            elif train_acc >= self.opt.save_train_acc and epochIdx>self.opt.least_save_epoch:
                self.do_save_model(acc, train_acc, epochIdx, net);                
            else:
                pass

    def do_save_model(self, acc, tr_acc, epochIdx, net):
        self.opt.for_save_sum = self.bestAcc+acc+self.bestAcc_tr+tr_acc

        save_model_name_ = self.opt.model_name.format(self.bestAcc, acc,self.bestAcc_tr , tr_acc, epochIdx, self.opt.for_save_sum);
        save_model_fullpath = self.opt.modelSaveDir + save_model_name_;
        print(f"save model to {save_model_fullpath}")
        torch.save({'weight':net.state_dict(), 'config':net.ch_config}, save_model_fullpath);
        self.opt.logObj.write(f"save model:{self.opt.model_name}, bestAcc:{self.bestAcc}, ValAcc:{acc}-TrAcc{tr_acc}-@{epochIdx}");
        self.opt.logObj.write("\n");
        self.opt.logObj.flush();
        if self.opt.for_save_sum>=self.opt.best_save_sum:
            self.opt.best_save_sum = self.opt.for_save_sum
            self.opt.best_save_name =  save_model_fullpath  

    def plot_loss_acc_epoch(self, tr_loss_list, tr_acc_list, val_loss_list, val_acc_list, epochIdex):
        val_loss_list = np.nan_to_num(val_loss_list, nan=0)
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1); plt.plot(tr_loss_list, label='train loss', color='blue');  plt.plot(val_loss_list, label='val loss', color='orange')
        plt.title(f'Loss Over Epochs {epochIdex+1}')
        plt.xlabel('Epoch'); plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2); plt.plot(tr_acc_list, label='train accuracy', color='blue'); plt.plot(val_acc_list, label='val accuracy', color='orange')
        plt.title(f'Accuracy Over Epochs {epochIdex+1}')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout(); 
        plt.savefig(f'{self.opt.logSaveDir}/loss_and_accuracy_at_epoch{epochIdex + 1}.png', bbox_inches='tight')
        plt.show()