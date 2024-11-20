import math;
import numpy as np;
import sys
import time;
import torch;
import torch.optim as optim;

import utils_aud as U;
import torch.nn.functional as F

import th.resources.calculator as calc;

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import th.resources.models_layer_cut as models

from trainer import TLTrainer 

class ReTrainer:
    def __init__(self, opt):
        self.opt = opt;
        self.bestAcc = 0.0; 
        self.bestAcc_tr = 0.0; 
        self.bestAcc_all = self.bestAcc+self.bestAcc_tr; 
        self.bestAccEpoch = 0;
        self.bestAccEpoch_tr= 0;
        self.opt.best_save_sum = 0.0
        self.sum_acc = 0.0
        self.trainer = TLTrainer(opt)  
        self.trainGen = TLTrainer.getTrainGen_choose(self)
        self.trainer.load_choose_val_data()
        self.valX = self.opt.valX;
        self.valY = self.opt.valY;        

    def Train(self):
        train_start_time = time.time();
        state = torch.load(self.opt.second_pruned_model, map_location="cuda")
        weights = state['weight']
        self.opt.config = state['config']
        print(f"config is {self.opt.config}")
        tr_loss_list, tr_acc_list, val_loss_list, val_acc_list = [],[],[],[]

        # net = models.GetACDNetModel(input_len=inp_len, sr=sr, nclass=self.opt.nClasses, channel_config=config)
        net = models.GetACDNetQuantModel_6_16k_32(self.opt)
        net.load_state_dict(weights);
        # net.load_state_dict(weights, strict=False)

        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in weights.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

         #show the acdnet structures
        calc.summary(net,(1,1,self.opt.inputLength))
        # net = getPrunedModel(pruned_model_path=pruned_acdnet)
        #print networks parameters' require_grade value
        for k_, v_ in net.named_parameters():
            print(f"{k_}:{v_.requires_grad}")
        print('ACDNet model has been prepared for training');

        calc.summary(net, (1,1,self.opt.inputLength));
        # net = net.cuda();
        # training_text = "Re-Training" if self.opt.retrain else "Training from Scratch";
        # print("{} has been started. You will see update after finishing every training epoch and validation".format(training_text));

        lossFunc = torch.nn.KLDivLoss(reduction='batchmean');
        optimizer = optim.SGD(net.parameters(), lr=self.opt.lr, weight_decay=self.opt.weightDecay, momentum=self.opt.momentum, nesterov=True);

        # self.opt.nEpochs = 1957 if self.opt.split == 4 else 2000;
        for epochIdx in range(self.opt.nEpochs):
            epoch_start_time = time.time();
            optimizer.param_groups[0]['lr'] = self.__get_lr(epochIdx+1);
            cur_lr = optimizer.param_groups[0]['lr'];
            running_loss = 0.0;
            running_acc = 0.0;
            n_batches = math.ceil(len(self.trainGen.data)/self.opt.batchSize);
            for batchIdx in range(n_batches):
                # with torch.no_grad():
                x,y = self.trainGen.getitem(batchIdx)
                x = torch.tensor(np.moveaxis(x, 3, 1)).to('cpu');
                y = torch.tensor(y).to('cpu');

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
                # zero the parameter gradients
                # optimizer.zero_grad();

                # # forward + backward + optimize
                # # outputs = net(x);#in office and use cpu
                # x = x.type(torch.FloatTensor) #use apple m2
                # outputs = net(x)
                # # in office use cpu, need to change to cuda
                # # running_acc += (((outputs.data.argmax(dim=1) == y.argmax(dim=1))*1).float().mean()).item();
                # # at home use apple m2
                # res_y = y.argmax(dim=1)
                # res_y = res_y.type(torch.FloatTensor)
                # running_acc += ((( outputs.data.argmax(dim=1) == res_y)*1).float().mean()).item();
                # y = y.type(torch.FloatTensor)
                
                # loss = lossFunc(outputs.log(), y);
                # loss.backward();
                # optimizer.step();

                running_loss += loss.item();

            tr_acc = (running_acc / n_batches)*100;
            tr_loss = running_loss / n_batches;

            #Epoch wise validation Validation
            epoch_train_time = time.time() - epoch_start_time;

            net.eval();
            val_acc, val_loss = self.__validate(net, lossFunc);
            #Save best model
            # self.__save_model(val_acc, epochIdx, net);
            # ratio, acc, tr_acc, epochIdx, net
            self.__save_model_refined(self.opt.pruningRatio*100, val_acc, tr_acc, epochIdx, net);
            self.__on_epoch_end(epoch_start_time, epoch_train_time, epochIdx, cur_lr, tr_loss, tr_acc, val_loss, val_acc);

            val_loss_list.append(val_loss); val_acc_list.append(val_acc);
            tr_loss_list.append(tr_loss); tr_acc_list.append(tr_acc);
            if (epochIdx + 1) % 50 == 0:
                self.trainer.plot_confusion_matrix_cpu(net, epochIdx)
                self.trainer.plot_loss_acc_epoch(tr_loss_list, tr_acc_list, val_loss_list, val_acc_list, epochIdx)

            running_loss = 0;
            running_acc = 0;
            net.train();

        total_time_taken = time.time() - train_start_time;
        print("Execution finished in: {}".format(U.to_hms(total_time_taken)));


    def __save_model_refined(self, ratio, acc, tr_acc, epochIdx, net):  # self.bestAcc_tr

        self.sum_acc = acc+tr_acc

        if tr_acc >= self.bestAcc_tr and tr_acc > self.opt.first_save_acc and epochIdx>self.opt.least_save_epoch:
            self.bestAcc_tr =tr_acc
            self.bestAccEpoch_tr = epochIdx +1;
            self.__do_save_model(ratio, acc, tr_acc, epochIdx, net);

        elif acc >= self.bestAcc and acc > self.opt.first_save_acc and epochIdx>self.opt.least_save_epoch:
            self.bestAcc = acc;
            self.bestAccEpoch = epochIdx +1;
            self.__do_save_model(ratio, acc, tr_acc, epochIdx, net);

        else:
            if acc >= self.opt.save_val_acc and epochIdx>self.opt.least_save_epoch:
                self.__do_save_model(ratio, acc, tr_acc, epochIdx, net);

            elif tr_acc >= self.opt.save_train_acc and epochIdx>self.opt.least_save_epoch:
                self.__do_save_model(ratio, acc, tr_acc, epochIdx, net);
           
            else:
                pass

    def __get_lr(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule]);
        decay = sum(epoch > divide_epoch);
        if epoch <= self.opt.warmup:
            decay = 1;
        return self.opt.lr * np.power(0.1, decay);

    def __validate(self, net, lossFunc):
        if self.valX is None:
            self.trainer.load_choose_val_data()
        net.eval();
        with torch.no_grad():
            y_pred = None;
            batch_size = len(self.valX);#(self.opt.batchSize//self.opt.nCrops)*self.opt.nCrops;
#             for idx in range(math.ceil(len(self.valX)/batch_size)):
#             for idx in range(len(self.valX)):
#             x = self.valX[idx*batch_size : (idx+1)*batch_size];
            x = self.valX[:];
            x = torch.tensor(x)
            x = x.type(torch.FloatTensor) # use apple mp2
            scores = net(x);
            y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data));
            acc, loss = self.__compute_accuracy(y_pred, self.valY, lossFunc);
        net.train();
        return acc, loss;

    #Calculating average prediction (10 crops) and final accuracy
    def __compute_accuracy(self, y_pred, y_target, lossFunc):
        print(f"shape of y_pred:{y_pred.shape}");
        print(f"shape of y_target:{y_target.shape}");
        
        with torch.no_grad():
            #Reshape to shape theme like each sample comtains 10 samples, calculate mean and find theindices that has highest average value for each sample
            if self.opt.nCrops == 1:
                y_pred = y_pred.argmax(dim=1);
                y_target = y_target.argmax(dim=1);
            else:
                y_pred = (y_pred.reshape(y_pred.shape[0]//self.opt.nCrops, self.opt.nCrops, y_pred.shape[1])).mean(dim=1).argmax(dim=1);
                y_target = (y_target.reshape(y_target.shape[0]//self.opt.nCrops, self.opt.nCrops, y_target.shape[1])).mean(dim=1).argmax(dim=1);
                print(f"after: len of y_pred:{len(y_pred)}, len of y_target:{len(y_target)}")
            y_target = y_target.cpu() #use apple m2, in office use cuda
            acc = (((y_pred==y_target)*1).float().mean()*100).item();
            # valLossFunc = torch.nn.KLDivLoss();
            loss = lossFunc(y_pred.float().log(), y_target.float()).item();
            # loss = 0.0;
        return acc, loss;

    def __on_epoch_end(self, start_time, train_time, epochIdx, lr, tr_loss, tr_acc, val_loss, val_acc):
        epoch_time = time.time() - start_time;
        val_time = epoch_time - train_time;
        line = 'SP-{} Epoch: {}/{} | Time: {} (Train {}  Val {}) | Train: LR {}  Loss {:.2f}  Acc {:.2f}% | Val: Loss {:.2f}  Acc(top1) {:.2f}%  HA {:.2f}| best sum {:.2f}@{}\n'.format(
            self.opt.splits, epochIdx+1, self.opt.nEpochs, U.to_hms(epoch_time), U.to_hms(train_time), U.to_hms(val_time),
            lr, tr_loss, tr_acc, val_loss, val_acc, self.bestAcc, self.opt.best_save_sum, self.bestAccEpoch);
        # print(line)
        sys.stdout.write(line);
        sys.stdout.flush();



    def __do_save_model(self, ratio, acc, tr_acc, epochIdx, net):
        self.opt.for_save_sum = self.bestAcc+acc+self.bestAcc_tr+tr_acc
        save_model_name_ = self.opt.model_name.format(ratio, acc, tr_acc, epochIdx+1);
        save_model_fullpath = self.opt.saveDir + save_model_name_;
        print(f"save model to {save_model_fullpath}")
        torch.save({'weight':net.state_dict(), 'config':net.ch_config}, save_model_fullpath);
        if self.opt.for_save_sum>=self.opt.best_save_sum:
            self.opt.best_save_sum = self.opt.for_save_sum
            self.opt.best_save_name =  save_model_fullpath  
        # logObj.write(f"save model:{self.opt.model_name}, bestAcc:{self.bestAcc}, valAcc:{acc}, trainAcc:{tr_acc}, record@{epochIdx}-epoch");
        # logObj.write("\n");
        # logObj.flush();
