import math;
import random;
import os 
import sys
import time
import torch;
import torch.optim as optim;
import numpy as np

import utils_aud as U;

import th.resources.no_softmax_quant_model_layer_cut as models;
import th.resources.calculator as calc;
import torch.nn.functional as F

from trainer import TLTrainer 

from tinynn.converter import TFLiteConverter

mask8 = 0x4000 # >> 8 : 16384
mask7 = 0x2000 # >> 7 :  8192
mask6 = 0x1000 # >> 6 :  4096
mask5 = 0x0800 # >> 5 :  2048
mask4 = 0x0400 # >> 4 :  1024
mask3 = 0x0200 # >> 3 :   512
mask2 = 0x0100 # >> 2 :   256
mask1 = 0x0080 # >> 1 :   128
mask0 = 0x0040 # >> 0 :    64 below the value, drop the value

def maskOP(x):
    x = np.int16(x)
    # print(f"begin:x:{x}")
    if (mask8&x):
        return x >> 8
    elif (mask7&x):
        return x >> 7
    elif (mask6&x):
        return x >> 6
    elif (mask5&x):
        return x >> 5
    elif (mask4&x):
        return x >> 4
    elif (mask3&x):
        return x >> 3
    elif (mask2&x):
        return x >> 2
    elif (mask1&x):
        return x >> 1
    elif (mask0&x):
        return x
    else:
        return 0;
    


# def quantize_int8(x):
#     len_of_x = len(x[0][0][0])
#     # print(f"len_of_x:{len_of_x}")
#     for i in range(len_of_x):
#         nflag = 2; #positive
#         # print("{}:{}".format(i,x[0][0][0][i]))
#         tmp_x = x[0][0][0][i]
#         if tmp_x < 0:
#             tmp_x = np.abs(tmp_x)
#             nflag = 1
#         tmp_x = maskOP(tmp_x)
#         if(nflag==1):
#             tmp_x = -1 * (tmp_x)
#         # print("{}:{}".format(i,x[0][0][0][i]))
#         # print("*********************************")
#         x[0][0][0][i] = tmp_x
#     return x
def quantize_int8(x, axis):
    len_of_x = len(x[0][0][0])
    # print(f"len_of_x:{len_of_x}")
    for i in range(len_of_x):
        nflag = 2; #positive
        # print("{}:{}".format(i,x[0][0][0][i]))
        tmp_x = x[0][0][0][i]
        if tmp_x < 0:
            tmp_x = np.abs(tmp_x)
            nflag = 1
        tmp_x = maskOP(tmp_x)
        if(nflag==1):
            tmp_x = -1 * (tmp_x)
        # print("{}:{}".format(i,x[0][0][0][i]))
        # print("*********************************")
        x[0][0][0][i] = tmp_x
    return x

def display_info(opt):
    print('+------------------------------+');
    print('| {} Sound classification'.format(opt.netType));
    print('+------------------------------+');
    print('| dataset  : {}'.format(opt.dataset));
    print('| nEpochs  : {}'.format(opt.nEpochs));
    print('| LRInit   : {}'.format(opt.lr));
    print('| schedule : {}'.format(opt.schedule));
    print('| warmup   : {}'.format(opt.warmup));
    print('| batchSize: {}'.format(opt.batchSize));
    print('| nFolds: {}'.format(opt.nFolds));
    print('| Splits: {}'.format(opt.splits));
    print('| Device: {}'.format(opt.device));
    print('| Model Path: {}'.format(opt.model_path));
    print('| Model Name: {}'.format(opt.model_name));
    print('+------------------------------+');


class QATTrainer:
    def __init__(self, opt=None, split=0):
        self.opt = opt;

        # self.opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
        self.opt.device = torch.device("cpu")
        self.trainGen = TLTrainer.getTrainGen_choose(self)
        self.qunt_nClass = opt.nClasses;
        self.bestAcc = 0.0;
        self.bestAccEpoch = 0;

        self.opt.valX = None;
        self.opt.valY = None;
        self.trainX = None;
        self.trainY = None;

        self.trainer = TLTrainer(opt)  
        self.trainer.load_choose_val_data()
        self.valX = self.opt.valX;
        self.valY = self.opt.valY;       

    def __compute_accuracy(self, y_pred, y_target):
        print(y_pred.shape);
        with torch.no_grad():
            y_pred = (y_pred.reshape(y_pred.shape[0]//self.opt.nCrops, self.opt.nCrops, y_pred.shape[1])).mean(dim=1);
            y_target = (y_target.reshape(y_target.shape[0]//self.opt.nCrops, self.opt.nCrops, y_target.shape[1])).mean(dim=1);

            y_pred = y_pred.argmax(dim=1);
            y_target = y_target.argmax(dim=1);

            acc = (((y_pred==y_target)*1).float().mean()*100).item();
        return acc;

    def __compute_accuracy(self, y_pred, y_target, lossFunc):
        # print("===========================")
        # print(f"shape of y_pred: {y_pred.shape}")
        # print(f"shape of y_target: {y_target.shape}")
        
        with torch.no_grad():
            if y_pred.ndim > 1:
                y_pred = F.log_softmax(y_pred, dim=-1)  # Apply log_softmax for KLDivLoss

            if y_pred.shape != y_target.shape:
                raise ValueError(f"Mismatched shapes y_pred: {y_pred.shape}, y_target: {y_target.shape}")

            # For accuracy, extract predicted classes
            y_pred_classes = y_pred.argmax(dim=1)
            acc = (((y_pred_classes == y_target.argmax(dim=1)) * 1).float().mean() * 100).item()

            # Ensure `y_target` is in a compatible shape or processing state
            loss = lossFunc(y_pred, y_target).item()  # Use y_pred already log_softmaxed

        return acc, loss
            

    def __load_model(self, quant=True):
        state = torch.load(self.opt.model_path, map_location=self.opt.device, weights_only=True);
        self.opt.ch_conf = state['config']
        print(self.opt.ch_conf);
        net = None;
        net = models.GetACDNetQuantModel(self.opt).to(self.opt.device);
        calc.summary(net, (1,1,self.opt.inputLength));
        # net.load_state_dict(state['weight']);
        net.load_state_dict(state['weight'], strict=False);

        return net;

    # def __load_train_data(self):
    #     print('Preparing calibration dataset..');
    #     data = np.load(self.opt.Data_npz_path, allow_pickle=True);
    #     print(f"device is :{self.opt.device}")
    #     print(f"len of Y:{len(data['labels_train'])}")
    #     dataX = data['sounds_train'].reshape(data['sounds_train'].shape[0],1,1,data['sounds_train'].shape[1]) # .astype(np.float32);
    #     self.trainX = torch.tensor(dataX).to(self.opt.device);
    #     self.trainY = torch.tensor(data['labels_train']).to(self.opt.device);          # .type(torch.float32)
    #     print('Calibration dataset is ready');

    # def __load_choose_train_data(self):
    #     data = np.load(self.opt.Data_npz_path, allow_pickle=True)
    #     print(f"device is: {self.opt.device}")

    #     sounds_train, labels_train = data['sounds_train'], data['labels_train']

    #     selected_indices = [i for i, label in enumerate(labels_train) if label in self.opt.choose_class]
        
    #     filtered_sounds_train = sounds_train[selected_indices]
    #     filtered_labels_train = labels_train[selected_indices]

    #     filtered_sounds_train = filtered_sounds_train.reshape(filtered_sounds_train.shape[0],1,1,filtered_sounds_train.shape[1]) 
    #     self.trainX = torch.tensor(filtered_sounds_train).to(self.opt.device);
    #     self.trainY = torch.tensor(filtered_labels_train).to(self.opt.device);   
    #     # self.valX = torch.tensor(dataX).to(self.opt.device)



    def __train(self, net):
        # self.__load_choose_train_data();
        # net.eval();
        # calc.summary(net, (1,1,self.opt.inputLength));
        tr_loss_list, tr_acc_list, val_loss_list, val_acc_list = [],[],[],[]

        lossFunc = torch.nn.KLDivLoss(reduction='batchmean');
        optimizer = optim.SGD(net.parameters(), lr=self.opt.lr, weight_decay=self.opt.weightDecay, momentum=self.opt.momentum, nesterov=True);
        train_start_time = time.time();
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
                x = torch.tensor(np.moveaxis(x, 3, 1)).to(self.opt.device);
                y = torch.tensor(y).to(self.opt.device);
                # print("!!!!!", x.shape, y.shape)
                # print(x[0])
                # zero the parameter gradients
                optimizer.zero_grad();

                # forward + backward + optimize
                try:
                    outputs = torch.softmax(input=net(x),dim=1); #need to check float NaN value?
                    # outputs_log_prob = F.log_softmax(outputs, dim=1)
                    running_acc += (((outputs.data.argmax(dim=1) == y.argmax(dim=1))*1).float().mean()).item();

                    eps = 1e-6  # a small epsilon value
                    loss = lossFunc(torch.log(torch.clamp(outputs, min=eps)), y)


                    # loss = lossFunc(outputs.log(), y);
                    loss.backward();
                    optimizer.step();
                    running_loss += loss.item();
                except ValueError as e:
                    print(f"error label:train{e}")# {y}
                    print(f"error data:train") # {x}
                    continue

            tr_acc = (running_acc / n_batches)*100;
            tr_loss = running_loss / n_batches;

            #Epoch wise validation Validation
            epoch_train_time = time.time() - epoch_start_time;

            net.eval();
            val_acc, val_loss = self.__validate(net, lossFunc);
            #Save best model
            self.__chk_bestAcc(val_acc, epochIdx, net);
            self.__on_epoch_end(epoch_start_time, epoch_train_time, epochIdx, cur_lr, tr_loss, tr_acc, val_loss, val_acc);

            val_loss_list.append(val_loss); val_acc_list.append(val_acc);
            tr_loss_list.append(tr_loss); tr_acc_list.append(tr_acc);

            if (epochIdx + 1) % 50 == 0:
                self.trainer.plot_confusion_matrix(net, epochIdx)
                self.trainer.plot_loss_acc_epoch(tr_loss_list, tr_acc_list, val_loss_list, val_acc_list, epochIdx)            
            running_loss = 0;
            running_acc = 0;
            net.train();

        total_time_taken = time.time() - train_start_time;
        print("Execution finished in: {}".format(U.to_hms(total_time_taken)));


    def __chk_bestAcc(self, acc, epochIdx, net):
        print(f"current best Acc is {self.bestAcc}")
        print(f"pass in acc is {acc}")
        if acc > self.bestAcc:
            self.bestAcc = acc
            self.bestAccEpoch = epochIdx + 1
            print(f"model saved....., acc: {acc}")

            # Save the best model's state_dict
            state_dict_path = f"{self.opt.save_pt_model_path}/best_model_state_dict.pt"
            torch.save(net.state_dict(), state_dict_path)
            print(f"State dict saved at {state_dict_path}")

            # Attempt to trace and save the model with TorchScript
            try:
                # First, try to remove or handle hooks to make the model scriptable
                for name, module in net.named_modules():
                    if hasattr(module, '_observer_forward_hook'):
                        print(f"Removing hook from {name}")
                        module._observer_forward_hook = None

                # Use torch.jit.trace if scripting is problematic due to hooks
                # Assuming you have an example input to trace the model
                example_input = torch.randn(1, 1, 1, self.opt.inputLength).to(self.opt.device)
                traced_net = torch.jit.trace(net, example_input)
                scripted_model_path = f"{self.opt.save_pt_model_path}/{self.opt.model_name}_traced.pt"
                torch.jit.save(traced_net, scripted_model_path)
                print(f"Traced model saved at {scripted_model_path}")


            except Exception as e:
                print(f"Failed to script or trace the model: {e}")

            # Save the model state along with its config for later use
            net_path = f"{self.opt.save_pt_model_path}/uncompressed_qat_models/{self.opt.model_name}_best_model_@{epochIdx+1}.pt"
            torch.save({'weight': net.state_dict(), 'config': net.ch_config}, net_path)
            print(f"Full model saved at {net_path}")


            # with torch.no_grad():
            #     print("====================saving====================")
            #     # dummy_input = torch.randn(1, 1, 30225, 1); wrong: RuntimeError: quantized::conv2d (qnnpack): each dimension of output tensor should be greater than 0.
            #     # dummy_input = torch.FloatTensor(quantize_int8(torch.randn(1, 1, 1, inp_len).numpy(),3)); #correct,workable
            #     dummy_input = torch.FloatTensor(quantize_int8(torch.randn(1, 1, 1, self.opt.inputLength).numpy()))
            #     # dummy_input = torch.randn(1, 1, 1, inp_len).numpy()
            #     # min_val = dummy_input.min()
            #     # max_val = dummy_input.max()
            #     # quantized_input = quantize_float32_2_int8(dummy_input, min_val, max_val)
            #     # print("check!!!", type(dummy_input))
            #     converter = TFLiteConverter(net,
            #                                 dummy_input,
            #                                 # torch.tensor(quantized_input, dtype=torch.int8).float(),
            #                                 quantize_input_output_type='int8',#設定此欄，輸入會強制為int8
            #                                 fuse_quant_dequant=True,
            #                                 quantize_target_type='int8',
            #                                 hybrid_conv=False,
            #                                 float16_quantization=True,
            #                                 optimize=5,
            #                                 tflite_path=f"{self.opt.save_tflite_model_path}/{self.opt.model_name}@{epochIdx+1}_bestacc_{self.bestAcc}.tflite")
            #     converter.convert()
            #     print("====================saving====================")
                        
            
    def __on_epoch_end(self, start_time, train_time, epochIdx, lr, tr_loss, tr_acc, val_loss, val_acc):
        epoch_time = time.time() - start_time;
        val_time = epoch_time - train_time;
        line = 'SP-{} Epoch: {}/{} | Time: {} (Train {}  Val {}) | Train: lr {}  Loss {:.2f}  Acc {:.2f}% | Val: Loss {:.2f}  Acc(top1) {:.2f}% | HA {:.2f}@{}\n'.format(
            self.opt.splits, epochIdx+1, self.opt.nEpochs, U.to_hms(epoch_time), U.to_hms(train_time), U.to_hms(val_time),
            lr, tr_loss, tr_acc, val_loss, val_acc, self.bestAcc, self.bestAccEpoch);
        # print(line)
        sys.stdout.write(line);
        sys.stdout.flush();
        
    
    def __get_lr(self, epoch):
        divide_epoch = np.array([self.opt.nEpochs * i for i in self.opt.schedule]);
        decay = sum(epoch > divide_epoch);
        if epoch <= self.opt.warmup:
            decay = 1;
        return self.opt.lr * np.power(0.1, decay);

    def __get_batch(self, index):
        x = self.trainX[index*self.opt.batchSize : (index+1)*self.opt.batchSize];
        y = self.trainY[index*self.opt.batchSize : (index+1)*self.opt.batchSize];
        return x.to(self.opt.device), y.to(self.opt.device);
    

    def __validate(self, net, lossFunc):
        if self.opt.valX is None:
            self.trainer.load_choose_val_data();
        net.eval();
        acc=0.0; 
        loss = 0.0;
        with torch.no_grad():
            y_pred = None;
            batch_size = len(self.opt.valX);#(self.opt.batchSize//self.opt.nCrops)*self.opt.nCrops;
            x = self.opt.valX[:];
            try:
                scores = net(x);
                y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data));
                acc, loss = self.__compute_accuracy(y_pred, self.opt.valY, lossFunc);
            except ValueError:
                print(f"error data:val")
                # print(f"error data:{x}")
        net.train();
        return acc, loss;

    # def __calibrate(self, net):
        # self.__load_choose_train_data();
        
        # net.eval();
        # with torch.no_grad():
        #     for i in range(1,2):
        #         x_pred = None;
        #         for idx in range(math.ceil(len(self.trainX)/self.opt.batchSize)):
        #             x = self.trainX[idx*self.opt.batchSize : (idx+1)*self.opt.batchSize];
        #             y = self.trainY[idx*self.opt.batchSize : (idx+1)*self.opt.batchSize];
        #             #print(x.shape);
        #             # exit();
        #             scores = net(x);
        #             x_pred = scores.data if x_pred is None else torch.cat((x_pred, scores.data));
                
        #         y = self.trainY
                
        #         x_pred = x_pred.argmax(dim=1);

        #         x_target = y #.argmax(dim=1);

        #         acc = (((x_pred==x_target)*1).float().mean()*100).item();
        #         print('calibrate accuracy is: {:.2f}'.format(acc));
        # return acc;
    

    def QuantizeModel(self):
        net = self.__load_model(True);
        # net = self.__load_model(False);
        config = net.ch_config;
        net.eval();
        
        #Fuse modules to
        torch.quantization.fuse_modules(net.sfeb, ['0','1','2'], inplace=True);
        torch.quantization.fuse_modules(net.sfeb, ['3','4','5'], inplace=True);

        torch.quantization.fuse_modules(net.tfeb, ['0','1','2'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['4','5','6'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['7','8','9'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['11','12','13'], inplace=True);
        torch.quantization.fuse_modules(net.tfeb, ['14','15','16'], inplace=True);

        # torch.quantization.fuse_modules(net.tfeb, ['18','19','20'], inplace=True);
        # torch.quantization.fuse_modules(net.tfeb, ['21','22','23'], inplace=True);
        # torch.quantization.fuse_modules(net.tfeb, ['25','26','27'], inplace=True);
        # torch.quantization.fuse_modules(net.tfeb, ['28','29','30'], inplace=True);
        # torch.quantization.fuse_modules(net.tfeb, ['33','34','35'], inplace=True);

        net.train();
        net.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        torch.backends.quantized.engine = 'qnnpack';
        print(f"net.qconfig : {net.qconfig}");
        torch.quantization.prepare_qat(net, inplace=True);
        
        # Calibrate with the training data
        # self.__calibrate(net);
        self.__train(net);

        #place trained model to cpu
        # net.to('cpu');
        # Convert to quantized model
        torch.quantization.convert(net, inplace=True);
        print('Post Training Quantization: Convert done');

        print("Size of model after quantization");
        torch.save(net.state_dict(), "temp.p")
        print('Size (MB):', os.path.getsize("temp.p")/1e6)
        os.remove('temp.p')

        self.trainer.load_choose_val_data();
        val_acc = self.__validate_test(net, True, self.opt.valX, self.opt.valY);
        print('Testing: Acc(top1) {:.2f}%'.format(val_acc));
        net.to('cpu');
        # torch.jit.save(torch.jit.script(net), '{}/th/quantized_models/{}.pt'.format(os.getcwd(), self.opt.model_name.format()));

        torch.jit.save(torch.jit.script(net), f"{self.opt.save_pt_model_path}/{self.opt.model_name}.pt".format());
        torch.save({'weight':net.state_dict(), 'config':net.ch_config}, f"{self.opt.save_pt_model_path}/uncompressed_qat_models/uncompress_{self.opt.model_name}.pt");
        
        # **************convert to tflite**********  save_tflite_model_s5
        with torch.no_grad():
            # dummy_input = torch.randn(1, 1, 30225, 1); wrong: RuntimeError: quantized::conv2d (qnnpack): each dimension of output tensor should be greater than 0.
            # dummy_input = torch.FloatTensor(quantize_int8(torch.randn(1, 1, 1, inp_len).numpy(),3)); #correct,workable
            # dummy_input = self.valX[:] #torch.FloatTensor(quantize_int8(torch.randn(1, 1, 1, self.opt.inputLength).numpy()))
            # dummy_input = torch.randn(1, 1, 1, inp_len).numpy()
            # min_val = dummy_input.min()
            # max_val = dummy_input.max()
            # quantized_input = quantize_float32_2_int8(dummy_input, min_val, max_val)
            # print("check!!!", type(dummy_input))
            self.opt.tflite_path = f"{self.opt.save_tflite_model_path}/{self.opt.model_name}.tflite"
            # converter = TFLiteConverter(net,
            #                             dummy_input,
            #                             # torch.tensor(quantized_input, dtype=torch.int8).float(),
            #                             # quantize_input_output_type='int8',#設定此欄，輸入會強制為int8
            #                             fuse_quant_dequant=True,
            #                             quantize_target_type='int8',
            #                             hybrid_conv=False,
            #                             float16_quantization=True,
            #                             optimize=5,
            #                             tflite_path=self.opt.tflite_path)
            # converter.convert()
            # print("====================saving====================")
            
            dummy_input = torch.FloatTensor(quantize_int8(torch.randn(1, 1, 1, self.opt.inputLength).numpy(),3)); #correct,workable
            
            converter = TFLiteConverter(net,
                                        dummy_input,
                                        quantize_input_output_type='int8',#設定此欄，輸入會強制為int8
                                        fuse_quant_dequant=True,
                                        quantize_target_type='int8',
                                        hybrid_conv=False,
                                        float16_quantization=True,
                                        optimize=5,tflite_path=self.opt.tflite_path)
            converter.convert()
        
    def TestModel(self, quant=False):
        if quant:
            print(f"the model name:{self.opt.model_name}");
            net = torch.jit.load(f"{self.opt.save_pt_model_path}/{self.opt.model_name}.pt")
        else:
            print("has not quanted, load unquanted model...");
            net = self.__load_model();
            # calc.summary(net, (1,1,self.opt.inputLength));
        self.trainer.load_choose_val_data()
        net.eval();
        val_acc = self.__validate_test(net, False, self.opt.valX, self.opt.valY);
        print('Testing: Acc(top1) {:.2f}%'.format(val_acc));

    def GetModelSize(self):
        orig_net_path = self.opt.model_path;
        print('Full precision model size (KB):', os.path.getsize(orig_net_path)/(1024));
        save_onnx_name = f"{self.opt.save_pt_model_path}/{self.opt.model_name}.onnx";
        quant_net_path = f"{self.opt.ave_pt_model_path}/has_qat_models/onnx_models/"+save_onnx_name;
        print('Quantized model size (KB):', os.path.getsize(quant_net_path)/(1024))

    def __validate_test(self, net, qat_done, valX, valY):
        net.eval();
        # if qat_done:
        #     valX.to('cpu');
        #     valY.to('cpu');
        # else:
        #     valX.to('cuda:0');
        #     valY.to('cuda:0');
            
        with torch.no_grad():
            y_pred = None;
            batch_size = len(self.valX);
            x = self.valX[:];
            scores = net(x);
            y_pred = scores.data if y_pred is None else torch.cat((y_pred, scores.data));
            acc = self.__compute_accuracy_2(y_pred, self.valY);
        return acc;  
      
    def __compute_accuracy_2(self, y_pred, y_target):
        print(y_pred.shape);
        with torch.no_grad():
            y_pred = (y_pred.reshape(y_pred.shape[0]//self.opt.nCrops, self.opt.nCrops, y_pred.shape[1])).mean(dim=1);
            y_target = (y_target.reshape(y_target.shape[0]//self.opt.nCrops, self.opt.nCrops, y_target.shape[1])).mean(dim=1);

            y_pred = y_pred.argmax(dim=1);
            y_target = y_target.argmax(dim=1);

            acc = (((y_pred==y_target)*1).float().mean()*100).item();
        return acc;