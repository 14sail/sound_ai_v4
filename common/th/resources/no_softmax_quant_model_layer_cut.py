import torch;
import torch.nn as nn;
import numpy as np;
import random;

#Reproducibility
seed = 1123;
random.seed(seed);
np.random.seed(seed);
torch.manual_seed(seed);
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed);
torch.backends.cudnn.deterministic = True;
torch.backends.cudnn.benchmark = False;
###########################################
from torch.quantization import QuantStub, DeQuantStub

# fcn_no_of_inputs = 7


class att_Model_q(nn.Module):
    def __init__(self, opt):

        # super(ACDNetQuant, self).__init__()
        super(att_Model_q, self).__init__()
        self.opt = opt
        self.adjust_conv = None
        self.linear_initialized = False
        # self.fcn = 1056
        # self.ch_config = [1, self.opt.ch_confing_10 //2, self.opt.ch_confing_10, 1, 
        #                   self.opt.ch_confing_10, self.opt.ch_confing_10 // 2, self.opt.ch_confing_10 // 4, 
        #                   self.fcn, 6]
        self.ch_config = self.opt.ch_conf
        print("self.ch_config", self.ch_config)
        stride1 = 2
        stride2 = 2
        k_size = (3, 3)
        n_frames = (self.opt.sr / 1000) * 10
        sfeb_pool_size = int(n_frames / (stride1 * stride2))
        
        conv1, bn1 = self.make_layers(self.ch_config[0], self.ch_config[1], (1, 9), (1, stride1))
        conv2, bn2 = self.make_layers(self.ch_config[1], self.ch_config[2], (1, 5), (1, stride2))
        
        self.fcn = self.ch_config[8]
        fcn = nn.Linear(self.fcn, self.opt.ch_n_class);
        nn.init.kaiming_normal_(fcn.weight, nonlinearity='sigmoid')
        
        self.sfeb = nn.Sequential(
            conv1, bn1, nn.ReLU(),
            conv2, bn2, nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, sfeb_pool_size))
        )
        
        # Add self-attention after SFEB
        # self.attention = SelfAttention(in_dim=self.ch_config[2])
        # conv3, bn3 = self.make_layers(1, self.ch_config[4], k_size, padding=1)
        conv8, bn8 = self.make_layers(in_channels=1, out_channels=self.ch_config[4], kernel_size=(3, 3), padding=1)
        # conv11, bn11 = self.make_layers(in_channels=self.opt.ch_confing_10, out_channels=self.opt.ch_confing_10 // 4, kernel_size=(3, 3), padding=1)
        conv9, bn9 = self.make_layers(in_channels=self.ch_config[4], out_channels=self.ch_config[5], kernel_size=(3,3), padding=1)

        conv10, bn10 = self.make_layers(in_channels=self.ch_config[5], out_channels=self.ch_config[6], kernel_size=(3,3), padding=1)
        conv11, bn11 = self.make_layers(in_channels=self.ch_config[6], out_channels=self.ch_config[7], kernel_size=(3,3), padding=1)

        conv12, bn12 = self.make_layers(in_channels=self.ch_config[7], out_channels=self.ch_config[8], kernel_size=(1,1), padding=1)

        self.tfeb_modules = [
            # conv3, bn3, nn.ReLU(), nn.MaxPool2d(kernel_size=(2,2)),
            conv8, bn8, nn.ReLU(), nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3), padding=0),
            conv9, bn9, nn.ReLU(), # nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3), padding=0),
            conv10, bn10, nn.ReLU(), nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3), padding=0),
            conv11, bn11, nn.ReLU(),  #nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3), padding=0),
            conv12, bn12, nn.ReLU(),  
            nn.AvgPool2d(kernel_size=(5,7)),
            nn.Flatten(), fcn
        ]
        self.tfeb = nn.Sequential(*self.tfeb_modules)
        self.output = nn.Sequential(nn.Softmax(dim=1))
        # print('self.tfeb',self.tfeb)
        self.quant = QuantStub();
        self.dequant = DeQuantStub();

        # self.quant = torch.ao.quantization.QuantStub();
        # self.dequant = torch.ao.quantization.DeQuantStub()
        
    def forward(self, x):
        #Quantize input
        x = self.quant(x);
        x = self.sfeb(x);
        #swapaxes
        x = x.permute((0, 2, 1, 3));
        x = self.tfeb(x);
        #DeQuantize features before feeding to softmax
        x = self.dequant(x);
        # y = self.output[0](x);
        
        return x;



    def make_layers(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=0, bias=False):
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias);
        nn.init.kaiming_normal_(conv.weight, nonlinearity='relu'); # kaiming with relu is equivalent to he_normal in keras
        bn = nn.BatchNorm2d(out_channels);
        return conv, bn;

    def get_tfeb_pool_sizes(self, con2_ch, width):
        h = self.get_tfeb_pool_size_component(con2_ch);
        w = self.get_tfeb_pool_size_component(width);
        # print(w);
        pool_size = [];
        for  (h1, w1) in zip(h, w):
            pool_size.append((h1, w1));
        return pool_size;

    def get_tfeb_pool_size_component(self, length):
        # print(length);
        c = [];
        index = 1;
        while index <= 6:
            if length >= 2:
                if index == 6:
                    c.append(length);
                else:
                    c.append(2);
                    length = length // 2;
            else:
               c.append(1);

            index += 1;

        return c;
    # def forward(self, x):
    #     # x = x.float()
    #     #Quantize input
    #     x = self.quant(x);
    #     x = self.sfeb(x);
    #     #swapaxes
    #     x = x.permute((0, 2, 1, 3));
    #     x = self.tfeb(x);
    #     #DeQuantize features before feeding to softmax
    #     x = self.dequant(x);
    #     y = self.output[0](x);
    #     return y;


    # def make_layers(self, in_channels, out_channels, kernel_size, stride=(1,1), padding=0, bias=False):
    #     conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias);
    #     nn.init.kaiming_normal_(conv.weight, nonlinearity='relu'); # kaiming with relu is equivalent to he_normal in keras
    #     bn = nn.BatchNorm2d(out_channels);
    #     return conv, bn;

    # def get_tfeb_pool_sizes(self, con2_ch, width):
    #     h = self.get_tfeb_pool_size_component(con2_ch);
    #     w = self.get_tfeb_pool_size_component(width);
    #     # print(w);
    #     pool_size = [];
    #     for  (h1, w1) in zip(h, w):
    #         pool_size.append((h1, w1));
    #     return pool_size;

    # def get_tfeb_pool_size_component(self, length):
    #     # print(length);
    #     c = [];
    #     index = 1;
    #     while index <= 6:
    #         if length >= 2:
    #             if index == 6:
    #                 c.append(length);
    #             else:
    #                 c.append(2);
    #                 length = length // 2;
    #         else:
    #            c.append(1);

    #         index += 1;

    #     return c;

def GetACDNetQuantModel(self):
    net = att_Model_q(self);
    return net;
