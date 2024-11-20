import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query_conv(x.view(batch_size, C, -1)).permute(0, 2, 1)
        proj_key = self.key_conv(x.view(batch_size, C, -1))
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x.view(batch_size, C, -1))

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        out = self.gamma * out + x
        return out

class att_Model(nn.Module):
    def __init__(self, opt=None):
        super(att_Model, self).__init__()
        self.opt = opt
        print("self.opt.ch_confing_10",self.opt.ch_confing_10)
        self.adjust_conv = None
        self.linear_initialized = False
        self.fcn = 16
        self.ch_config = [1, self.opt.ch_confing_10 //2, self.opt.ch_confing_10, 1, 
                          self.opt.ch_confing_10*2, self.opt.ch_confing_10,
                          self.opt.ch_confing_10*2, self.opt.ch_confing_10, self.opt.ch_confing_10 // 2,
                          self.fcn, 6]
        print("self.ch_config", self.ch_config)
        
        stride1 = 2
        stride2 = 2
        k_size = (3, 3)
        n_frames = (self.opt.sr / 1000) * 10
        sfeb_pool_size = int(n_frames / (stride1 * stride2))
        
        conv1, bn1 = self.make_layers(1, self.ch_config[1], (1, 9), (1, stride1))
        conv2, bn2 = self.make_layers(self.ch_config[1], self.ch_config[2], (1, 5), (1, stride2))
        self.sfeb = nn.Sequential(
            conv1, bn1, nn.ReLU(),
            conv2, bn2, nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, sfeb_pool_size))
        )

        # Add self-attention after SFEB
        # self.attention = SelfAttention(in_dim=self.ch_config[2])

        conv8, bn8 = self.make_layers(in_channels=1, out_channels=self.ch_config[4], kernel_size=(3, 3), padding=1)
        # conv11, bn11 = self.make_layers(in_channels=self.opt.ch_confing_10, out_channels=self.opt.ch_confing_10 // 4, kernel_size=(3, 3), padding=1)
        conv9, bn9 = self.make_layers(in_channels=self.ch_config[4], out_channels=self.ch_config[5], kernel_size=(3,3), padding=1)

        conv10, bn10 = self.make_layers(in_channels=self.ch_config[5], out_channels=self.ch_config[6], kernel_size=(3,3), padding=1)
        conv11, bn11 = self.make_layers(in_channels=self.ch_config[6], out_channels=self.ch_config[7], kernel_size=(3,3), padding=1)

        conv12, bn12 = self.make_layers(in_channels=self.ch_config[7], out_channels=self.ch_config[8], kernel_size=(1,1), padding=1)
        # conv12, bn12 = self.make_layers(in_channels=self.opt.ch_confing_10 // 4, out_channels=2, kernel_size=(1, 1), padding=1)
        

        self.tfeb_modules = [
            # conv3, bn3, nn.ReLU(), nn.MaxPool2d(kernel_size=(2,2)),
            conv8, bn8, nn.ReLU(), nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3), padding=0),
            conv9, bn9, nn.ReLU(), # nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3), padding=0),
            conv10, bn10, nn.ReLU(), nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3), padding=0),
            conv11, bn11, nn.ReLU(),  #nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3), padding=0),
            conv12, bn12, nn.ReLU(), 
            nn.AvgPool2d(kernel_size=(5, 7)),
            nn.Flatten(), nn.Linear(self.fcn, self.opt.ch_n_class)
        ]
        self.tfeb = nn.Sequential(*self.tfeb_modules)
        self.output = nn.Sequential(nn.Softmax(dim=1))



    def forward(self, x):
        x = self.sfeb(x)
        # x = self.attention(x)  # Apply attention
        x = x.permute((0, 2, 1, 3))
        x = self.tfeb(x)
        y = self.output(x)
        return y

    def make_layers(self,in_channels, out_channels, kernel_size, stride=(1,1), padding=0, bias=None):
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias);
        nn.init.kaiming_normal_(conv.weight, nonlinearity='relu'); # kaiming with relu is equivalent to he_normal in keras
        bn = nn.BatchNorm2d(out_channels);
        return conv, bn;
