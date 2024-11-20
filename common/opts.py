import os
import argparse

def parse():
    parser = argparse.ArgumentParser(description='ACDNet Sound Classification');

    # General settings
    parser.add_argument('--netType', default='TLACDNet',  required=False);
    parser.add_argument('--data', default='../datasets/processed/',  required=False);
    parser.add_argument('--dataset', required=False, default='fold1_dataset.npz', choices=['10']);
    parser.add_argument('--BC', default=True, action='store_true', help='BC learning');
    parser.add_argument('--strongAugment', default=True,  action='store_true', help='Add scale and gain augmentation');
    #在ipynb中，不能使用parser.parse，要改用parser.parse_known_args()
    opt, unknown = parser.parse_known_args()
    #Leqarning settings
    opt.batchSize = 5;#64;
    opt.weightDecay = 5e-4;
    opt.momentum = 0.09;
    opt.nEpochs = 10;#2000;
    opt.LR = 0.01#0.1;
    opt.schedule = [0.03, 0.06, 0.09]#[0.3, 0.6, 0.9];
    opt.warmup = 10;

    #Basic Net Settings
    opt.nClasses = 6#50;
    opt.nFolds = 1;#5;
    opt.splits = 1#[i for i in range(1, opt.nFolds + 1)];
    opt.sr = 16000#20000;
    opt.inputLength = 30225;
    #Test data
    opt.nCrops = 5;

    return opt


def display_info(opt):
    print('+------------------------------+');
    print('| {} Sound classification'.format(opt.netType));
    print('+------------------------------+');
    print('| dataset  : {}'.format(opt.dataset));
    print('| nEpochs  : {}'.format(opt.nEpochs));
    print('| LRInit   : {}'.format(opt.LR));
    print('| schedule : {}'.format(opt.schedule));
    print('| warmup   : {}'.format(opt.warmup));
    print('| batchSize: {}'.format(opt.batchSize));
    print('| Splits: {}'.format(opt.splits));
    print('+------------------------------+');
