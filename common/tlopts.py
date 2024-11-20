import os
import argparse

def parse():
    parser = argparse.ArgumentParser(description='Transfer Learning for ACDNet');

    # General settings
    parser.add_argument('--netType', default='TLACDNet',  required=False);
    parser.add_argument('--data', default='{}/datasets/processed/'.format(os.getcwd()),  required=False);
    parser.add_argument('--dataset', required=False, default='uec_iot', choices=['esc10','esc50', 'frog']);
    parser.add_argument('--BC', default=True, action='store_true', help='BC learning');
    parser.add_argument('--strongAugment', default=True,  action='store_true', help='Add scale and gain augmentation');

    opt = parser.parse_args();

    # #Leqarning settingscc
    # opt.batchSize = 64;
    # opt.weightDecay = 5e-4;
    # opt.momentum = 0.09;
    # opt.nEpochs = 10;#2000;
    # opt.LR = 0.01#0.1;
    # opt.schedule = [0.03, 0.06, 0.09]#[0.3, 0.6, 0.9];
    # opt.warmup = 10;

    # #Basic Net Settings
    # opt.nClasses = 6#50;
    # opt.nFolds = 1;#5;
    # opt.split = [i for i in range(1, opt.nFolds + 1)];
    # opt.sr = 16000#20000;
    # opt.inputLength = 30225;

    # #Test data
    # opt.nCrops = 10;

    return opt


def display_info(opt):
    print('+------------------------------+');
    print('| {} Sound classification'.format(opt.netType));
    print('+------------------------------+');
    print('| dataset  : {}'.format(opt.dataset));
    print('| nEpochs  : {}'.format(opt.nEpochs));
    print('| LRInit   : {}'.format(opt.lr));
    print('| batchSize: {}'.format(opt.batchSize));
    print('| Momentum   : {}'.format(opt.momentum));
    print('| weightDecay: {}'.format(opt.weightDecay));
    print('| schedule : {}'.format(opt.schedule));
    print('| warmup   : {}'.format(opt.warmup));
    print('| nFolds: {}'.format(opt.nFolds));
    print('| Splits: {}'.format(opt.splits));
    print('+------------------------------+');
