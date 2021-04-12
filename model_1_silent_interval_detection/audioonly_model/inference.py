import argparse
import json
import os
import librosa

import numpy as np 
import torch 
from networks import get_network
from tools import *
from utils import ensure_dir, get_parent_dir
from common import (EXPERIMENT_DIR, PHASE_PREDICTION, PHASE_TESTING,
                    PHASE_TRAINING, get_config)

from dataset import (DATA_REQUIRED_SR, DATA_MAX_AUDIO_SAMPLES)

SIGMOID_THRESHOLD = 0.5
EXPERIMENT_PREDICTION_OUTPUT_DIR = os.path.join(EXPERIMENT_DIR, 'outputs')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir','-a',type=str,default='',required=True, help="Audio directory to process")
    parser.add_argument('--ckpt', type=str, default='latest', required=False, help="desired checkpoint to restore")
    parser.add_argument('-o', '--outputs', default=EXPERIMENT_PREDICTION_OUTPUT_DIR, type=str, help="outputs dir to write results")
    args = parser.parse_args()

    net = get_network()

    if args.ckpt == 'latest' or args.ckpt == 'best_acc':
        name = args.ckpt    
    else:
        name = "ckpt_epoch{}".format(args.ckpt)

    load_path = os.path.join(EXPERIMENT_DIR,"{}.pth".format(name))

    print('Load saved model: {}'.format(load_path))

    net.load_state_dict(torch.load(load_path)['model_state_dict'])

    if torch.cuda.device_count() > 1:
        print('For multi-GPU')
        net = nn.DataParallel(net.cuda())   # For multi-GPU
    elif torch.cuda.device_count() == 1:
        print('For single-GPU')
        net = net.cuda()    # For single-GPU
    else:
        net = net
    # Set model to evaluation mode
    net.eval()




    for file in os.listdir(args.audio_dir):
        if file.endswith('.wav'):
            filepath = os.path.join(args.audio_dir, file)
            print("file_path:",filepath)
            
            


