import train_policy
import racer
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import time
import os

from dataset_loader import DaggerDrivingDataset, DrivingDataset
from driving_policy import DiscreteDrivingPolicy
from utils import DEVICE, str2bool


def get_env_args():
    ns = argparse.Namespace()
    ns.out_dir = './dataset/train'
    ns.save_expert_actions = True
    ns.expert_drives = False
    ns.timesteps = 2000
    ns.learner_weights = ''
    ns.n_steering_classes = 20
    return ns

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/dagger_learner_0.weights", default='./weights/dagger_learner_0.weights')
    parser.add_argument("--dagger_iterations", help="", default=10)
    args = parser.parse_args()

    
    
    #####
    ## Enter your DAgger code here
    ## Reuse functions in racer.py and train_policy.py
    ## Save the learner weights of the i-th DAgger iteration in ./weights/learner_i.weights where 
    #####
    data_transform = transforms.Compose([ transforms.ToPILImage(),
                                          transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                          transforms.RandomRotation(degrees=80),
                                          transforms.ToTensor()])

    datas = []
    env_args = get_env_args()
    for i in range(1, args.dagger_iterations+1):
        env_args.run_id = i
        driving_policy = train_policy.main(args)
        args.start_time = time.time()
        racer.run(driving_policy, env_args)

        
        # training_iterator = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

        # train_policy.train_discrete(driving_policy, training_iterator, opt, args)
    
    #print ('TRAINING LEARNER ON INITIAL DATASET')


    
    #print ('GETTING EXPERT DEMONSTRATIONS')
        
    #print ('RETRAINING LEARNER ON AGGREGATED DATASET')
    
