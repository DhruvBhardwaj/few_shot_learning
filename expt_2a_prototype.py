
import torch
import torchvision
import torchvision.transforms as transforms
import os
import sys
import json
import time
import copy
import datetime
from torchmeta.datasets import Omniglot
from torchmeta.transforms import Categorical, ClassSplitter, Rotation
from torchvision.transforms import Compose, Resize, ToTensor
from torchmeta.utils.data import BatchMetaDataLoader

import utils as util

from torch.utils.data import random_split
from config_prototype import cfg
from models import protoClassifier
#################################################
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/expt_2a')
#tensorboard --logdir ./runs/ --load_fast=false --reload_multifile=true --reload_mtifile_inactive_secs=-1
#################################################
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

sys.stdout = util.Logger(cfg['training']['save_path'],'expt_2a_prototype.txt')
#################################################
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)
print()
#################################################
dataset = Omniglot("data",                   
                   num_classes_per_task=cfg['training']['N-way'],                   
                   transform=Compose([Resize(28), ToTensor()]),                   
                   target_transform=Categorical(num_classes=5),
                   class_augmentations=[Rotation([90, 180, 270])],
                   meta_train=True,
                   download=True)
dataset = ClassSplitter(dataset, shuffle=True, num_train_per_class=cfg['training']['K-shot'], num_test_per_class=cfg['training']['K-shot'])
dataloader = BatchMetaDataLoader(dataset, batch_size=cfg['training']['batch_size'], num_workers=4)

testdataset = Omniglot("data",                   
                   num_classes_per_task=cfg['training']['N-way'],                   
                   transform=Compose([Resize(28), ToTensor()]),                   
                   target_transform=Categorical(num_classes=5),
                   class_augmentations=[Rotation([90, 180, 270])],
                   meta_test=True,
                   download=True)
testdataset = ClassSplitter(testdataset, shuffle=True, num_train_per_class=cfg['training']['K-shot'], num_test_per_class=cfg['training']['K-shot'])
testdataloader = BatchMetaDataLoader(testdataset, batch_size=cfg['training']['batch_size'], num_workers=4)

#################################################
def test(model):
    train_cfg=cfg['training']
    #tb = SummaryWriter()
    model.to(device)

    e = 1
    avg_episodic_correct = 0.
    model.eval()
    for tasks in testdataloader:                
        train_inputs, train_targets = tasks["train"]
        test_inputs, test_targets = tasks["test"]

        train_inputs = train_inputs.to(device)
        test_inputs = test_inputs.to(device)
        train_targets = train_targets.to(device)
        test_targets = test_targets.to(device)
                                    
        correct = 0 
        for i in range(train_inputs.size(0)):            

            x = train_inputs[i]
            y = train_targets[i]            
            xt = test_inputs[i]
            yt = test_targets[i]            

            model.compute_prototype(x,y)
            yt_hat = model(xt)
                        
            _, predicted = torch.max(yt_hat, 1)            
            correct += (predicted == yt).sum().item()
        
        avg_episodic_correct += correct
                                    
        N = test_inputs.size(0)*test_inputs.size(1)
        print("Test Episode {} Done, Accuracy={:12.5}".format(e, 100.0*correct/N))

        #tb.add_scalars("Accuracy", {'Training Meta Accuracy':100.0*correct/N}, e)        
        if(e == train_cfg['num_test_episodes']):
            break        
        
        e +=1
    print(avg_episodic_correct,N)
    avg_episodic_accuracy = 100.0*avg_episodic_correct/(N*train_cfg['num_test_episodes'])
    print("Test Done, Avg Episodic Accuracy={:12.5}".format(avg_episodic_accuracy))
    #tb.add_scalar("Average Episodic Accuracy",avg_episodic_accuracy , e)        
    #tb.close()
    return avg_episodic_accuracy
#################################################
def train(model):

    print('-' * 59)
    train_cfg=cfg['training']

    model.train()
    model.to(device)
    
    #tb = SummaryWriter()
    #tb.add_text('Config',json.dumps(cfg))
    
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    meta_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    e = 1
    
    for tasks in dataloader:                
        train_inputs, train_targets = tasks["train"]
        test_inputs, test_targets = tasks["test"]

        train_inputs = train_inputs.to(device)
        test_inputs = test_inputs.to(device)
        train_targets = train_targets.to(device)
        test_targets = test_targets.to(device)
                
        meta_optimizer.zero_grad()
        meta_loss_c = 0.
        correct = 0   
        for i in range(train_inputs.size(0)):            
            x = train_inputs[i]
            y = train_targets[i]
            xt = test_inputs[i]
            yt = test_targets[i]            

            model.compute_prototype(x,y)
            yt_hat = model(xt)
            loss = criterion(yt_hat,yt)                     

            meta_loss_c +=loss.item()
            _, predicted = torch.max(yt_hat, 1)            
            correct += (predicted == yt).sum().item()        
            
            loss.backward()
        meta_optimizer.step()                            
                         
        N = test_inputs.size(0)*test_inputs.size(1)
        print("Episode {} Done, Loss = {:12.5}, Accuracy={:12.5}".format(e, meta_loss_c/N,100.0*correct/N))

        #tb.add_scalars("Loss", {'Training Meta loss':meta_loss_c/N}, e)
        #tb.add_scalars("Accuracy", {'Training Meta Accuracy':100.0*correct/N}, e)        
        
        
        if(e == train_cfg['num_episodes']):
            break        
        
        e +=1

    #tb.close()
    torch.save({
                    'epoch': e,                    
                    'model_state_dict': model.state_dict()
                }, os.path.join(train_cfg['chkpt_path'],train_cfg['chkpt_file']))
    print('-' * 59)

    
    return model

if __name__ == '__main__':
    print(cfg)
    torch.cuda.empty_cache()
    model = protoClassifier(cfg['model'], device)

    ## TRAIN Prototype CLASSIFIER
    model = train(model)

    ## TEST Prototype CLASSIFIER
    chkpt_file = os.path.join(cfg['training']['chkpt_path'],cfg['training']['chkpt_file'])        
    print('Loading checkpoint from:',chkpt_file)        
    checkpoint = torch.load(chkpt_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    test(model)

    