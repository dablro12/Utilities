#!/usr/bin/env python

#default
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#trasnform
from torchvision import transforms

#dataset
from dataset import CustomDataset
from torch.utils.data import DataLoader

#model 
import torchvision.models as models
from custom_models import *

#metric
from sklearn.metrics import recall_score, f1_score, accuracy_score


#numeric
import numpy as np
import pandas as pd 

#visualization
import matplotlib.pyplot as plt

#system 
from tqdm import tqdm
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import wandb

#parser
from arg import save_args 



torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.autograd.set_detect_anomaly(True)

class Train(nn.Module):
    def __init__(self, args):
        super().__init__()
        print('=' * 100)
        print('=' * 100)
        print("\033[41mStart Initialization\033[0m")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\033[41mCUDA Status : {self.device.type}\033[0m")
        
        ########################## Data set & Data Loader ##############################
        # Data set & Data Loader
        train_transform = transforms.Compose([
            transforms.Resize((224, 224), antialias= True), # 혹은 모델에 맞는 다른 사이즈로 조정
            transforms.ToTensor(),
            transforms.RandomApply([
                transforms.RandomRotation(degrees=15),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.75, 1.33), antialias= True),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
            ], p=0.7),
        ])



        valid_transform = transforms.Compose([
            transforms.Resize((224,224), antialias=True),
            transforms.ToTensor(),
            # transforms.Resize((299,299), antialias=True),
            transforms.Grayscale(num_output_channels=3),
            # transforms.Normalize((0.5), (0.5)),
            
        ])

        train_dataset = CustomDataset(root_dir = 'data/train', transform= train_transform)
        valid_dataset = CustomDataset(root_dir = 'data/valid', transform= valid_transform)

        self.train_loader = DataLoader(
                                    dataset = train_dataset,
                                    batch_size = args.ts_batch_size,
                                    shuffle = True,
                                    num_workers= 8,
                                    pin_memory= True,
                                    )
        self.valid_loader = DataLoader(
                                    dataset = valid_dataset,
                                    batch_size = args.vs_batch_size,
                                    shuffle = False,
                                    num_workers=2,
                                    pin_memory= True,
                                    )
        
        ######################################### Wan DB #########################################
        # wandb 실행여부 yes/그외
        self.w = args.wandb
        if args.wandb == 'yes':
            wandb.init(
                project = 'PCOS', 
                entity = 'dablro1232',
                notes = 'baseline',
                config = args.__dict__,
            )
            name = wandb.run.name
        else:
            name = args.model + f'_{args.version}'
        ######################################### Wan DB #########################################
        
        ######################################### Saving File #########################################
        # model save할 경로 설정
        self.save_path = os.path.join(args.save_path, f"{name}")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.arg_path = f"{self.save_path}/{name}.json" #인자 save할 경로 설정
        save_args(self.arg_path)
        self.model_save_path = f"{self.save_path}/{name}.pt"
        ######################################### Saving File #########################################
        
        ############################## Model Initialization & GPU Setting ##############################
        if args.pretrain == 'yes': #pretrained model 사용여부
            wandb_name = args.pretrained_model
            PATH = f"./result/{wandb_name}/{wandb_name}.pt"
            print(f"Previous model : {PATH} | \033[41mstatus : Pretrained Update\033[0m")
            
            model_file = torch.load(PATH)
            # self.model = custom_models.AutoEncoder(input_dim = 28*28, hidden_dim1 =32, hidden_dim2= 64).to(self.device)
            
            self.model.load_state_dict(model_file['model_state_dict'])
                
            self.epochs = model_file['epochs']
            if args.error_signal == 'yes': #-> 에러뜬거면 yes로 지정하기
                self.epoch = model_file['epoch']
            else:
                self.epoch = 0
            self.lr = model_file['learning_rate']
            self.loss = model_file['loss'].to(self.device)
            self.optimizer = optim.AdamW(self.model.parameters(), lr = self.lr)
            self.optimizer.load_state_dict(model_file['optimizer_state_dict'])
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer = self.optimizer, lr_lambda=lambda epoch: 0.95 ** self.epochs)
            self.name = model_file['model']
            self.t_loss_li = model_file['t_loss']
            self.v_loss_li = model_file['v_loss']
            self.version = args.version
            self.ts_batch = args.ts_batch_size
            self.vs_batch = args.vs_batch_size
            
        else:
            self.model = pretrained_vgg16_binary()
            self.model.to(self.device)

            print(f"Training Model : {args.model} | status : \033[42mNEW\033[0m")

            ############################# Hyper Parameter Setting ################################
            self.loss = nn.BCEWithLogitsLoss().to(self.device)
            self.optimizer = optim.AdamW(self.model.parameters(), lr = args.learning_rate)
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer = self.optimizer,
                                                        lr_lambda=lambda epoch: 0.95 ** epoch,
                                                        last_epoch = -1,
                                                        verbose= True)
            self.epochs = args.epochs
            self.epoch = 0
            self.ve = args.valid_epoch 
            self.lr = args.learning_rate
            self.name = name 
            self.ts_batch = args.ts_batch_size
            self.vs_batch = args.vs_batch_size
            self.version = args.version
            self.model_name = args.model    
        
            ############################# Hyper Parameter Setting ################################
            self.early_stopping_epochs, self.early_stop_cnt = 5, 0
            self.best_loss = float('inf')
            
        ############################### Metrics Setting########################################
        self.metrics = {
            'train_loss' : [],
            'valid_loss' : [],
            'train_accuracy' : [],
            'train_f1' : [],
            'train_recall' : [],
            'valid_accuracy' : [],
            'valid_f1' : [],
            'valid_recall' : [],
        }

        ############################### Metrics Setting########################################
        
        # Training 
        print("\033[41mFinished Initalization\033[0m")
        print("\033[41mStart Training\033[0m")
    
    def fit(self):
        for epoch in tqdm(range(self.epoch, self.epochs)):
            train_losses, valid_losses = 0., 0.
            train_target, train_pred, valid_target, valid_pred = [], [], [], [] 
            
            self.model.train()
            for _, (inputs, labels) in tqdm(enumerate(self.train_loader)):
                self.optimizer.zero_grad()
                inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                outputs = self.model(inputs)
                
                train_loss = self.loss(outputs, labels)
                train_loss.backward()
                self.optimizer.step()
                train_losses += train_loss.item()
                
                # 예측 값을 이진 레이블로 변환
                pred = (F.sigmoid(outputs) >0.5).float()
                train_target.extend(labels.detach().cpu().numpy())
                train_pred.extend(pred.detach().cpu().numpy())

            self.metrics['train_loss'].append(train_losses/len(self.train_loader))
            self.metrics['train_accuracy'].append(accuracy_score(train_target, train_pred))
            self.metrics['train_f1'].append(f1_score(train_target, train_pred))
            self.metrics['train_recall'].append(recall_score(train_target, train_pred))
            
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]
                
            if self.w == "yes":
                wandb.log({
                    "Learning_Rate" : lr,
                    "train_LOSS" : self.metrics['train_loss'][-1],
                    "train_ACC" : self.metrics['train_accuracy'][-1],
                    "train_F1Score" : self.metrics['train_f1'][-1],
                    "train_RECALL" : self.metrics['train_recall'][-1],
                }, step = epoch)
                
        
            ################################# valid #################################
            with torch.no_grad():
                self.model.eval()
                if epoch % 1 == 0: 
                    for _ , (inputs, labels) in enumerate(self.valid_loader):
                        inputs, labels = inputs.to(self.device), labels.float().to(self.device)
                        outputs = self.model(inputs)
                        
                        valid_loss = self.loss(outputs, labels)
                        valid_losses += valid_loss.item()

                        # 예측 값을 이진 레이블로 변환
                        pred = (F.sigmoid(outputs) > 0.5).float()
                        valid_target.extend(labels.detach().cpu().numpy())
                        valid_pred.extend(pred.detach().cpu().numpy())
                        
                    self.metrics['valid_loss'].append(valid_losses/len(self.valid_loader))
                    self.metrics['valid_accuracy'].append(accuracy_score(valid_target, valid_pred))
                    self.metrics['valid_f1'].append(f1_score(valid_target, valid_pred))
                    self.metrics['valid_recall'].append(recall_score(valid_target, valid_pred))
                    
                    
            print("#"*100)    
            print(f"LOSS : {self.metrics['train_loss'][-1]} | {self.metrics['valid_loss'][-1]}\n ACC : {self.metrics['train_accuracy'][-1]} | {self.metrics['valid_accuracy'][-1]}\n F1 : {self.metrics['train_f1'][-1]} | {self.metrics['valid_f1'][-1]}\n RECALL : {self.metrics['train_recall'][-1]} | {self.metrics['valid_recall'][-1]}")
            print("#"*100)
                    
            ## Display to Wandb for validation loss
            if self.w == "yes":
                wandb.log({
                    "valid_LOSS" : self.metrics['valid_loss'][-1],
                    "valid_ACC" : self.metrics['valid_accuracy'][-1],
                    "valid_F1Score" : self.metrics['valid_f1'][-1],
                    "valid_RECALL" : self.metrics['valid_recall'][-1],
                }, step = epoch)
            
            # 조기 종료 조건 확인
            if self.early_stop_cnt >= self.early_stopping_epochs:
                print(f"Early Stops!!! : {epoch}/{self.epochs}")
                torch.save({
                    "model" : f"{self.name}" + f"{self.version}_{epoch}",
                    "epoch" : epoch,
                    "epochs" : self.epochs,
                    "model_state_dict" : self.model.state_dict(),
                    "optimizer_state_dict" : self.optimizer.state_dict(),
                    "learning_rate" : lr,
                    "loss" : self.loss,
                    "metric" : self.metrics,
                    "description" : f"Training status : {epoch}/{self.epochs}"
                },
                self.model_save_path)
                
                print(f"SAVE MODEL PATH : {self.model_save_path}")
                break

            elif self.metrics['valid_recall'][-1] >= np.array(self.metrics['valid_recall']).max():
                torch.save({
                    "model" : f"{self.name}" + f"{self.version}_{epoch}",
                    "epoch" : epoch,
                    "epochs" : self.epochs,
                    "model_state_dict" : self.model.state_dict(),
                    "optimizer_state_dict" : self.optimizer.state_dict(),
                    "learning_rate" : lr,
                    "loss" : self.loss,
                    "metric" : self.metrics,
                    "description" : f"Training status : {epoch}/{self.epochs}"
                },
                self.model_save_path)
                
                print(f"SAVE MODEL PATH : {self.model_save_path}")
                    

        print("="*100)
        print(f"\033[41mFinished Training\033[0m | Save model PATH : {self.model_save_path}")
        if self.w == "yes":
            wandb.finish()