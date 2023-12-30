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
import torchvision.models as models
import custom_models as custom_models

#metric
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import torchmetrics.functional as tmf

#numeric
import numpy as np
import pandas as pd 

#visualization
import matplotlib.pyplot as plt

#system 
from tqdm import tqdm
import os 
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
            transforms.ToTensor(),
            transforms.Resize((224,224), antialias=True),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation(degrees=30),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(contrast=2),
        ])

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224), antialias=True),
            transforms.Grayscale(num_output_channels=3)
        ])

        train_dataset = CustomDataset(root_dir = 'data/train', transform= train_transform)
        valid_dataset = CustomDataset(root_dir = 'data/valid', transform= valid_transform)

        self.train_loader = DataLoader(
                                    dataset = train_dataset,
                                    batch_size = args.ts_batch_size,
                                    shuffle = True,
                                    num_workers= 4,
                                    pin_memory= False
                                    )
        self.valid_loader = DataLoader(
                                    dataset = valid_dataset,
                                    batch_size = args.vs_batch_size,
                                    shuffle = False,
                                    num_workers=2,
                                    pin_memory= False
                                    )
        
        ######################################### Wan DB #########################################
        # wandb 실행여부 yes/그외
        self.w = args.wandb
        if args.wandb == 'yes':
            wandb.init(
                project = 'MobileNet_v2',
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
            self.model = custom_models.AutoEncoder(input_dim = 28*28, hidden_dim1 =32, hidden_dim2= 64).to(self.device)
            
            self.model.load_state_dict(model_file['model_state_dict'])
                
            self.epochs = model_file['epochs']
            if args.error_signal == 'yes': #-> 에러뜬거면 yes로 지정하기
                self.epoch = model_file['epoch']
            else:
                self.epoch = 0
            self.lr = model_file['learning_rate']
            self.loss = model_file['loss'].to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
            self.optimizer.load_state_dict(model_file['optimizer_state_dict'])
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer = self.optimizer, lr_lambda=lambda epoch: 0.95 ** self.epochs)
            self.name = model_file['model']
            self.t_loss_li = model_file['t_loss']
            self.v_loss_li = model_file['v_loss']
            self.version = args.version
            self.ts_batch = args.ts_batch_size
            self.vs_batch = args.vs_batch_size
            
        else:
            self.model = models.mobilenet_v2(pretrained = True)
            self.model.classifier[-1] = nn.Linear(1280, 3) #mobilenet_v2  #3개의 채널에 대해 해줌
            self.model.to(self.device)
            # self.model = custom_models.AutoEncoder(input_dim = 28*28, hidden_dim1 =32, hidden_dim2= 64).to(self.device)
            print(f"Training Model : {args.model} | status : \033[42mNEW\033[0m")

            ############################# Hyper Parameter Setting ################################
            self.loss = nn.CrossEntropyLoss().to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr = args.learning_rate)
            self.scheduler = optim.lr_scheduler.LambdaLR(optimizer = self.optimizer, lr_lambda=lambda epoch: 0.95)
            self.epochs = args.epochs
            self.epoch = 0
            self.ve = args.valid_epoch 
            self.lr = args.learning_rate
            self.name = name 
            self.t_loss_li = []
            self.v_loss_li = []
            self.ts_batch = args.ts_batch_size
            self.vs_batch = args.vs_batch_size
            self.version = args.version
            ############################# Hyper Parameter Setting ################################
            
        ############################### Metrics Setting########################################
        self.accuracy = accuracy_score
        self.precsion = precision_score
        self.recall = recall_score
        self.f1 = f1_score
        self.roc_curve = roc_curve
        ############################### Metrics Setting########################################
        
        # Training 
        print("\033[41mFinished Initalization\033[0m")
        print("\033[41mStart Training\033[0m")
        
    def fit(self):
        train_loss_li = self.t_loss_li
        valid_loss_li = self.v_loss_li
        
        for epoch in tqdm(range(self.epoch, self.epochs)):
            ################################# train ################################# 
            self.model.train() 
            train_losses = 0. 
            
            predicted_probs = []
            true_labels = []
            for _ , (inputs, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                
                inputs, labels = inputs.to(self.device), labels.long().view(-1).to(self.device)
                # Forward
                outputs = self.model(inputs)
                # softmax거쳐서 클래스 확률로 변환
                outputs = F.softmax(outputs, dim = 1)
                train_loss = self.loss(outputs, labels)
                # backward & optimizer 
                train_loss.backward()
                self.optimizer.step() 
                
                train_losses += train_loss.item()

                # metrics 뽑고 싶을떄만 보기 # 각 행에서 가장 큰 값의 인덱스를 예측 클래스로 선택
                # predicted_probs.extend(torch.argmax(outputs, dim = 1).cpu().numpy())
                # true_labels.extend(labels.cpu().numpy())
                # print(predicted_probs)
                # print(true_labels)
                
            train_loss_li.append(train_losses / len(labels))
            self.scheduler.step()
            if self.w == "yes":
                wandb.log({
                    "learning_rate" : self.lr,
                    "training_loss" : train_losses / len(labels),
                }, step = epoch)
            
        
            ################################# valid #################################
            self.model.eval()
            if epoch % 1 == 0: 
                valid_losses = 0.
                for _ , (inputs, labels) in enumerate(self.valid_loader):
                    inputs, labels = inputs.to(self.device), labels.long().view(-1).to(self.device)
                    
                    # Forward
                    outputs = self.model(inputs)
                    outputs = F.softmax(outputs, dim = 1)
                    
                    valid_loss = self.loss(outputs, labels)
                    valid_losses += valid_loss.item()
                valid_loss_li.append(valid_losses / len(labels))
                print(f"\033[42mEpoch [{epoch}/{self.epochs}], Train Loss : {(train_losses/self.ts_batch):.4f}, Valid Loss : {(valid_losses/self.vs_batch):.4f}\033[0m")

                ## Display to Wandb for validation loss
                if self.w == "yes":
                    wandb.log({
                        "validation_loss" : valid_losses / len(labels),
                    }, step = epoch)
                
                ## model save
                torch.save({
                    "model" : str(self.name) + f"{self.version}_{epoch}",
                    "epoch" : epoch,
                    "epochs" : self.epochs,
                    "model_state_dict" : self.model.state_dict(),
                    "optimizer_state_dict" : self.optimizer.state_dict(),
                    "learning_rate" : self.lr,
                    "loss" : self.loss,
                    "t_loss" : train_loss_li,
                    "v_loss" : valid_loss_li,
                    "description" : f"{self.name} training status : {epoch}/{self.epochs}"
                },
                self.model_save_path)
        
        print("="*100)
        print(f"\033[41mFinished Training\033[0m | Save model PATH : {self.model_save_path}")
        if self.w == "yes":
            wandb.finish()