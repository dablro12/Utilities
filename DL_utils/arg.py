import argparse
import json

parser = argparse.ArgumentParser(description='DL Hyper-Parameter Setting')
############################################################################
# Data Setting
parser.add_argument("--save_path", "-path", default='./result')
parser.add_argument("--version", "-v", default='v1')
parser.add_argument("--cuda", "-gpu", type=str, default='0')
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--wandb", type=str, default="no")
parser.add_argument("--pretrain", type=str, default="no") #pretrained model loading
parser.add_argument("--pretrained_model", type=str, default='no') 
parser.add_argument("--error_signal", type=str, default='no') #-> 에러뜬거면 yes로 지정 

############################################################################
# Model Setting
parser.add_argument("--ts_batch_size", "-tbs", type=int, default=10)
parser.add_argument("--vs_batch_size", "-vbs", type=int, default=2)
parser.add_argument("--learning_rate", "-lr", type=float, default=1e-03)
parser.add_argument("--epochs", type=int, default=30) #만약 pretrain 모델을 불러오면 필요없음
parser.add_argument("--optimizer", "-opt", type=str, default="Adam")
parser.add_argument("--scheduler","-s", type=str, default="lambda")
parser.add_argument("--loss", "-l", type=str, default="mse")
parser.add_argument("--valid_epoch","-ve", type=int, default=1)
############################################################################
# Model Save
parser.add_argument("--model", "-m", type=str, default="temp")


args = parser.parse_args()


def get_args():
    return args


def save_args(save_path):
    with open(save_path, 'w') as file:
        json.dump(args.__dict__, file)
