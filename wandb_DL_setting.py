#!/usr/bin/env python

## https://minimin2.tistory.com/186
import wandb
import random 


# 새로운 wandb 시작 스클비트
wandb.init(
    # project name 쓰기
    project = "my-first-project",
    # 실행시 init()설정을 재시작 
    reinit = True,
    
    #hyper parmeter setting 및 meta data 시작
    config = {
        "learning_rate" : 42e-4,
        "architecture" : "CNN",
        "dataset" : "CIRFAR-100",
        "epochs" : 10,
    }
)

# Training 시뮬레이션
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1- 2 ** - epoch - random.random() / epoch - offset 
    loss =  2 ** -epoch + random.random() / epoch + offset 
    
    # log metrics 결과를 wandb 로 보냄 
    wandb.log({"acc" : acc, "loss" : loss})
    
# wandb 시작한거 종료 * jupyter 에서는 필수로 적용
wandb.finish()


