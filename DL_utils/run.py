#!/usr/bin/env python
from train import Train 
from arg import get_args 


def main():
    args = get_args()
    trainer =  Train(args)
    trainer.fit()
    
if __name__ == '__main__':
    main() 