python ./run.py --model "vgg16" \
                --version "v3" \
                --cuda "0"\
                --ts_batch_size 50\
                --vs_batch_size 25\
                --epochs 200\
                --loss "BCE"\
                --optimizer "AdamW"\
                --learning_rate 0.0001\
                --scheduler "lambda"\
                --pretrain "no" --pretrained_model "practice" --error_signal no\
                --wandb "yes"\