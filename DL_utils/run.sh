python3 run.py --model "model_name" \
                --version "v1" \
                 --cuda "1"\
                 --ts_batch_size 3\
                 --vs_batch_size 1\
                 --epochs 30\
                 --loss "BCEWithLogitsLoss"\
                 --optimizer "Adam"\
                 --learning_rate 1e-03\
                 --scheduler "lambda"\
                 --pretrain "no" --pretrained_model "practice" --error_signal no\
                 --wandb "yes"\