python3.9 main_memo.py \
    -model memo_kdd \
    -init 2 \
    -incre 2 \
    -net cic_ids_memo_dnn \
    --dataset cic-ids-2017 \
    --train_base \
    --scheduler steplr \
    --init_epoch 300 \
    --epochs 300 \
    --batch_size 512 \
    -d -1   