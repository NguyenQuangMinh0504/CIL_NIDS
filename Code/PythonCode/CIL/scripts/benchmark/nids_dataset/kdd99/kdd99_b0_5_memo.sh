python3 main_memo.py \
    -model memo \
    -init 5 \
    -incre 5 \
    -net kdd_fc \
    --dataset kdd99 \
    --train_base \
    --scheduler steplr \
    --init_epoch 20 \
    --epochs 20 \
    --batch_size 512 \
    -d -1