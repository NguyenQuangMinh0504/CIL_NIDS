python3.9 main_memo.py \
    -model memo \
    -init 5 \
    -incre 5 \
    -net memo_resnet32 \
    --dataset cifar100 \
    --train_base \
    --scheduler steplr \
    --init_epoch 200 \
    --batch_size 128 \
    -d -1 \
    