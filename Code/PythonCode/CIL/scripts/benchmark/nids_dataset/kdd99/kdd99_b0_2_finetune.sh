python3.9 main.py \
    -model finetune \
    -init 2 -incre 2 \
    --dataset kdd99 \
    -net kdd_ann \
    -p benchmark -d -1 \
    --init_epoch 300 \
    --epoch 300 \
    --batch_size 512