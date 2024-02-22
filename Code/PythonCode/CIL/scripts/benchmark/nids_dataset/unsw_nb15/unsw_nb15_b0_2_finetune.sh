python3 main.py \
    -model finetune \
    -init 2 -incre 2 \
    --dataset unsw-nb15 \
    -net unsw_nb15_ann \
    -p benchmark -d -1 \
    --init_epoch 300 \
    --epochs 300 \
    --batch_size 128