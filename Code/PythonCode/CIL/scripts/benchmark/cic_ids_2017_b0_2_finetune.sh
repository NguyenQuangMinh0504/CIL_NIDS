python3 main.py \
    -model finetune \
    -init 2 -incre 2 \
    --dataset cic-ids-2017 \
    -net cic_ids_dnn \
    -p benchmark -d -1 \
    --init_epoch 300 \
    --epochs 300 \
    --batch_size 128