python3.9 main_memo.py \
    -model memo_kdd \
    -init 2 \
    -incre 2 \
    -net ton_iot_network_memo_ann \
    --dataset ton-iot-network \
    --train_base \
    --scheduler steplr \
    --init_epoch 50 \
    --epochs 50 \
    --batch_size 128 \
    -d -1