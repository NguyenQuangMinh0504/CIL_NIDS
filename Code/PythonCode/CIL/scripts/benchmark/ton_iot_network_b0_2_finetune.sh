nohup python3.9 main.py \
    -model finetune \
    --dataset ton-iot-network \
    -net ton_iot_network_ann \
    -init 2 \
    -incre 2 \
    -p benchmark \
    -d -1 &