nohup python3.9 main.py \
    -model finetune \
    -init 5 -incre 5 \
    --dataset kdd99 \
    -net kdd_ann \
     -p benchmark -d -1 &