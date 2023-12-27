nohup python3.9 main.py \
    -model finetune \
    -init 5 -incre 5 \
    --dataset cic-ids-2017 \
    -net cic_ids_ann \
    -p benchmark -d -1 &