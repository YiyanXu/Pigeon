LOG_DIR="./log"

for seed in 123
do
    for lr in 5e-6
    do
        for dropout in 0.05    
        do
            echo "lr: $lr, dropout: $dropout , seed: $seed,"
            LOG_FILE="$LOG_DIR/finetune_dpo_seed${seed}_lr${lr}_dropout${dropout}.txt"
            CUDA_VISIBLE_DEVICES=4 python finetune_dpo.py \
                --model_path /path/to/LaVIT-7B-v2/ \
                --model_dtype bf16 \
                --output_dir /checkpoints/sticker/DPO/ \
                --pre_ckpt /checkpoints/sticker/SFT/checkpoint-2975/ \
                --use_xformers \
                --load_in_8bit \
                --pixel_decoding highres \
                --mask_type mutual \
                --num_heads 4 \
                --num_layers 1 \
                --drop_prob 0.2 \
                --hist_mask_ratio 0.2 \
                --scenario sticker \
                --data_path /dataset/SER-30K/processed_seq/ \
                --img_folder_path /dataset/SER-30K/Images/ \
                --mode dpo \
                --batch_size 8 \
                --micro_batch_size 8 \
                --num_epochs 50 \
                --learning_rate $lr \
                --lr_schedule_type linear \
                --min_learning_rate 1e-6 \
                --lora_r 8 \
                --lora_alpha 16\
                --lora_dropout $dropout \
                --lora_target_modules '[q_proj,v_proj]' \
                --seed $seed \
                --resume_from_checkpoint \
                --logging_steps 25 \
                --eval_steps 25 \
                --save_steps 25 \
                --eval_num 200 \
                > "$LOG_FILE" 2>&1
        done
    done
done

# nohup sh run_dpo.sh >log_dpo.txt 2>&1 &
