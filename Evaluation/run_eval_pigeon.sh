LOG_DIR="./log/"

for scenario in sticker
do
    for mode in test
    do
        for ckpt in 375
        do
            for mask_ratio in 2.0
            do
                for scale_llama in 1.0
                do
                    for scale_dm in 7.0
                    do
                        echo "scenario:$scenario, mode:$mode, ckpt:$ckpt, mask_ratio:$mask_ratio, scale_llama:$scale_llama, scale_dm:$scale_dm,"
                        LOG_FILE="$LOG_DIR/eval_DPO_${scenario}_${mode}_ckpt${ckpt}_histmask${mask_ratio}_scale_llama${scale_llama}_scale_dm${scale_dm}.txt"
                        CUDA_VISIBLE_DEVICES=3 python evaluate_pigeon.py \
                            --output_dir /checkpoints/sticker/DPO/ \
                            --data_path /datasets/SER-30K/processed_seq/ \
                            --img_folder_path /datasets/SER-30K/Images/ \
                            --dino_model_path /path/to/DINO/models--facebook--dinov2-large \
                            --batch_size 50 \
                            --clip_batch_size 512 \
                            --dataset $dataset \
                            --mode $mode \
                            --ckpt $ckpt \
                            --with_mask \
                            --scale_for_llama $scale_llama \
                            --scale_for_dm $scale_dm \
                            --mask_ratio $mask_ratio \
                            --seed 123 \
                            > "$LOG_FILE" 2>&1
                    done
                done
            done
        done
    done
done

# nohup sh run_eval_pigeon.sh >log_eval_pigeon.txt 2>&1 &
