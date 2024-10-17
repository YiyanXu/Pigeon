LOG_DIR="./log"

for mode in test
do
    for checkpoint in 375
    do
        for mask_ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
        do
            for llama_scale in 1.0
            do
                for dm_scale in 7.0    
                do
                    echo "Running Inference on checkpoint-$checkpoint, mask_ratio: $mask_ratio, llama_scale: $llama_scale , dm_scale: $dm_scale,"
                    LOG_FILE="$LOG_DIR/inference_${mode}_llama_ckpt${checkpoint}_mask_ratio${mask_ratio}_llama_scale${llama_scale}_dm_scale${dm_scale}.txt"
                    CUDA_VISIBLE_DEVICES=3 python inference.py \
                        --model_path /path/to/LaVIT-7B-v2/ \
                        --model_dtype bf16 \
                        --output_dir /checkpoints/sticker/DPO/ \
                        --pre_ckpt /checkpoints/sticker/SFT/checkpoint-2975/ \
                        --mode $mode \
                        --use_xformers \
                        --load_in_8bit \
                        --pixel_decoding highres \
                        --mask_type mutual \
                        --mask_ratio $mask_ratio \
                        --hist_mask_ratio 0.2 \
                        --with_mask \
                        --scenario sticker \
                        --data_path /dataset/SER-30K/processed_seq/ \
                        --img_folder_path /dataset/SER-30K/Images/ \
                        --batch_size 3 \
                        --dm_batch_size 4 \
                        --seed 123 \
                        --resume_from_checkpoint $checkpoint \
                        --use_nucleus_sampling \
                        --top_p 1.0 \
                        --top_k 50 \
                        --temperature 1 \
                        --num_beams 4 \
                        --min_length 20 \
                        --length_penalty 1 \
                        --num_return_sequences 1 \
                        --guidance_scale_for_llm $llama_scale \
                        --ratio 1:1 \
                        --guidance_scale_for_dm $dm_scale \
                        --num_inference_steps 25 \
                        --num_return_images 1 \
                        > "$LOG_FILE" 2>&1
                done
            done
        done
    done
done

# nohup sh run_inf.sh >log_inf.txt 2>&1 &
