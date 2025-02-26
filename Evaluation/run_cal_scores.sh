LOG_DIR="./log/"

for scenario in sticker
do
    for mode in test
    do
        for ckpt in 375
        do
            for scale_llama in 1.0
            do
                for scale_dm in 7.0
                do
                    echo "scenario:$scenario, mode:$mode, ckpt:$ckpt, scale_llama:$scale_llama, scale_dm:$scale_dm,"
                    LOG_FILE="$LOG_DIR/cal_scores_DPO_${scenario}_${mode}_ckpt${ckpt}_scale_llama${scale_llama}_scale_dm${scale_dm}.txt"
                    CUDA_VISIBLE_DEVICES=3 python cal_scores.py \
                        --output_dir /checkpoints/sticker/DPO/ \
                        --data_path /datasets/SER-30K/processed_seq/ \
                        --img_folder_path /datasets/SER-30K/Images/ \
                        --batch_size 50 \
                        --mode $mode \
                        --ckpt $ckpt \
                        --scale_for_llm $scale_llama \
                        --scale_for_dm $scale_dm \
                        --seed 123 \
                        > "$LOG_FILE" 2>&1
                done
            done
        done
    done
done

# nohup sh run_cal_scores.sh >log_cal_scores.txt 2>&1 &
