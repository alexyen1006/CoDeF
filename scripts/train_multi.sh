GPUS=1

NAME=bear
EXP_NAME=base

ROOT_DIRECTORY="DAVIS/$NAME/$NAME"
MODEL_SAVE_PATH="ckpts/all_sequences/$NAME"
LOG_SAVE_PATH="logs/all_sequences/$NAME"
REAL_FILL_PATH="realfill/data/$NAME/target"
#MASK_DIRECTORY="all_sequences/$NAME/${NAME}_masks_0 all_sequences/$NAME/${NAME}_masks_1"
FLOW_DIRECTORY="DAVIS/$NAME/${NAME}_flow"

python train.py --root_dir $ROOT_DIRECTORY \
                --model_save_path $MODEL_SAVE_PATH \
                --log_save_path $LOG_SAVE_PATH  \
                --real_fill_path $REAL_FILL_PATH\
                --flow_dir $FLOW_DIRECTORY \
                --gpus $GPUS \
                --encode_w --annealed \
                --config configs/${NAME}/${EXP_NAME}.yaml \
                --exp_name ${EXP_NAME}
                # --mask_dir $MASK_DIRECTORY \
