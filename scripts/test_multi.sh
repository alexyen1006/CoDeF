GPUS=0

NAME=miami-surf
EXP_NAME=base

ROOT_DIRECTORY="DAVIS/$NAME/$NAME"
LOG_SAVE_PATH="logs/test_all_sequences/$NAME"
REAL_FILL_PATH="realfill/data/$NAME/target"

# MASK_DIRECTORY="all_sequences/$NAME/${NAME}_masks_0 all_sequences/$NAME/${NAME}_masks_1"

#WEIGHT_PATH=ckpts/all_sequences/$NAME/${EXP_NAME}/${NAME}.ckpt
WEIGHT_PATH=ckpts/all_sequences/$NAME/${EXP_NAME}/step=14000.ckpt

python train.py --test --encode_w \
                --root_dir $ROOT_DIRECTORY \
                --log_save_path $LOG_SAVE_PATH \
                --weight_path $WEIGHT_PATH \
                --real_fill_path $REAL_FILL_PATH\
                --gpus $GPUS \
                --config configs/${NAME}/${EXP_NAME}.yaml \
                --exp_name ${EXP_NAME} \
                --save_deform False
                # --mask_dir $MASK_DIRECTORY \
