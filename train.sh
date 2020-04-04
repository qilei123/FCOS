python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net.py \
    --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml \
    DATALOADER.NUM_WORKERS 2 \
    OUTPUT_DIR /data2/qilei_chen/DATA/COCO/FCOS_training_dir/fcos_imprv_R_50_FPN_1x