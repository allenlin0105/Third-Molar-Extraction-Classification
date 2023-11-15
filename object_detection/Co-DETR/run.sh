# train
# python tools/train.py path_to_exp/checkpoints/co_deformable_detr_r50_1x_coco.py --gpu-id 0 --work-dir 'path_to_exp/checkpoints/'

# test 
# python tools/test.py path_to_exp/checkpoints/co_deformable_detr_r50_1x_coco.py path_to_exp/checkpoints/best_bbox_mAP_epoch_13.pth --gpu-id 0 --eval bbox

# visualize 
python output.py
