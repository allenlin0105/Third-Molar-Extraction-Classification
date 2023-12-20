# Co-DETR

## Set up your environment
run the following to build the environment:
```
conda env create -f ./environment.yml
conda activate co-detr
cd Co-DETR
bash set_env.sh
```

## How to run the code
1. Train
```
python3 tools/train.py \
    path_to_exp/checkpoints/co_deformable_detr_r50_1x_coco.py \
    --gpu-id 0 \
    --work-dir 'path_to_exp/checkpoints/'
```
2. Test
```
python tools/test.py \
    path_to_exp/checkpoints/co_deformable_detr_r50_1x_coco.py \
    path_to_exp/checkpoints/best_bbox_mAP_epoch_13.pth \
    --gpu-id 0 --eval bbox
```
3. Visualization
```
python3 output.py --split test --visualize --save_bbox
```
Output images will be saved in `Co-DETR/path_to_exp/test_result_img/`.
