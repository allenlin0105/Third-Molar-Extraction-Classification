# your environment
run the following to build the environment:

```
conda env create -f ./environment.yml
conda activate co-detr
cd Co-DETR
bash set_env.sh
```

# Pretrained Model
Please download the pretrained model from https://drive.google.com/file/d/1Trs9WNPK6NP84nAJeDEQReYthTbMnIA9/view?usp=drive_link,
rename it as `pretrained_co_deformable_detr_r50_1x_coco.pth`,
and place it into `Co-DETR/path_to_exp/checkpoints/`.

# How to run your code
1. Train
```
python tools/train.py path_to_exp/checkpoints/co_deformable_detr_r50_1x_coco.py --gpu-id 0 --work-dir 'path_to_exp/checkpoints/'
```
2. Test metrics
```
python tools/test.py path_to_exp/checkpoints/co_deformable_detr_r50_1x_coco.py path_to_exp/checkpoints/best_bbox_mAP_epoch_13.pth --gpu-id 0 --eval bbox
```
3. Visualization
```
python output.py
```

output images will be saved in `Co-DETR/path_to_exp/result_img/`.


# Some modifications
I set the gpu usage limit to half in tools/train.py line 25, you can uncomment it and enlarge the batch size for better efficiency.