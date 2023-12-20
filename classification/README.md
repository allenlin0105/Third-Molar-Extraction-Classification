# Classification pipeline

## NOTICE
To run the classification, `data/odontoai-classification` dataset should be created first.

You can refer to the README at the root folder and see how to set up this dataset.


## How to run
1. Train
```
python3 main.py --do_train \
    --model resnet \
    --contrast 2.5
```
2. Test
```
python3 main.py --do_test --test_version 0 \
    --model resnet \
    --dataset_folder ../data/odontoai-cropped/ \
    --contrast 2.5
```
Output will be saved in `lightning_logs/version_0/output/`


## How to train your custom model
1. Declare your model at `models/` folder
2. The `forward()` function for your model will read a batch of images as input, and expected to output a batch of predicted classes. For example, the input might be a tensor with size `(batch_size, channels, image_size, image_size)` and the output should be a tensor with size `(batch_size, n_classes)`
3. After setting up your model, you should modify `self.model` in `common/base_model.py` with your custom model, and you can train successfully with your model!
