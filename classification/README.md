# Classification pipeline

## How to run with sample model
```
# Training process
python3 main.py --do_train
```

## How to test your custom model
1. Declare your model at `models/` folder
2. The `forward()` function for your model will read a batch of images as input, and expected to output a batch of predicted classes. For example, the input might be a tensor with size `(batch_size, channels, image_size, image_size)` and the output should be a tensor with size `(batch_size, n_classes)`
3. After setting up your model, you should modify `self.model` in `common/base_model.py` with your custom model, and you can train successfully with your model!
