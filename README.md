# Third Molar Extraction Type Classification
This is the repository for the final project in the course "Special Topics in Innovative Integration of Medicine and EECS".

We create a pipeline to detect the location of third molars and classify the extraction type of the corresponding third molar.

## Data processing
### Prerequsite
1. Set up a virtual environment and install required packages
```
pip install -r data_preprocess/requirements.txt
```
2. Download dataset
```
gdown --fuzzy https://drive.google.com/file/d/1vRJb1qpQ-NWRb1jHDksmvvNiLbcQurbD/view?usp=drive_link -O odontoai.zip
unzip odontoai.zip

mkdir data
mv odontoai/ data/
```

### main.py
Goal: Do preprocessing for the downstream task
```
python3 data_preprocess/main.py --version $version
```
The new dataset will be placed at `data/odontoai-v$version`

Provided versions
- 2: remove images with duplicate labels on tooth-18, 28, 38, 48
- 3: remove images which do not have tooth-18, 28, 38, 48

### cropper.py
Goal: Crop specific tooth to obtain training images for classification

**NOTICE**: `data/odontoai-v3` should be created in advance
```
for split in "train" "val"; do
    python3 data_preprocess/cropper.py \
        --source_image_folder data/odontoai-v3/$split/images \
        --source_anns_json data/odontoai-v3/$split/$split.json \
        --target_image_folder data/odontoai-cropped/$split/images
done
```
The new dataset will be placed at `data/odontoai-cropped`

### merge_label.py
Goal: Merge multiple annotated excels into one csv file

**NOTICE**: `data/odontoai-cropped` should be created in advance
```
gdown https://drive.google.com/drive/folders/11xrCLz9uMcruRNbHDKVb1ZuEjPYsj3wU?usp=drive_link -O data/odontoai-annotated-csv --folder

python3 data_preprocess/merge_label.py
```
The new dataset will be places at `data/odontoai-classification`


## Object detection
Please refer to `object_detection/Co-DETR/README.md` to see how to run the codes.


## Classification
Please refer to `classification` to see how to run the codes.


## Testing pipeline
You can refer to `test.sh`. Since the training of classification is fast, we do not provide the pretrained weight. Therefore, you should run the training process. Otherwise, there might be an error with weight not found.