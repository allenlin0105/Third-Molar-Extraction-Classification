# Tooth-Final-Project
Group2's final project repo

## Possible folder structure
```
.
├── data
│   └── odontoai
├── object_detection
│   └── dino
│       └── source codes (.py files)
└── data_preprocess
    └── main.py
```

## How to run data preprocess
### Prerequsite
1. Create a `data` folder and place `odontoai` into the folder
```
.
├── data
│   └── odontoai
└── data_preprocess
    └── main.py
```
2. Set up a virtual environment
3. Install required packages
```
pip install -r requirements.txt
```

### main.py
Goal: Filter images for downstream task
```
python3 data_preprocess/main.py --version $version
```
The new dataset will be placed at `data/odontoai-v$version`

Provided versions
- version 2: remove images with duplicate labels on tooth-18, 28, 38, 48
- version 3: remove images which do not have tooth-18, 28, 38, 48

### cropper.py
Goal: Crop specific tooth images for classification

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