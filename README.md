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
4. Run preprocessing
```
python3 data_preprocess/main.py --version 2
```

The new dataset will be placed in the `data` folder, which is `odontoai-v2`
```
.
├── data
│   ├── odontoai
│   └── odontoai-v2
└── data_preprocess
    └── main.py
```

### Provided versions
- version 2: remove images with duplicate labels on tooth-18, 28, 38, 48
- version 3: remove images which do not have tooth-18, 28, 38, 48