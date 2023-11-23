DEVICE=0

cd object_detection/Co-DETR
source .venv/bin/activate
gdown --fuzzy https://drive.google.com/file/d/1XfVSeKAxpAq9sgxlnw5buX5KWS_jDXSw/view?usp=sharing
python3 output.py --save_bbox
deactivate

cd ../../
source .venv/bin/activate
python3 data_preprocess/cropper.py \
    --source_image_folder data/odontoai/test/images \
    --source_anns_json object_detection/Co-DETR/path_to_exp/test_bbox.json \
    --target_image_folder data/odontoai-cropped/test/images

cd classification
python3 main.py --do_test \
    --dataset_folder ../data/odontoai-cropped/ \
    --test_version 0 