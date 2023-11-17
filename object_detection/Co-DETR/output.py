import os
from tqdm import tqdm
 
from mmdet.apis import init_detector
from mmdet.apis import inference_detector
 
# 模型配置文件
config_file = 'path_to_exp/checkpoints/co_deformable_detr_r50_1x_coco.py'
# 預訓練模型文件
checkpoint_file = 'path_to_exp/checkpoints/best_bbox_mAP_epoch_13.pth'
# 通過模型配置文件與預訓練文件構建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# valid
# img_dir = ../../data/odontoai-v2/val/images/
# test
img_dir = '../../data/odontoai-v2/test/images/'
out_dir = 'path_to_exp/result_img/'

# output_dict = dict()
for filename in tqdm(os.listdir(img_dir)):
    if (filename == '.DS_Store'):
        continue
    # print(filename)

    result = inference_detector(model, img_dir+filename)
    # result[label][bbox]
    # print(result)

    model.show_result(img_dir+filename, result, out_file=out_dir+filename)

    # """Convert detection results to json."""
    # json_results = {"boxes": [], "labels": [], "scores": []}
    # for label in range(len(result)):
    #     bboxes = result[label]      # bboxes[x, y, x, y, score]

    #     for i in range(bboxes.shape[0]):
    #         # print(bboxes[i])
    #         json_results["boxes"].append(bboxes[i][0:4].tolist())
    #         json_results["labels"].append(label)
    #         json_results["scores"].append(float(bboxes[i][4]))
        
    # output_dict[filename] = json_results
    # print(output_dict)

# with open('path_to_exp/output.json', 'w') as output_json:
#     output_json.write(json.dumps(output_dict, indent=4))