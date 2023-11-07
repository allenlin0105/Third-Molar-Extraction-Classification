import json
from collections import defaultdict

class COCOParser:
    """
    image_dict: id maps to a image
    e.g., {"id": 1, "width": 2440, "height": 1292, "file_name": "pan-00577.jpg"}

    category_dict: id maps to a category
    e.g., {"id": 1, "name": "tooth-11", "supercategory": "tooth"}

    image_id2anns_dict: id maps to a list of dict
    [{
        "id": 10, 
        "image_id": 1, 
        "category_id": 20, 
        "segmentation": [[1537, 984, 1534, 985, 1528, 978, 1522, 960, 1508, 905, 1498, 850, 1493, 832, 1483, 812, 1482, 801, 1482, 789, 1489, 779, 1496, 772, 1501, 769, 1507, 773, 1522, 764, 1530, 765, 1549, 779, 1554, 779, 1560, 789, 1555, 912, 1545, 970, 1541, 981]], 
        "area": 10979, 
        "bbox": [1482, 764, 78, 221], 
        "iscrowd": false, 
        "width": 2440, 
        "height": 1292
    }, ...]
    """
    def __init__(self, anns_file):
        with open(anns_file, 'r') as f:
            coco = json.load(f)
            
        self.image_dict = {}
        self.category_dict = {}
        self.image_id2anns_dict = defaultdict(list)

        for image in coco['images']:
            self.image_dict[image['id']] = image
        for category in coco['categories']:
            self.category_dict[category['id']] = category
        for ann in coco['annotations']:           
            self.image_id2anns_dict[ann['image_id']].append(ann) 

    def get_image_ids(self):
        return list(self.image_dict.keys())
    
    def get_image_filename(self, image_id):
        return self.image_dict[image_id]["file_name"]
    
    def get_image_wh(self, image_id):
        return (self.image_dict[image_id]["width"], self.image_dict[image_id]["height"])
    
    def get_category_ids(self,):
        return list(self.category_dict.keys())
    
    def get_category_name(self, category_id):
        return self.category_dict[category_id]["name"]
    
    def get_anns(self, image_id):
        return self.image_id2anns_dict[image_id]
   
