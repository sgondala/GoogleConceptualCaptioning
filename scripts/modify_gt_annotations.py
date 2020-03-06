import json

images_list = json.load(open('data/coco_gt_objs.json', 'r'))['images']
images_to_gt_json = {}

for entry in images_list:
    image_id = entry['image_id']
    category = entry['category']
    images_to_gt_json['image_id'] = category

json.dump(images_to_gt_json, open('data/coco_gt_objs_modified.json', 'w'))