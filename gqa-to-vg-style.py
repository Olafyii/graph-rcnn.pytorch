import json
import os
from tqdm import tqdm

print(os.path.dirname(__file__))
f = open('test.txt', 'w')
f.close()
f = open('VG/image_data.json')
img_data = json.load(f)
f.close()
f = open('VG/objects.json')
obj_data = json.load(f)
f.close()
f = open('VG/relationships.json')
rel_data = json.load(f)
f.close()

f = open('C:\\E\\data\\GQA\\sceneGraphs\\train_sceneGraphs.json')
gqa_train = json.load(f)
f.close()

gqa_im_data = []
for key in tqdm(gqa_train.keys()):
    gqa_im_data.append({'width':gqa_train[key]['width'], 'height':gqa_train[key]['height'], 'image_id':int(key)})
gqa_im_data = sorted(gqa_im_data, key=lambda x:int(x['image_id']))

gqa_obj_data = []
for key in tqdm(gqa_train.keys()):
    temp_d = {'image_id':int(key), 'split':'train'}
    obj_list = []
    for obj_key in gqa_train[key]['objects'].keys():
        obj_list.append({'object_id':obj_key, 'w':gqa_train[key]['objects'][obj_key]['w'], 'h':gqa_train[key]['objects'][obj_key]['h'], 'x':gqa_train[key]['objects'][obj_key]['x'], 'y':gqa_train[key]['objects'][obj_key]['y'], 'names':[gqa_train[key]['objects'][obj_key]['name']]})
    temp_d['objects'] = obj_list
    gqa_obj_data.append(temp_d)
gqa_obj_data = sorted(gqa_obj_data, key=lambda x:int(x['image_id']))

gqa_rel_data = []
for key in tqdm(gqa_train.keys()):
    rels = []
    for obj in gqa_train[key]['objects']:
        for rel in gqa_train[key]['objects'][obj]['relations']:
            rels.append({'predicate':rel['name'], 'object':{'object_id':obj}, 'subject':{'object_id':rel['object']}})
    gqa_rel_data.append({'image_id':int(key), 'relationships':rels})
gqa_rel_data = sorted(gqa_rel_data, key=lambda x:int(x['image_id']))

f = open('C:\\E\\data\\GQA\\sceneGraphs\\val_sceneGraphs.json')
gqa_val = json.load(f)
f.close()

for key in tqdm(gqa_val.keys()):
    gqa_im_data.append({'width':gqa_val[key]['width'], 'height':gqa_val[key]['height'], 'image_id':int(key)})
gqa_im_data = sorted(gqa_im_data, key=lambda x:int(x['image_id']))

for key in tqdm(gqa_val.keys()):
    temp_d = {'image_id':int(key), 'split':'test'}
    obj_list = []
    for obj_key in gqa_val[key]['objects'].keys():
        obj_list.append({'object_id':obj_key, 'w':gqa_val[key]['objects'][obj_key]['w'], 'h':gqa_val[key]['objects'][obj_key]['h'], 'x':gqa_val[key]['objects'][obj_key]['x'], 'y':gqa_val[key]['objects'][obj_key]['y'], 'names':[gqa_val[key]['objects'][obj_key]['name']]})
    temp_d['objects'] = obj_list
    gqa_obj_data.append(temp_d)
gqa_obj_data = sorted(gqa_obj_data, key=lambda x:int(x['image_id']))

for key in tqdm(gqa_val.keys()):
    rels = []
    for obj in gqa_val[key]['objects']:
        for rel in gqa_val[key]['objects'][obj]['relations']:
            rels.append({'predicate':rel['name'], 'object':{'object_id':obj}, 'subject':{'object_id':rel['object']}})
    gqa_rel_data.append({'image_id':int(key), 'relationships':rels})
gqa_rel_data = sorted(gqa_rel_data, key=lambda x:int(x['image_id']))

f = open('gqa_im_data.json', 'w')
json.dump(gqa_im_data, f)
f.close()
f = open('gqa_obj_data.json', 'w')
json.dump(gqa_obj_data, f)
f.close()
f = open('gqa_rel_data.json', 'w')
json.dump(gqa_rel_data, f)
f.close()