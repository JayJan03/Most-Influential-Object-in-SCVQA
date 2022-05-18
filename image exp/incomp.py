#to visualise bbox
#not considering att
#understanding bbox

import json
import pickle
from itertools import chain
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import base64
import csv
import sys
import zlib
import time
import mmap
import pandas as pd


# image attention data
att_data = []
with open('/home/cvpr/jayant/scvqa/att_gv_100.pkl', 'rb') as fr:
    try:
        while True:
            att_data.append(pickle.load(fr))
    except EOFError:
        pass

att_data = att_data[0][0].cpu().detach().numpy()

# Question and answer data from mcan-small model
#with open('/Project/temp/result_run_epoch13.pkl_31907550.json', 'r') as f:
#    pred_data = json.load(f)

#pred_by_qid = {}
#for entry in pred_data:
#    pred_by_qid[entry['question_id']] = entry['answer']

# vqa-v2 dataset questions
with open('/home/cvpr/jayant/scvqa/data/v2_OpenEnded_mscoco_train2014_questions.json') as fd:
    ques_data = json.load(fd)

ques_data = ques_data['questions']
question_image_by_id = {}
for entry in ques_data:
    question_image_by_id[entry['question_id']] = (entry['image_id'], entry['question'])

qix = {}
for i in range(len(ques_data)):
    qix[ques_data[i]['question_id']] = i

# subset of questions chosen for evaluation
with open('/home/cvpr/jayant/scvqa/qid_100.pkl', 'rb') as f:
    qids1 = pickle.load(f)

#qids = []

#for ques_id1 in qids1:
#  qids.append(int(ques_id1))

#qids = list(chain.from_iterable(qids))



csv.field_size_limit(sys.maxsize)
   
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
infile = '/home/cvpr/jayant/trainval_36/rcnn_36.tsv'

"""
# Verify we can read a tsv
in_data = {}
with open(infile, "r") as tsv_in_file:
    reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
    for item in reader:
        item['image_id'] = int(item['image_id'])
        item['image_h'] = int(item['image_h'])
        item['image_w'] = int(item['image_w'])   
        item['num_boxes'] = int(item['num_boxes'])

#        item['num_boxes'] = int(item['num_boxes'])

#        for field in ['boxes', 'features']:
#            item[field] = np.frombuffer(base64.decodestring(item[field]), 
#                  dtype=np.float32).reshape((item['num_boxes'],-1))
        in_data[item['image_id']] = item
        break

print(in_data.keys())
"""

#tsv_data = pd.read_csv(infile, sep='\t')
df = pd.DataFrame()
oo = 1000
for chunk in pd.read_csv(infile, sep='\t', chunksize=1000):
    df = pd.concat([df, chunk], ignore_index=True)
    print(oo, "done!")
    oo +=1000

ii = 0

#for ques_id in tqdm(qids):
for ques_id in qids1:

    # use ques_id to retrieve img from dataset.
    ques_index = qix[ques_id]

    print("ques_index",ques_index)

    #att_index_i = ques_index // 32						#why?? to see if this is relevant to my model
    #att_index_j = ques_index % 32
    #att = att_data[att_index_i][att_index_j]
    image_prefix = "000000000000"  						# general format of validation set images
    image_path = "/home/cvpr/jayant/coco/images/train2017/"
    #image_features_path = '/Project/val2014/'
    image_id = question_image_by_id[ques_id][0]  # replace last n characters of image_prefix with this
    image_prefix = image_prefix[:-(len(str(image_id)))]
    final_img_name = image_path + image_prefix + str(image_id) + ".jpg"
    #final_img_feat_name = image_features_path + image_prefix + str(image_id) + ".jpg.npz"
    question = question_image_by_id[ques_id][1]

    for jj in in_data:
        if jj == image_id:
            print("match in tsv found")
            break

#here need to change so that bbox info is obtained from faster rcnn file 

    img_feat = in_data[jj]
    bboxes = img_feat['boxes']

    plt.rcParams['figure.figsize'] = (8.0, 8.0)
    f, ax = plt.subplots()
    plt.suptitle("sc vqa::{}".format(question))			#made a change
    im = cv2.imread(final_img_name)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.axis('off')
    gca = ax
    gca.axis('off')
    #att = att.flatten()
    shape = (img.shape[0], img.shape[1], 1)
    #A = np.zeros(shape)
    #for k in range(len(bboxes)):
    #    bbox = bboxes[k].astype(int)
    #    A[bbox[1]:bbox[3], bbox[0]:bbox[2]] += att[k]
    #A /= np.max(A)
    #A = A * img + (1.0 - A) * 255
    #A = A.astype('uint8')

#currently considering only most inf.. can plot multiple boxes too

    bbox = bboxes[np.argmax(att_data[ii])]
    print(bbox)
    print(len(bbox))
    ii+=1
    #gca.add_patch(plt.Rectangle((bbox[0], bbox[1]),
    #                            bbox[2] - bbox[0],
    #                            bbox[3] - bbox[1], fill=False,
    #                            edgecolor='red', linewidth=1))
    #gca.imshow(A, interpolation='bicubic')
    #gca.axis('off')
    #plt.tight_layout()
    #plt.savefig("{}.jpg".format(ques_id))





if __name__ == '__main__':

    # Verify we can read a tsv
    in_data = {}
    iii = 0
    with open(infile, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            item['image_id'] = int(item['image_id'])
            item['image_h'] = int(item['image_h'])
            item['image_w'] = int(item['image_w'])   
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(item[field]), 
                      dtype=np.float32).reshape((item['num_boxes'],-1))
            in_data[item['image_id']] = item
            print(iii)
            iii+=1
            #break
    print(in_data.keys())
    #print(in_data[1])
