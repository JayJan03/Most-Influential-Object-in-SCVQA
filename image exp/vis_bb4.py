#considering train2014

#discovery: this read file works when not in environment.. don't know why
#jugaad: pickle out box data separately.. not worked.. so try csv.. prob.. has \n in arrays
#try json..fails
#try npy.. error
#better not to dump, just load this every time

#!/usr/bin/env python


import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import pickle
import json

csv.field_size_limit(sys.maxsize)
   
FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
infile = '/home/cvpr/jayant/trainval_36/rcnn_36.tsv'




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
            in_data[item['image_id']] = item['boxes']			#just considering boxes
            if iii%1000 == 0:
                print(iii)
            iii+=1
            #break
    #print(in_data)

    #with open('boxes36.pickle', 'wb') as handle:
    #    pickle.dump(in_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

# open file for writing, "w" is writing
    #w = csv.writer(open("boxes36.csv", "w"))

# loop over dictionary keys and values
    #for key, val in in_data.items():
    # write every key and value to file
    #    w.writerow([key, val])



# Serialize data into file:
    #json.dump( in_data, open( "boxes36.json", 'w' ) )
    #print("done..!")

    #np.save('boxes36.npy',  in_data)


import json
import pickle
from itertools import chain
import cv2
import matplotlib.pyplot as plt
import numpy as np
#from tqdm import tqdm

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
    qids = pickle.load(f)
#qids = list(chain.from_iterable(qids))

ii = 0

for ques_id in qids:
#for ques_id in tqdm(qids):
    # use ques_id to retrieve img from dataset.
    ques_index = qix[ques_id]
    #att_index_i = ques_index // 32
    #att_index_j = ques_index % 32
    att = att_data[ii]
    ii+=1
    image_prefix = "000000000000"  # general format of validation set images
    image_prefix1 = "COCO_train2014_000000000000"  # general format of validation set images
    image_path = "/home/cvpr/jayant/coco/images/train2017/"
    image_features_path = '/home/cvpr/jayant/train2014/train2014/'
    image_id = question_image_by_id[ques_id][0]  # replace last n characters of image_prefix with this
    image_prefix = image_prefix[:-(len(str(image_id)))]
    image_prefix1 = image_prefix1[:-(len(str(image_id)))]

    final_img_name = image_path + image_prefix + str(image_id) + ".jpg"
    final_img_feat_name = image_features_path + image_prefix1 + str(image_id) + ".jpg.npz"
    question = question_image_by_id[ques_id][1]
    img_feat = np.load(final_img_feat_name)
    #bboxes = img_feat['bbox']
    bboxes = in_data[image_id]			#only change made in vis_bb3


    plt.rcParams['figure.figsize'] = (8.0, 8.0)
    f, ax = plt.subplots()
    plt.suptitle("sc-vqa::{}".format(question))
    im = cv2.imread(final_img_name)
    img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.axis('off')
    gca = ax
    gca.axis('off')
    att = att.flatten()
    shape = (img.shape[0], img.shape[1], 1)
    #A = np.zeros(shape)
    #for k in range(len(bboxes)):
    #    bbox = bboxes[k].astype(int)
    #    A[bbox[1]:bbox[3], bbox[0]:bbox[2]] += att[k]
    #A /= np.max(A)
    #A = A * img + (1.0 - A) * 255
    #A = A.astype('uint8')
    bbox = bboxes[np.argmax(att)]
    gca.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                                bbox[2] - bbox[0],
                                bbox[3] - bbox[1], fill=False,
                                edgecolor='red', linewidth=1))
    #gca.imshow(A, interpolation='bicubic')
    gca.axis('off')
    plt.tight_layout()
    plt.savefig("{}.jpg".format(ques_id))
    print("ITERATION DONE")