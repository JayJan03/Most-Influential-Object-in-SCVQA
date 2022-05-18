#for train subset

#considering train2014
#with a known to work pickled boxes file
#for top5 boxes

import json
import pickle
from itertools import chain
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# image attention data
att_data = []
with open('/home/cvpr/jayant/scvqa/att_gv_100_vqx.pkl', 'rb') as fr:
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
with open('/home/cvpr/jayant/scvqa/qid_100_vqx.pkl', 'rb') as f:
    qids = pickle.load(f)
#qids = list(chain.from_iterable(qids))

ff = open("/home/cvpr/jayant/scvqa/vis_img/boxes36_2.pkl", "rb")
u = pickle._Unpickler( ff )
u.encoding = 'latin1'
loadboxes = u.load()

ff1 = open("/home/cvpr/jayant/scvqa/vis_img/top5_100_vqx.pkl", "rb")
u1 = pickle._Unpickler( ff1 )
u1.encoding = 'latin1'
top5 = u1.load()

#ff2 = open("/home/cvpr/jayant/scvqa/vis_img/top5cf_100_train.pkl", "rb")
#u2 = pickle._Unpickler( ff2 )
#u2.encoding = 'latin1'
#top5cf = u2.load()

ffa = open("/home/cvpr/jayant/scvqa/vis_img/qid2mio_vqx.pkl", "rb")
ua = pickle._Unpickler( ffa )
ua.encoding = 'latin1'
mio = ua.load()


ii = 0

for ques_id in qids:
#for ques_id in tqdm(qids):
    # use ques_id to retrieve img from dataset.

    if ques_id in qix.keys():

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

    #if ques_id in question_image_by_id.keys():
        question = question_image_by_id[ques_id][1]
        img_feat = np.load(final_img_feat_name)
    #bboxes = img_feat['bbox']
        bboxes = loadboxes[image_id]

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
        collist = ['red','orange','yellow','green','blue']
        for kk in range(1):
            bbox = bboxes[mio[ques_id]]
            gca.add_patch(plt.Rectangle((bbox[0], bbox[1]),
                                        bbox[2] - bbox[0],
                                        bbox[3] - bbox[1], fill=False,
                                        edgecolor=collist[kk], linewidth=2))
    #gca.imshow(A, interpolation='bicubic')
        gca.axis('off')
        plt.tight_layout()
        plt.savefig("{}.jpg".format(ques_id))