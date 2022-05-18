#CF FOR ALL OF VQX.. no hint


#evaluate for understanding att_gv in model

#evaluate for dataset with entriess[0:100]
#dump imgid, qid, qtype, match info


import sys
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.init as init
import numpy as np
import pickle
from dataset import Dictionary, SelfCriticalDataset
from tqdm import tqdm
from models import Model
import utils
import opts

def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def compute_score_with_k_logits(logits, labels, k=5):
    logits = torch.sort(logits, 1)[1].data  # argmax
    scores = torch.zeros((labels.size(0), k))

    for i in range(k):
        one_hots = torch.zeros(*labels.size()).cuda()
        one_hots.scatter_(1, logits[:, -i - 1].view(-1, 1), 1)
        scores[:, i] = (one_hots * labels).squeeze().sum(1)
    scores = scores.max(1)[0]
    return scores

def evaluate(model, dataloader):
    score = 0
    scorek = 0
    score1 = 0
    V_loss = 0
    V_loss1 = 0
    qid2type = pickle.load(open('qid2type.pkl', 'rb'))

    upper_bound = 0
    num_data = 0
    score_yesno = 0
    score_number = 0
    score_other = 0
    total_yesno = 0
    total_number = 0
    total_other = 0
    dict = {}
    att = {}
    mostinfind = {}
    cnt = 0
    cnt1 = 0
    cnt2 = 0
    qid2loss = {}
    cnt3 = 0
    sum_sample = 0
    #opt.num_sub = 36								#note

    for objs, q, a, hintscore, _, qids in tqdm(iter(dataloader)):
        #print("********* NEXT ITERATION **************")
        objs = objs.cuda().float().requires_grad_()
        q = q.cuda().long()
        a = a.cuda()  # true labels
        hintscore = hintscore.cuda().float()

        pred, _, ansidx = model(q, objs)
        #att[cnt1] = att_gv
        #cnt1+=1

        loss_vqa = instance_bce_with_logits(pred, a)
        vqa_grad = torch.autograd.grad((pred * (a > 0).float()).sum(), objs, create_graph=True)[0]  # [b , 80, 2048]
        vqa_grad_cam = vqa_grad.sum(2)
        aidx = a.argmax(1).detach().cpu().numpy().reshape((-1))

        loss_hint = torch.zeros((vqa_grad_cam.size(0), opt.num_sub, 36)).cuda()
        hintscore = hintscore.squeeze()
        hint_sort, hint_ind = hintscore.sort(1, descending=True)

        thresh = hint_sort[:, opt.num_sub:opt.num_sub + 1] - 0.00001
        thresh += ((thresh < 0.2).float() * 0.1)
        #hintscore = (hintscore > thresh).float()

        for j in range(opt.num_sub):
        #for j in range(36):
            for k in range(36):
                if j == k:
                    continue
                hint1 = hintscore.gather(1, hint_ind[:, j:j + 1]).squeeze()
                hint2 = hintscore.gather(1, hint_ind[:, k:k + 1]).squeeze()

                vqa1 = vqa_grad_cam.gather(1, hint_ind[:, j:j + 1]).squeeze()
                vqa2 = vqa_grad_cam.gather(1, hint_ind[:, k:k + 1]).squeeze()
                if j < k:
                    #mask = ((hint1 - hint2) * (vqa1 - vqa2 - 0.0001) < 0).float()
                    mask = ((vqa1 - vqa2 - 0.0001) < 0).float()
                    loss_hint[:, j, k] = torch.abs(vqa1 - vqa2 - 0.0001) * mask
                else:
                    #mask = ((hint2 - hint1) * (vqa2 - vqa1 - 0.0001) < 0).float()
                    mask = ((vqa2 - vqa1 - 0.0001) < 0).float()
                    loss_hint[:, j, k] = torch.abs(vqa2 - vqa1 - 0.0001) * mask

        loss_hint *= opt.hint_loss_weight
        loss_hint = loss_hint.sum(2)  # b num_sub
        eps = 1e-6
        loss_hint += ( ((loss_hint.sum(1).unsqueeze(1) > eps).float() * (loss_hint < eps).float() ) * 10000)

        #loss_hint, loss_hint_ind =  loss_hint.min(1) # loss_hint_ind b

        num_mio = 5							#num of bbs to be noised out

        loss_hint, loss_hint_ind =  torch.topk(loss_hint,num_mio,largest=False)  # loss_hint_ind b


        objs = objs.cpu().detach().numpy()

        sum_sample+=len(objs)

        infind = []
        #att100 = pickle.load(open('/home/cvpr/jayant/scvqa/qid2mio_top5.pkl', 'rb'))
        listqid = qids.tolist()
        listind = loss_hint_ind.tolist()
        listbb = hint_ind.tolist()						#not req cause not using object detector here

        qids = qids.detach().cpu().int().numpy()

        #for jjj in qids:						#to align order of indices to order of qids here
            
        #    index = listqid.index(jjj)
        #    infind.append(listind[index])

        #print("listqid:::::",listqid)

        #print("qids:::::",qids)

        #print("infind:::::",infind)

        dictt = []
        for i in range(len(listind)):
            entry = []
            for j in listind[i]:
                entry.append(listbb[i][j])
            dictt.append(entry)



        for jj in range(len(objs)):
            #print(qids[jj])
            for ll in range(num_mio):
                for kk in range(2048):
            
                    objs[jj][dictt[jj][ll]][kk] = np.random.rand(1)		#counterfactual
                #print(dictt[jj][ll])		#counterfactual
            #print(objs[jj])
            #print("***********")
        #print("objs_cf:::::",objs)


        objs = torch.tensor(objs, requires_grad=True, device = torch.device('cuda:0'))

        #print("********* NEXT ITERATION **************")
        objs = objs.cuda().float().requires_grad_()


        pred, _, ansidx = model(q, objs)					#prediction after cf


        #hint_ind.tolist()
        #loss_hint_ind.tolist()
        #print("qids ::::::: ", qids)

        #print("hint_ind::::::: ",hint_ind)
        #print(len(qids))
        #print(len(hint_ind))
        #print(len(loss_hint_ind))

        #qid2loss[cnt3] = [qids,loss_hint_ind,hint_ind]
        #cnt3+=1

        #aa = 0
        #for i in range(len(loss_hint_ind)):
        #    aa = hint_ind[i][loss_hint_ind[i]].item()
        #    mostinfind[cnt2] = aa
        #    cnt2+=1

        #loss = instance_bce_with_logits(pred, a)

        #V_loss += loss.item() * objs.size(0)
        batch_score = compute_score_with_logits(pred, a.data).cpu().numpy().sum(1)
        score += batch_score.sum()

        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)
        #qids = qids.detach().cpu().int().numpy()

        for j in range(len(qids)):
            qid = qids[j]
            typ = qid2type[qid]
            if typ == 'yes/no':
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')


            #dict[cnt] = new_ent

            #cnt+=1


    score = score / len(dataloader.dataset)
    #V_loss /= len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    score_number /= total_number

    #pickle.dump(att, open('att_gv_100.pkl', 'wb'))
    #pickle.dump(mostinfind, open('most_inf_ind.pkl', 'wb'))
    #pickle.dump(loss_hint_ind, open('loss_hint_ind.pkl', 'wb'))
    #pickle.dump(hint_ind, open('hint_ind.pkl', 'wb'))
    #pickle.dump(qid2loss, open('qid2mio_top5_nh_vqx_exp.pkl', 'wb'))

    print("No. of test samples::::   ",sum_sample)

    return score, score_yesno, score_other, score_number


if __name__ == '__main__':
    opt = opts.parse_opt()
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    opt.ntokens = dictionary.ntoken
    model = Model(opt)

    model = model.cuda()
    model = nn.DataParallel(model).cuda()
    # model = model.cuda()

    opt.use_all = 1
    eval_dset = SelfCriticalDataset('v2cp_test_vqx', dictionary, opt)
    eval_loader = DataLoader(eval_dset, opt.batch_size, shuffle=False, num_workers=0)

    states_ = torch.load('saved_models/%s/model-best.pth'%opt.load_model_states)
    states = model.state_dict()
    for k in states_.keys():
        if k in states:
            states[k] = states_[k]
            print('copying  %s' % k)
        else:
            print('ignoring  %s' % k)
    model.load_state_dict(states)
    model.eval()
    score, score_yesno, score_other, score_number = evaluate(model, eval_loader)
    print('Overall: %.3f\n' % score)
    print('Yes/No: %.3f\n' % score_yesno)
    print('Number: %.3f\n' % score_number)
    print('Other: %.3f\n' % score_other)

