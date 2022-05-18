#evaluate for counterfactual analysis: first replaces feature vector of most influential bb, then dumps attention of model evaluation + prediction data


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
from dataset import Dictionary, SelfCriticalDataset1
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

    for objs, q, a, hintscore, _, qids,imgid in tqdm(iter(dataloader)):

        objs = objs.cpu().detach().numpy()
        qids = qids.detach().cpu().int().numpy()

        infind = []
        att100 = pickle.load(open('/home/cvpr/jayant/scvqa/qid2mio_top5.pkl', 'rb'))
        listqid = att100[0][0].tolist()
        listind = att100[0][1].tolist()


        for jjj in qids:						#to align order of indices to order of qids here
            
            index = listqid.index(jjj)
            infind.append(listind[index])

        num_mio = 5							#num of bbs to be noised out

        for ll in range(36):
            for jj in range(len(objs)):
                for kk in range(2048):            
                    objs[jj][ll][kk] = np.random.rand(1)		#0 img

        objs = torch.tensor(objs, requires_grad=True, device = torch.device('cuda:0'))

        print("********* NEXT ITERATION **************")
        objs = objs.cuda().float().requires_grad_()
        q = q.cuda().long()
        a = a.cuda()  # true labels
        hintscore = hintscore.cuda().float()

        pred, att_gv, ansidx = model(q, objs)
        att[cnt1] = att_gv
        cnt1+=1

        loss_vqa = instance_bce_with_logits(pred, a)
        vqa_grad = torch.autograd.grad((pred * (a > 0).float()).sum(), objs, create_graph=True)[0]  # [b , 80, 2048]
        vqa_grad_cam = vqa_grad.sum(2)
        aidx = a.argmax(1).detach().cpu().numpy().reshape((-1))

        loss_hint = torch.zeros((vqa_grad_cam.size(0), opt.num_sub, 36)).cuda()
        hintscore = hintscore.squeeze()
        hint_sort, hint_ind = hintscore.sort(1, descending=True)

        thresh = hint_sort[:, opt.num_sub:opt.num_sub + 1] - 0.00001
        thresh += ((thresh < 0.2).float() * 0.1)
        hintscore = (hintscore > thresh).float()

        for j in range(opt.num_sub):
            for k in range(36):
                if j == k:
                    continue
                hint1 = hintscore.gather(1, hint_ind[:, j:j + 1]).squeeze()
                hint2 = hintscore.gather(1, hint_ind[:, k:k + 1]).squeeze()

                vqa1 = vqa_grad_cam.gather(1, hint_ind[:, j:j + 1]).squeeze()
                vqa2 = vqa_grad_cam.gather(1, hint_ind[:, k:k + 1]).squeeze()
                if j < k:
                    mask = ((hint1 - hint2) * (vqa1 - vqa2 - 0.0001) < 0).float()
                    loss_hint[:, j, k] = torch.abs(vqa1 - vqa2 - 0.0001) * mask
                else:
                    mask = ((hint2 - hint1) * (vqa2 - vqa1 - 0.0001) < 0).float()
                    loss_hint[:, j, k] = torch.abs(vqa2 - vqa1 - 0.0001) * mask

        loss_hint *= opt.hint_loss_weight
        loss_hint = loss_hint.sum(2)  # b num_sub
        eps = 1e-6
        loss_hint += ( ((loss_hint.sum(1).unsqueeze(1) > eps).float() * (loss_hint < eps).float() ) * 10000)

        loss_hint, loss_hint_ind =  loss_hint.min(1) # loss_hint_ind b



        #hint_ind.tolist()
        #loss_hint_ind.tolist()

        aa = 0
        for i in range(len(loss_hint_ind)):
            aa = hint_ind[i][loss_hint_ind[i]].item()
            mostinfind[cnt2] = aa
            cnt2+=1

        #loss = instance_bce_with_logits(pred, a)

        #V_loss += loss.item() * objs.size(0)
        batch_score = compute_score_with_logits(pred, a.data).cpu().numpy().sum(1)
        score += batch_score.sum()

        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)

        for j in range(len(qids)):
            qid = qids[j]
            #print(qid)
            typ = qid2type[qid]
            if typ == 'yes/no':
                #print(batch_score[j])
                if batch_score[j] >= 0.5:
                    new_ent = { 'image_id': imgid[j], 'qid': qid, 'qtype': typ, 'match': "yes"}
                else:
                    new_ent = { 'image_id': imgid[j], 'qid': qid, 'qtype': typ, 'match': "no"}
                score_yesno += batch_score[j]
                total_yesno += 1
            elif typ == 'other':
                #print(batch_score[j])
                if batch_score[j] >= 0.5:
                    new_ent = { 'image_id': imgid[j], 'qid': qid, 'qtype': typ, 'match': "yes"}
                else:
                    new_ent = { 'image_id': imgid[j], 'qid': qid, 'qtype': typ, 'match': "no"}
                score_other += batch_score[j]
                total_other += 1
            elif typ == 'number':
                #print(batch_score[j])
                if batch_score[j] >= 0.5:
                    new_ent = { 'image_id': imgid[j], 'qid': qid, 'qtype': typ, 'match': "yes"}
                else:
                    new_ent = { 'image_id': imgid[j], 'qid': qid, 'qtype': typ, 'match': "no"}
                score_number += batch_score[j]
                total_number += 1
            else:
                print('Hahahahahahahahahahaha')


            dict[cnt] = new_ent

            cnt+=1


    score = score / len(dataloader.dataset)
    V_loss /= len(dataloader.dataset)
    score_yesno /= total_yesno
    score_other /= total_other
    try:
        score_number /= total_number
    except:
        score_number = total_number

    #pickle.dump(dict, open('eval_100_cf_vqx_{nn}.pkl'.format(nn = num_mio), 'wb'))

    #pickle.dump(att, open('att_gv_100_cf_vqx_{n}.pkl'.format(n = num_mio), 'wb'))
    #pickle.dump(mostinfind, open('most_inf_ind.pkl', 'wb'))
    #pickle.dump(loss_hint_ind, open('loss_hint_ind.pkl', 'wb'))
    #pickle.dump(hint_ind, open('hint_ind.pkl', 'wb'))

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
    eval_dset = SelfCriticalDataset1('v2cp_test_vqx', dictionary, opt)
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

