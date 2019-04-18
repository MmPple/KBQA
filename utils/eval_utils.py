# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
import sys
import json
import re
import shutil
import torch
import torch.nn as nn
from torch.autograd import Variable
from data_provider import VQADataProvider
from PIL import Image
from PIL import ImageFont, ImageDraw
sys.path.append("..")
import config
import mfh_baseline

def exec_validation(model, opt, mode, folder, it, visualize=False):
    model.eval()
    criterion = nn.NLLLosss()
    dp = VQADataProvider(opt, batchsize=opt.VAL_BATCH_SIZE, mode='val', folder=folder)
    epoch = 0
    pred_list = []
    testloss_list = []
    stat_list = []
    total_questions = len(dp.getQuesIds())
    print 'Validating...'
    
    
    while epoch == 0:
        data, t_glove_q, t_relation, t_glove_r, t_frelation, t_rvec_set, t_glove_rvec_set,  t_qid_list, epoch = dp.get_batch_vec()
        
        data = Variable(torch.from_numpy(data).cuda())
        glove_q = Variable(torch.from_numpy(t_glove_q)).cuda()
        relation = Variable(torch.from_numpy(t_relation)).cuda()
        glove_r = Variable(torch.from_numpy(t_glove_r)).cuda()
        frelation = Variable(torch.from_numpy(t_frelation)).cuda()
        glove_fr = Variable(torch.from_numpy(t_glove_fr)).cuda()
        
        label = relation
        
        pred = model(, 'val')
        pred = (pred.data).cpu().numpy()
        if mode == 'test-dev' or 'test':
            pass
        else:
            loss = criterion(pred, label.long())
            loss = (loss.data).cpu().numpy()
            testloss_list.append(loss)
        t_pred_list = np.argmax(pred, axis = 1)
        t_pred_str = [dp.vec_to_answer(pred_symbol) for pred_symbol in t_pred_list]
        
        print '--------------------------',pred.size,'----------------------------------------',pred.shape
        for qid, iid, ans, pred in zip(t_qid_list, t_iid_list, t_answer.tolist(), t_pred_str):
            pred_list.append((pred,int(dp.getStrippedQuesId(qid))))
            if visualize:
                q_list = dp.seq_to_list(dp.getQuesStr(qid))
                if mode == 'test-dev' or 'test':
                    ans_str = ''
                    ans_list = ['']*10
                else:
                    ans_str = dp.vec_to_answer(ans)
                    ans_list = [ dp.getAnsObj(qid)[i]['answer'] for i in range(10)]
                stat_list.append({\
                                    'qid'   : qid,
                                    'q_list' : q_list,
                                    'iid'   : iid,
                                    'answer': ans_str,
                                    'ans_list': ans_list,
                                    'pred'  : pred })
        percent = 100 * float(len(pred_list)) / total_questions
        sys.stdout.write('\r' + ('%.2f' % percent) + '%')
        sys.stdout.flush()
        
    # 剔除重复元素
    print ('Deduping arr of len', len(pred_list))
    deduped = []
    seen = set()
    for ans, qid in pred_list:
        if qid not in seen:
            seen.add(qid)
            deduped.append((ans, qid))
    print ('New len', len(deduped))

    # 生成预测答案文件，final_list就是答案list
    final_list=[]
    for ans,qid in deduped:
        final_list.append({u'answer': ans, u'question_id': qid})

    if mode == 'val': # 验证的数据集为val时，先将答案文件用json文件格式存储下来，然后使用VQAEval工具来生生成一个基于验证集的分析结果
        mean_testloss = np.array(testloss_list).mean()
        valFile = './%s/val2015_resfile'%folder
        with open(valFile, 'w') as f:
            json.dump(final_list, f)
        #if visualize:
        #    visualize_failures(stat_list,mode)
        annFile = config.DATA_PATHS['val']['ans_file']
        quesFile = config.DATA_PATHS['val']['ques_file']
        vqa = VQA(annFile, quesFile)
        vqaRes = vqa.loadRes(valFile, quesFile)
        vqaEval = VQAEval(vqa, vqaRes, n=2)
        vqaEval.evaluate()
        acc_overall = vqaEval.accuracy['overall']
        acc_perQuestionType = vqaEval.accuracy['perQuestionType']
        acc_perAnswerType = vqaEval.accuracy['perAnswerType']
        return mean_testloss, acc_overall, acc_perQuestionType, acc_perAnswerType
    elif mode == 'test-dev': # 为test-dev数据集时直接将答案文件存储下来，因为本来就没有给标准答案，所以就不进行结果分析了
        filename = './%s/vqa_OpenEnded_mscoco_test-dev2015_%s-'%(folder,folder)+str(it).zfill(8)+'_results'
        with open(filename+'.json', 'w') as f:
            json.dump(final_list, f)
        #if visualize:
        #    visualize_failures(stat_list,mode)
    elif mode == 'test': # 为test数据集时直接将答案文件存储下来，因为本来就没有给标准答案，所以就不进行结果分析了
        filename = './%s/vqa_OpenEnded_mscoco_test2015_%s-'%(folder,folder)+str(it).zfill(8)+'_results'
        with open(filename+'.json', 'w') as f:
            json.dump(final_list, f)
        #if visualize:
        #    visualize_failures(stat_list,mode)