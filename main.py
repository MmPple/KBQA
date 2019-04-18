# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
import sys
import config
import mfh_baseline
import utils.data_provider as data_provider
from utils.data_provider import VQADataProvider
import json
import datetime
# from tensorboardX import SummaryWriter 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def make_relation_vocab(rdic):
    """
    Returns a dictionary that maps words to indices.得到一个关系字典，将关系中的单词与数字序号对应起来
    """
    rdict = {'' : 0}
    rid = 1
    #for qid in rdic.keys():
        # sequence to list 先将关系字符串转为一个字符串 list
    relation_list = rdic['all_relation']
    r_list = []
    for relation in relation_list:
        re = relation.split('/')[-2:]
        r_str_list = re[0].split('_') 
        r_list.extend(r_str_list)
        r_str_list = re[1].split('_') 
        r_list.extend(r_str_list)
        
    # create dict  将关系中所包含的所有的单词存在一个 字典当中
    for w in r_list:
        if not rdict.has_key(w):
            rdict[w] = rid
            rid += 1
# debug        
#nalist = []
#for k, v in sorted(nadict.items(), key = lambda x:x[1]) :
#    nalist.append((k, v))

    # remove words that appear less than once 
    """
    n_del_ans = 0
    n_valid_ans = 0
    adict_nid = {} 
    for i, w in enumerate(nalist[:-vocab_size]): # 删除前3000个答案单词
        del adict[w[0]]
        n_del_ans += w[1]
    for i, w in enumerate(nalist[-vocab_size:]): # 将后3000个答案单词保存在adict_nid这个 list中
        n_valid_ans += w[1]
        adict_nid[w[0]] = i
    """
    return rdict # 返回一个包含了出现次数最多的3000个答案单词的字典 : {'a_word1' : 1, 'a_word2' : 2, ...,'a_word3000': 3000 }

def make_question_vocab(qdic):
    """
    Returns a dictionary that maps words to indices. 得到一个问题单词字典，将问题里面所出现过的所有的单词都记录在vdict这个字典当中
    """
    vdict = {'': 0}
    vid = 1
    for qid in qdic.keys():
        # sequence to list 先将问题字符串转为一个字符串 list
        q_str = qdic[qid]['qstr']
        #print (q_str)
        q_list = VQADataProvider.seq_to_list(q_str) 
        
        # create dict  将问题中所包含的所有的单词存在一个 字典当中
        for w in q_list:
            if not vdict.has_key(w):
                vdict[w] = vid
                vid += 1
                
    return vdict # 返回一个包含了所有问题当中出现的单词的字典：{'q_words1' : 1, 'q_words2' : 2, 'q_words3' : 3, ......}

def make_vocab_files():
    """
    Produce the question and answer vocabulary files. 调用上面的连个函数，生成问题字典和答案字典
    """
    print('making question vocab...', opt.QUESTION_VOCAB_SPACE)
    qdic_train, _ = VQADataProvider.load_data(opt.QUESTION_VOCAB_SPACE, 'mfh_baseline_train')# qdic = {data_split/Question_id : {'qstr' : Question_str} }
    print('making question vocab...', 'val')
    qdic_val, _ = VQADataProvider.load_data('val', 'mfh_baseline_val')
    print('making question vocab...', 'test')
    qdic_test, _ = VQADataProvider.load_data('test', 'mfh_baseline_test')
    qdic = qdic_train
    qdic.update(qdic_val)
    qdic.update(qdic_test)
    question_vocab = make_question_vocab(qdic)                                       # question_vocab = {'q_words' : q_words_numbers}
    
    print('making relation vocab...', opt.RELATION_VOCAB_SPACE)
    _, rdic_train = VQADataProvider.load_data(opt.RELATION_VOCAB_SPACE, 'mfh_baseline_train')
    print('making relation vocab...', 'val')
    _, rdic_val = VQADataProvider.load_data('val', 'mfh_baseline_val') 
    print('making relation vocab...', 'test')
    _, rdic_test = VQADataProvider.load_data('test', 'mfh_baseline_test') # rdic = {data_split/Question_id : {"right_relation" : rstr , "false_relation" : frstr ,"relation_candiate":[cand1, cand2, cand3, ..., candn]}
    rdic = rdic_train
    rdic.update(rdic_val)
    rdic.update(rdic_test)
    relation_vocab = make_relation_vocab(rdic)   # relation_vocab = {'r_words' : r_words_numbers}
    return question_vocab, relation_vocab # 返回问题单词字典和答案单词字典

opt = config.parse_opt()
# writer = SummaryWriter()
folder = 'mfh_baseline_%s'%opt.TRAIN_DATA_SPLITS
if not os.path.exists('./%s'%folder):
    os.makedirs('./%s'%folder)
question_vocab, relation_vocab = {}, {}
if os.path.exists('./%s/vdict.json'%folder) and os.path.exists('./%s/rdict.json'%folder):
    print ('restoring vocab')
    with open('./%s/vdict.json'%folder, 'r') as f:
        question_vocab = json.load(f)
    with open('./%s/rdict.json'%folder, 'r') as f:
        relation_vocab = json.load(f)
else:
    question_vocab, relation_vocab = make_vocab_files()
    with open('./%s/vdict.json'%folder, 'w') as f:
        json.dump(question_vocab, f)
    with open('./%s/rdict.json'%folder, 'w') as f:
        json.dump(relation_vocab, f)
print ('question vocab size : ', len(question_vocab))
print ('relation vocab size :', len(relation_vocab))
opt.quest_vob_size = len(question_vocab)
opt.rela_vob_size = len(relation_vocab)
print ('------------------data-provider works well------------------------------')

mfh_baseline.train(opt, folder)
#writer.close()
    
