# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import sys
import os
import json
import re
import datetime
sys.path.append("..")
import shutil
import config
import mfh_baseline
import utils.data_provider as data_provider
from utils.data_provider import VQADataProvider
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.backends.cudnn.enabled = False

number = 50000
PATH = str(sys.argv[1])

opt = config.parse_opt()
def compute_cosine_sim(input1, input2):
    # input1, input2--混合向量和relation向量
    CosineSim_Tensor = F.cosine_similarity(input1, input2)
    return CosineSim_Tensor#, len(CosineSim_Tensor)
    # 计算余弦相似度，返回Tensor和计算个数

def rank_candidate_relation(score_tensor):
    #score_tensor--得分向量
    ranked_tensor, rank = score_tensor.sort(descending=True)
    golden_relation_index = rank[0]
    return golden_relation_index, ranked_tensor[0]
    # 对每个问题的relation降序排序，去得分最高
    #返回 得分最高的relation的索引

def compare_with_label(golden_relation_index ,relation_list, label):
    #golden_relation_index--得分最高的relation索引
    if relation_list[golden_relation_index] == label:
        return 1
    else:
        return 0
    # 判断预测是否正确，返回1 or 0

def compute_accuracy(mode, result_list):
    # result_list = [1,0,1,1,0,...]
    right_number = sum(result_list)
    accuracy = float(right_number) / float(len(result_list)) * 100 
    print ('Accuracy of ' + mode + 'is %f' %accuracy)
    # 计算精确度

    
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

folder = 'mfh_baseline_%s'%opt.TRAIN_DATA_SPLITS
if not os.path.exists('%s'%folder):
    os.makedirs('%s'%folder)
question_vocab, relation_vocab = {}, {}
if os.path.exists('%s/vdict.json'%folder) and os.path.exists('%s/rdict.json'%folder):
    print ('restoring vocab')
    with open('%s/vdict.json'%folder, 'r') as f:
        question_vocab = json.load(f)
    with open('%s/rdict.json'%folder, 'r') as f:
        relation_vocab = json.load(f)
else:
    question_vocab, relation_vocab = make_vocab_files()
    with open('%s/vdict.json'%folder, 'w') as f:
        json.dump(question_vocab, f)
    with open('%s/rdict.json'%folder, 'w') as f:
        json.dump(relation_vocab, f)
print ('question vocab size : ', len(question_vocab))
print ('relation vocab size :', len(relation_vocab))
opt.quest_vob_size = len(question_vocab)
opt.rela_vob_size = len(relation_vocab)
print ('------------------data-provider works well------------------------------')
class mfh_baseline(nn.Module):
    def __init__(self, opt):
        super(mfh_baseline, self).__init__()
        self.opt = opt
        self.JOINT_EMB_SIZE = 2*opt.LSTM_UNIT_NUM
        # self.Embedding1 = nn.Embedding(opt.quest_vob_size, 300)
        # self.Embedding2 = nn.Embedding(opt.rela_vob_size, 300)
        self.LSTM1 = nn.LSTM(input_size=300, hidden_size=opt.LSTM_UNIT_NUM, num_layers=1, batch_first=False, bidirectional=True)
        self.LSTM2 = nn.LSTM(input_size=300, hidden_size=opt.LSTM_UNIT_NUM, num_layers=1, batch_first=False, bidirectional=True)
        self.LSTM3 = nn.LSTM(input_size=300, hidden_size=opt.LSTM_UNIT_NUM, num_layers=1, batch_first=False, bidirectional=True)
        self.LSTM4 = nn.LSTM(input_size=300, hidden_size=opt.LSTM_UNIT_NUM, num_layers=1, batch_first=False, bidirectional=True)
        self.linear = nn.Linear(1, 1, bias=True)
        
        
    def forward(self, data, data_glove, rb_vec, rb_glove, wrb_vec, wrb_glove, rc_vec, rc_glove, wrc_vec, wrc_glove, mode):
    ## def forward(self, data, data_glove, s_vec, s_glove, mode):
        if mode == 'val' or mode == 'test' :
            self.batch_size = self.opt.VAL_BATCH_SIZE
        else:
            self.batch_size = self.opt.BATCH_SIZE
        # data = torch.transpose(data, 1, 0).long()
        #print (data.shape) # 33*1
        
        data_glove = data_glove.permute(1, 0, 2)
        #print (data_glove.shape) # 33*1*300
        # data = F.tanh(self.Embedding1(data))
        #print (data.shape) # 33*1*300
        # data = torch.cat((data, data_glove) , 2)   # 33 * batch * 600
        #print (data.shape) #33*1*600
        data_1_lstm, _ = self.LSTM1(data_glove)
        #H_q = self.relu(data_1_lstm)
        ## H_q = H_q.permute(1,0,2) # batch * seq_len * 1024
        qb_feat, _= torch.max(data_1_lstm, 0)
        data_2_lstm, _ = self.LSTM2(data_glove)
        qc_feat, _= torch.max(data_2_lstm, 0)


        # rb = rb_vec
        bglove = rb_glove
        bglove = bglove.permute(1, 0, 2)
        # rb = torch.transpose(rb, 1, 0).long()
        # rb = F.tanh(self.Embedding2(rb))
        # rb = torch.cat((rb, bglove), 2) # 17*batch*600
        rb_feat, _ = self.LSTM3(bglove)
        rb_feat, _ = torch.max(rb_feat, 0)

        # rc = rc_vec
        cglove = rc_glove
        cglove = cglove.permute(1, 0, 2)
        # rc = torch.transpose(rc, 1, 0).long()
        # rc = F.tanh(self.Embedding2(rc))
        # rc = torch.cat((rc, cglove), 2) # 17*batch*600
        rc_feat, _ = self.LSTM4(cglove)
        rc_feat, _ = torch.max(rc_feat, 0)

        ab = self.linear(qb_feat.mul(rb_feat).sum(1).unsqueeze(1))
        ac = self.linear(qc_feat.mul(rc_feat).sum(1).unsqueeze(1))

        rb_feat = ab * rb_feat
        rc_feat = ac * rc_feat

        rb_score = F.cosine_similarity(qb_feat, rb_feat)
        rc_score = F.cosine_similarity(qc_feat, rc_feat)
        # print(rb_score.shape)
        
        # t_score = F.cosine_similarity(q_feat, t_feat)
        # r_score = r_score.view(-1, 1)
        # t_score = t_score.view(-1, 1)
        
        final_score = rb_score+rc_score
        
        if mode == "val" or mode == "test":
            return 1, final_score
        
        # print(final_score.shape)
        wrb = wrb_vec
        wbglove = wrb_glove
        wbglove = wbglove.permute(1, 0, 2)
        wrb = torch.transpose(wrb, 1, 0).long()
        wrb = F.tanh(self.Embedding2(wrb))
        wrb = torch.cat((wrb, wbglove), 2) # 17*batch*600
        wrb_feat, _ = self.LSTM3(wrb)
        wrb_feat, _ = torch.max(wrb_feat, 0)
        
        wrc = wrc_vec
        wcglove = wrc_glove
        wcglove = wcglove.permute(1, 0, 2)
        wrc = torch.transpose(wrc, 1, 0).long()
        wrc = F.tanh(self.Embedding2(wrc))
        wrc = torch.cat((wrc, wcglove), 2) # 17*batch*600
        wrc_feat, _ = self.LSTM4(wrc)
        wrc_feat, _ = torch.max(wrc_feat, 0)
        
        wrb_score = F.cosine_similarity(qb_feat, wrb_feat).unsqueeze(1)
        wrc_score = F.cosine_similarity(qc_feat, wrc_feat).unsqueeze(1)
        
        
        wfinal_score = self.linear(torch.cat((wrb_score, wrc_score), 1)).squeeze(1)
        # wfinal_score = torch.sigmoid(wr_score)
        
        return 1, final_score, wfinal_score

model = mfh_baseline(opt)
model.cuda()
model.eval()

#model = torch.load('./data/mfh_baseline_glove_iter50000.pth')
model.load_state_dict(torch.load('./{}/mfh_baseline_glove_iter'.format(opt.pth_path)+str(sys.argv[2])+'.pth'))
folder = "mfh_baseline_%s"%PATH
dp = VQADataProvider(opt, batchsize=opt.VAL_BATCH_SIZE, mode='%s'%PATH,  folder=folder)
epoch = 0
total_questions = len(dp.getQuesIds())
print ('Validating...')
#qdic = dp.qdic
#rdic = dp.rdic
#with open('./%s/qdic.json'%folder, 'w') as f:
#        json.dump(qdic, f)
#with open('./%s/rdic.json'%folder, 'w') as f:
#        json.dump(rdic, f)
result_list = []
count = 0
question_dic = dp.qdic
relation_dic = dp.rdic

try:
    fobj=open(opt.save_path+"/Fast_"+PATH+"_RelationDetection_result_"+str(sys.argv[2])+".txt",'a')
except IOError:
    print ('*** file open error:')


while epoch == 0 and count<number:
    count += 1
    if count ==1 or count%50 == 0:
        print ('%s : number of questions : [%d]'%(time.asctime(time.localtime(time.time())), count))
    
    
    t_question, t_q_glove, t_rset, t_glove_rset, t_qid_list, epoch = dp.get_batch_vec('%s'%PATH)
    # if count < 195: continue
     #读取问题标号向量，使用 Variable包装
    #读取问题glove向量，使用 Variab95000
    batch_rela_length = len(t_rset) #读取batch中的问题个数，存在变量batch_rela_length中
    #print (batch_rela_length)
    
    # print (t_rset, t_glove_rset.shape)
    for i in xrange(batch_rela_length):
        q_id = t_qid_list[i]   #遍历问题，一个问题一个问题地处理，对每一个问题求出最佳的 relation
        # print(q_id)
        fobj.write(str(q_id)+' %%%% ')
        label_entity = relation_dic[str(q_id)]['right_entity']  
        label_relation = relation_dic[str(q_id)]['right_relation']
        fobj.write(label_entity+" %%%% " + label_relation)
        
        score_list = []
        
        for k, ((result,rbvec_set,rcvec_set), (g_rbvec_set,g_rcvec_set)) in enumerate(zip(t_rset[i], t_glove_rset[i])):
            # (result, rvec_set) = er_tuple
            r_num = len(g_rbvec_set)
            if r_num > 150:
                r_num = 150
            # rb_vec = Variable(torch.from_numpy(rbvec_set).cuda()).long() #取出这个问题的所有relation向量
            # rc_vec = Variable(torch.from_numpy(rcvec_set).cuda()).long()
            #rela_len = rela_set[0]
            # type_vec = Variable(torch.from_numpy(typevec_set).cuda()).long()
            
            input_question = t_question[i] #取出这个问题的问题向量
            input_q_glove = t_q_glove[i] #取出这个问题对应的问题 glove向量
            input_q_glove = np.tile(input_q_glove, (r_num, 1, 1))
            input_q_glove = Variable(torch.from_numpy(input_q_glove).cuda()).float()
            if r_num == 150:
                g_rbvec_set = g_rbvec_set[:r_num]
                g_rcvec_set = g_rcvec_set[:r_num]
            rb_glove = Variable(torch.from_numpy(g_rbvec_set).cuda()).float()
            rc_glove = Variable(torch.from_numpy(g_rcvec_set).cuda()).float() #取出关系向量对应的 glove向量
            # type_glove = Variable(torch.from_numpy(g_typevec_set).cuda()).float() #取出关系向量对应的 glove向量
            
            
            q_id = t_qid_list[i] #取出这个问题的 id ， 为最后找 label准备
            # score_tensor = torch.Tensor(len(rc_glove)).float() #初始化一个计分张量，记录每个relation的得分

            # input_type = type_vec[j].view(1,2)
            # input_type_glove = type_glove[j].view(1,2,300)
            pred, score_tensor = model(0, input_q_glove, 0, rb_glove, 0, rb_glove, 0, rc_glove, 0, rc_glove, '%s'%PATH)
            # print(score)
            # relation = model.get_LSTM_vec(input_relation, input_relation_glove)
            # score = compute_cosine_sim(pred, relation)
            # score_tensor[j] = float(score)
            # print(score)
            # print(score_tensor)
            
            # print(datetime.datetime.now().strftime('%b-%d-%Y %H:%M:%S'))
            # print("5")
            index, gold_score = rank_candidate_relation(score_tensor)
            (result, relation_list) = relation_dic[str(q_id)]['relation_candidate'][k]         
            # print(int(index))
            fobj.write(' %%%% '+ result[0]+'\t'+result[1]+'\t'+relation_list[index]+'\t'+str(gold_score.item()))
            # print(str(gold_score)+'\n')
            #print(type_vec[index])
        # exit()    
        fobj.write("\n")
        # print("----------------------------------------------------------")
        #result = compare_with_label(index ,relation_list, label)
            
            #result_list.append(result)
    #compute_accuracy('%s'%PATH, result_list)


'''
while epoch == 0 and count<number:
    count += 1
    if count ==1 or count%50 == 0:
        print ('%s : number of questions : [%d]'%(time.asctime(time.localtime(time.time())), count))

    t_question, t_q_glove, t_rset, t_glove_rset, t_qid_list, epoch = dp.get_batch_vec('%s'%PATH)
    
    t_question = Variable(torch.from_numpy(t_question).cuda()).long() #读取问题标号向量，使用 Variable包装
    t_q_glove = Variable(torch.from_numpy(t_q_glove).cuda()).float() #读取问题glove向量，使用 Variab95000
    batch_rela_length = len(t_rset) #读取batch中的问题个数，存在变量batch_rela_length中
    #print (batch_rela_length)
    
    # print (t_rset, t_glove_rset.shape)
    for i in xrange(batch_rela_length):
        q_id = t_qid_list[i]   #遍历问题，一个问题一个问题地处理，对每一个问题求出最佳的 relation
        # print(q_id)
        fobj.write(str(q_id)+' %%%% ')
        label_entity = relation_dic[str(q_id)]['right_entity']  
        label_relation = relation_dic[str(q_id)]['right_relation']
        fobj.write(label_entity+" %%%% " + label_relation)
        
        score_list = []
        
        for k, ((result,rbvec_set,rcvec_set), (g_rbvec_set,g_rcvec_set)) in enumerate(zip(t_rset[i], t_glove_rset[i])):
            # (result, rvec_set) = er_tuple
            rb_vec = Variable(torch.from_numpy(rbvec_set).cuda()).long() #取出这个问题的所有relation向量
            rc_vec = Variable(torch.from_numpy(rcvec_set).cuda()).long()
            #rela_len = rela_set[0]
            # type_vec = Variable(torch.from_numpy(typevec_set).cuda()).long()
            
            input_question = t_question[i] #取出这个问题的问题向量
            input_q_glove = t_q_glove[i] #取出这个问题对应的问题 glove向量
            size_q = input_q_glove.size()
            input_question = input_question.view(1,-1)
            input_q_glove = input_q_glove.view(1,-1,300)
            rb_glove = Variable(torch.from_numpy(g_rbvec_set).cuda()).float()
            rc_glove = Variable(torch.from_numpy(g_rcvec_set).cuda()).float() #取出关系向量对应的 glove向量
            # type_glove = Variable(torch.from_numpy(g_typevec_set).cuda()).float() #取出关系向量对应的 glove向量
            
            
            q_id = t_qid_list[i] #取出这个问题的 id ， 为最后找 label准备
            score_tensor = torch.Tensor(len(rc_glove)).float() #初始化一个计分张量，记录每个relation的得分
            for j in xrange(len(rc_glove)):
                input_rb = rb_vec[j].view(1,3)
                input_rb_glove = rb_glove[j].view(1,3,300)
                input_rc = rc_vec[j].view(1,6)
                input_rc_glove = rc_glove[j].view(1,6,300)
                # input_type = type_vec[j].view(1,2)
                # input_type_glove = type_glove[j].view(1,2,300)
                pred, score = model(input_question, input_q_glove, input_rb, input_rb_glove, input_rb, input_rb_glove, input_rc, input_rc_glove, input_rc, input_rc_glove, '%s'%PATH)
                # print(score)
                # relation = model.get_LSTM_vec(input_relation, input_relation_glove)
                # score = compute_cosine_sim(pred, relation)
                score_tensor[j] = float(score)
                # print(score)
            # print(score_tensor)
            
            # print(datetime.datetime.now().strftime('%b-%d-%Y %H:%M:%S'))
            # print("5")
            index, gold_score = rank_candidate_relation(score_tensor)
            (result, relation_list) = relation_dic[str(q_id)]['relation_candidate'][k]         
            # print(index)
            fobj.write(' %%%% '+ result[0]+'\t'+result[1]+'\t'+relation_list[index]+'\t'+str(float(gold_score)))
            # print(str(gold_score)+'\n')
            #print(type_vec[index])
        # exit()    
        fobj.write("\n")
        # print("----------------------------------------------------------")
        #result = compare_with_label(index ,relation_list, label)
            
            #result_list.append(result)
    #compute_accuracy('%s'%PATH, result_list)
'''
