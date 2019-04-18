# -*- coding: utf-8 -*-
import re
import numpy as np
import re, json, random
import config
import torch.utils.data as data
import spacy
import en_vectors_web_lg
import random
import os
import sys
from collections import defaultdict

arg = config.parse_opt()

defaultencoding = 'utf-8'
if sys.getdefaultencoding() != defaultencoding:
    reload(sys)
    sys.setdefaultencoding(defaultencoding)

EPOCHNUM = 1000
QUESTIONDIM = 33
RBDIM = 3
RCDIM = 6
RELATIONNUMBER = 151
GLOVE_EMBEDDING_SIZE = 300
datapath = "/home/chenshuo/KBQA/"

class VQADataProvider : 
    def __init__(self, opt, folder = './result', batchsize = 32, question_dim = QUESTIONDIM , rb_dim = RBDIM, rc_dim = RCDIM, mode = 'train'):
        self.opt = opt
        self.batchsize = batchsize
        self.d_vocabulary = None
        self.batch_index = None
        self.batch_len = None
        self.rev_adict = None
        self.question_dim = question_dim
        self.rb_dim = rb_dim
        self.rc_dim = rc_dim
        #self.relation_number = relation_number
        self.mode = mode
        self.folder = folder
        self.qdic , self.rdic = VQADataProvider.load_data(mode, self.folder)
        
        folder1 = 'mfh_baseline_%s'%opt.TRAIN_DATA_SPLITS 
       
        '''
        with open('./%s/vdict.json'%folder1, 'r') as f:
            self.vdict = json.load(f)
        with open('./%s/rdict.json'%folder1, 'r') as f:
            self.rdict = json.load(f)
        '''
        self.nlp = en_vectors_web_lg.load()
        self.glove_dict = {}

    @staticmethod
    def get_relation_set():
        # fdata = open("/home/chenshuo/.pyenvs/KBQApy35/BuboQA/data/SimpleQuestions_v2/freebase-subsets/processed-FB2M.txt", "r")
        fdata = open(arg.fb_file, 'r')
        r_list = []
        for line in fdata:
            line = line.split('\t')
            re = line[1][17:]
            if re not in r_list:
                r_list.append(re)
        return r_list       
        
    @staticmethod    
    def load_data(mode, folder1):
        qdic = {}
        rdic = {}
        if mode == 'val':
            mode = 'valid'
        
        if os.path.exists('./%s/'%folder1+mode+'_qdic.json') and os.path.exists('./%s/'%folder1+mode+'_rdic.json'):
            print ('restoring vocab of qdic & rdic')
            with open('./%s/'%folder1+mode+'_qdic.json', 'r') as f:
                qdic = json.load(f)
            with open('./%s/'%folder1+mode+'_rdic.json', 'r') as f:
                rdic = json.load(f)
            return qdic, rdic
        
        relation_list = VQADataProvider.get_relation_set()
        rdic["all_relation"] = relation_list
        negative_pool = {"b":{}, "c":{}}
        for r in relation_list:
            b, c = r.split("/")[-2:]
            if not negative_pool["b"].has_key(b):
                negative_pool["b"][b] = [r]
            else:
                negative_pool["b"][b].append(r)
            if not negative_pool["c"].has_key(c):
                negative_pool["c"][c] = [r]
            else:
                negative_pool["c"][c].append(r)

            

        if mode == "train":
            
            felpath = arg.el_file+"/{}-h100.txt".format(mode)
            #"/home/chenshuo/.pyenvs/KBQApy35//BuboQA/entity_linking/results/lstm/%s"%(mode+"-h100.txt")
            fqpath = arg.dataset_file+"/{}.txt".format(mode)
#="/home/chenshuo/.pyenvs/KBQApy35//BuboQA/data/processed_simplequestions_dataset/%s"% (mode+".txt")
            factspath = arg.raw_fb_file
    #"/home/chenshuo/.pyenvs/KBQApy35//BuboQA/data/SimpleQuestions_v2/freebase-subsets/freebase-FB2M.txt"
            
            fq = open(fqpath, "r")
            fel = open(felpath, "r")
            ffacts = open(factspath, "r")
            
            facts = defaultdict(list)
            elresult = defaultdict(list)
            
            ##  将所有KB fact 存入 facts {object: [rel1, rel2, ...], ...}
            for line in ffacts:
                line = line.strip().split("\t")
                facts[line[0]].append(line[1][17:])
            
            ## 将entity linking结果存入elresult {"train-0": [(obj1, score1), ...]}
            error = 0
            for line in fel:
                line = line.strip().split(" %%%% ")
                if len(line) < 2:
                    # print(" empty plus ")
                    error += 1
                else:
                    for result in line[1:]:
                        result = result.split("\t")
                        # print(line[0], result)
                        elresult[line[0]].append(("www.freebase.com/m/"+result[0][5:], result[3]))
            print("number of empty %d"%error)
            
            ## 填充qdic, rdic
            ## qdic = {t_qdic1, ...}
            ## rdic = {t_rdic1, ...}
            ## t_qdic = {q_str: #}
            ## t_rdic = {right_entity:# , right_relation:# , wrong_relation:#}
            
            count1 = 0 
            count2 = 0
            count3 = 0
            for line in fq:
                error = 0
                t_qdic = {}
                t_rdic = {}
                line = line.split("\t")
                t_rdic['right_entity'] = "www.freebase.com/m/"+line[1][5:]
                
                ## 将 question中 entity相关换成标识符 
                el = line[6].split()
                start_position = el.index('I')
                try:
                    end_position = el[start_position:].index('O') + start_position
                except:
                    end_position = 100
                q_list = line[5].split()
                q_str = ' '.join(q_list[:start_position]) + ' HEADENTITY ' + ' '.join(q_list[end_position:])
                
                # print start_position, end_position,  q_str                
                t_qdic['qstr'] = q_str
                
                ## right_relaiton
                right_r = line[3][3:].replace(".","/")
                t_rdic['right_relation'] = right_r
                t_rlist = []
                e_in = False
                rs_set = []
                wr_select = 0
                
                ## 生成负样本
                
                right_b, right_c = right_r.split("/")[-2:]
                if len(negative_pool["b"][right_b]) > 1:    
                    while True:
                         b = random.sample(negative_pool["b"][right_b], 1)[0]
                         if b != right_r: break
                else:
                    while True:
                        b = random.sample(negative_pool["b"].keys(), 1)[0]
                        if b != right_b:
                            b = "/".join(right_r.split("/")[:-2])+"/"+b+"/"+right_r.split("/")[-1]
                            break
                wrong_c = b

                if len(negative_pool["c"][right_c]) > 1:    
                    while True:
                         c = random.sample(negative_pool["c"][right_c], 1)[0]
                         if c != right_r: break
                else:
                    while True:
                        c = random.sample(negative_pool["c"].keys(), 1)[0]
                        if c != right_c:
                            c = "/".join(right_r.split("/")[:-2])+"  /"+c+"/"+right_r.split("/")[-1]
                            break
                wrong_b = c

                while True:
                         wrong_r = random.sample(relation_list, 1)[0]
                         if wrong_r != right_r: break
                
                

                
                try:
                    for result in elresult[line[0]]:
                        s_set = facts[result[0]]
                        t_rlist.append((result, s_set))
                        if t_rdic["right_entity"] == result[0]:
                            e_in = True
                            rs_set = s_set
                except:
                    error += 1
                # t_rdic['subgraph_candidate'] = t_rlist ### [((entity, score) , [re1, re2,...]),...]
                # t_rdic['right_relation'] = facts[t_rdic['right_entity']] ### [r1,r2,r3...]
                '''
                t_rdic['wrong_relation'] = VQADataProvider.get_wrong_relation(relation_list, t_qdic['qstr'], t_rdic['right_relation'])
                '''
                if e_in and len(rs_set)>1:
                    for r in rs_set:
                        if r != t_rdic['right_relation']:
                            t_rdic["wrong_relation"] = r
                            wr_select = 1
                            count1 += 1
                            break
                elif len(elresult[line[0]])>1:
                    for (e, s) in elresult[line[0]]:
                        if e != t_rdic['right_entity']:
                            for r in facts[e]:
                                if r != t_rdic["right_relation"]:
                                    t_rdic["wrong_relation"] = r
                                    wr_select = 2
                                    count2 += 1
                                    break
                        if wr_select == 2:
                            break
                
                if wr_select not in [1, 2]:
                    while True:
                        wrong_relation = relation_list[random.randint(0, len(relation_list)-1)]
                        if wrong_relation != t_rdic["right_relation"]:
                            t_rdic["wrong_relation"] = wrong_relation
                            wr_select = 3
                            count3 += 1
                            break
                t_rdic['wrong_relation'] = [t_rdic["wrong_relation"], wrong_c, wrong_r]
                        
                '''
                if len(t_rlist)>1:
                    if t_rlist[-1][0][0] != t_rdic['right_entity']:
                        t_rdic['wrong_relation'] = facts[t_rlist[-1][0][0]]
                    else:
                        t_rdic['wrong_subgraph'] = facts[t_rlist[-2][0][0]]
                else:
                    t_rdic['wrong_subgraph'] = facts[previous_entity]
                '''
           
                
                qdic[line[0]] = t_qdic
                rdic[line[0]] = t_rdic
                # previous_entity = "www.freebase.com/m/"+line[1][5:]
                
            # print(count1, count2, count3)
                
            with open('./%s/'%folder1+'train_'+'qdic.json', 'w') as f:
                json.dump(qdic, f)
            with open('./%s/'%folder1+'train_'+'rdic.json', 'w') as f:
                json.dump(rdic, f)
                
            return qdic, rdic

    
        
        elif mode == "valid" or mode == "test":
            
            felpath = arg.el_file+"/{}-h100.txt".format(mode)
#"/home/chenshuo/.pyenvs/KBQApy35//BuboQA/entity_linking/results/lstm/%s"%(mode+"-h100.txt")
            fqpath = arg.dataset_file+"/{}.txt".format(mode)
#"/home/chenshuo/.pyenvs/KBQApy35//BuboQA/data/processed_simplequestions_dataset/%s"% (mode+".txt")
            factspath =arg.raw_fb_file
#"/home/chenshuo/.pyenvs/KBQApy35//BuboQA/data/SimpleQuestions_v2/freebase-subsets/freebase-FB2M.txt"
            
            fq = open(fqpath, "r")
            fel = open(felpath, "r")
            ffacts = open(factspath, "r")
            
            facts = defaultdict(list)
            elresult = defaultdict(list)
            noname_count = 0            

            for line in ffacts:
                line = line.strip().split("\t")
                facts[line[0]].append(line[1][17:])  ### facts
            error = 0
            for line in fel:
                line = line.strip().split(" %%%% ") 
                if len(line) < 2:
                    # print(" empty plus ")
                    error += 1
                else:
                    for result in line[1:]:
                        result = result.split("\t")
                        # print(line[0], result)
                        elresult[line[0]].append(("www.freebase.com/m/"+result[0][5:], result[3]))
            print("number of empty %d"%error)
            error = 0
            for line in fq:
                
                t_qdic = {}
                t_rdic = {}
                line = line.split("\t")
                t_rdic['right_entity'] = "www.freebase.com/m/"+line[1][5:]

                try:
                    el = line[6].split()
                    start_position = el.index('I')
                except:
                    noname_count += 1
                    continue                

                try:
                    end_position = el[start_position:].index('O') + start_position
                except:
                    end_position = 100
                q_list = line[5].split()
                q_str = ' '.join(q_list[:start_position]) + ' HEADENTITY ' + ' '.join(q_list[end_position:])

                t_qdic['qstr'] = q_str
                right_r = line[3][3:].replace(".","/")
                t_rdic['right_relation'] = right_r
                t_rlist = []
                try:
                    for result in elresult[line[0]]:
                        t_rlist.append((result, facts[result[0]]))
                except:
                    error += 1 
                t_rdic['relation_candidate'] = t_rlist ### [((entity, score) , [re1, re2,...]),...]
                
                qdic[line[0]] = t_qdic
                rdic[line[0]] = t_rdic
            print error 
            print noname_count
            if not os.path.exists('./%s'%folder1):
                os.makedirs('./%s'%folder1)              

    
            with open('./%s/'%folder1+mode+'_qdic.json', 'w') as f:
                json.dump(qdic, f)
            with open('./%s/'%folder1+mode+'_rdic.json', 'w') as f:
                json.dump(rdic, f)                

            return qdic, rdic
        
        else:
            exit(1)


        
        '''
        elif mode == "val" or mode == "test":
            if mode == "val":
                mode = "valid"            
            fapath = mode+".txt"
            dicpath = mode+"-entity-about.json"
            fq = open("/home/chenshuo/.pyenvs/KBQApy35//BuboQA/data/processed_simplequestions_dataset/%s"%fapath, "r")
            fdic = open("/home/chenshuo/.pyenvs/KBQApy35//BuboQA/data/SimpleQuestions_v2/freebase-subsets/%s"%dicpath, "r")
            sro_dic = json.load(fdic)
            for line in fq:
                t_qdic = {}
                t_rdic = {}
                line = line.split("\t")
                index = line[0]+" "+"www.freebase.com/m/"+line[1][5:]
                t_qdic['qstr'] = line[5]
                right_r = line[3][3:].replace(".","/")
                t_rdic['right_relation'] = right_r
                t_rlist = []
                for fact in sro_dic[index]:
                    t_rlist.append(fact.split("\t")[0][17:])
                t_rdic['relation_candidate'] = t_rlist
                qdic[line[0]] = t_qdic
                rdic[line[0]] = t_rdic
            return qdic, rdic
        
        else:
            exit(1)
        '''
    
    @staticmethod    
    def get_wrong_relation(relation_list, q_str, right_relation):
        word_dic = VQADataProvider.seq_to_list(q_str)
        word_dic.extend(VQADataProvider.seq_to_list(right_relation))
        while True:
            wrong_relation = relation_list[random.randint(0, len(relation_list)-1)]
            re_word_list = VQADataProvider.seq_to_list(wrong_relation)
            for w in re_word_list:
                if w in word_dic:
                    break
            else:
                return wrong_relation
    
    
    @staticmethod
    def get_qlist_and_labellist(PATH):
        with open(PATH) as f:
            lines = f.readlines()
            list_1 = []
            list_2 = []
            for line in lines:
                line = line.rstrip('\n')
                line = line.rstrip('\r')
                data = line.split('\t')
                relation_data = data[1].split('com/') 
                list_1.append(relation_data[1])
                list_2.append(data[3])
        return list_1, list_2
    
    @staticmethod
    def get_all_relation_list(PATH):
        with open(PATH) as f:
            lines = f.readlines()
            count = 0
            list_relation_str = []
            for line in lines:
                line = line.rstrip('\n')
                line = line.rstrip('\r')
                line = line[1:]
                #line_list =line.split('/')
                #relation_str = line_list[1]
                relation_str = line
                list_relation_str.append(relation_str)
                count += 1
        return list_relation_str # 得到全部relation_str的list和个数
    
    @staticmethod
    def get_relation_list(mode):
        r_id_list, _= VQADataProvider.get_list_two_data(datapath+'freebaselsex/' + mode +'.replace_ne.txt')
        all_relation_list = VQADataProvider.get_all_relation_list(datapath+'freebaselsex/relation.2M.list.txt')
        relation_list = []
        count_relation_for_question_list = []
        counter = 0
        for index_list in r_id_list:
            t_r_list = []
            index_list = index_list.split(' ')
            for index in index_list:
                t_r_list.append(all_relation_list[int(index) - 1 ])
            relation_list.append(t_r_list)
            count_relation_for_question_list.append(len(t_r_list))
        return relation_list, count_relation_for_question_list   # 返回数据集的relation_list和每个问题对应的relation个数
    
    @staticmethod
    def get_list_two_data(PATH):
        with open(PATH) as f:
            lines = f.readlines()
            list_1 = []
            list_2 = []
            for line in lines:
                line = line.rstrip('\n')
                line = line.rstrip('\r')
                data = line.split('\t')
                if data[1] == 'noNegativeAnswer':
                    list_1.append(data[0])
                else:
                    list_1.append(data[0] + ' ' + data[1])
                list_2.append(data[2])
        return list_1, list_2 #得到两种数据的list

    
    def getQuesIds(self):
        return self.qdic.keys()
    
    def getQuesStr(self, qid):
        return self.qdic[qid]['qstr']
        
    def getRelationStr(self, qid):
        return self.rdic[qid]['right_relation']
        
    def getFalseRelationStr(self, qid):
        return self.rdic[qid]['wrong_relation']

    @staticmethod
    def seq_to_list(s):
            t_str = s.lower()
            for i in [r'\?' , r'\!' , r'\'' ,r'\"' , r'\$' , r'\:' , r'\@', r'\(' , r'\)' , r'\,' , r'\.' , r'\;']:
                t_str = re.sub(i , ' ' , t_str)
            for i in [r'\-' , r'\/', r'\_']:
                t_str = re.sub(i , ' ', t_str)
            t_str = t_str.replace('#head_entity#', 'headentity')
            q_list = t_str.lower().split(' ')
            q_list = filter(lambda x: len(x) > 0, q_list)
            q_list = list(q_list)
            return q_list # 数据清洗，将字符串中的标点符号替换，并将空数据剔除,得到完全由单词组成的一个list

    def tlist_to_vec(self, max_length, t_list, question_or_relation):
        vec = np.zeros(max_length)
        glove_matrix = np.zeros((max_length, GLOVE_EMBEDDING_SIZE))
        for i in xrange(max_length):
            if i >= len(t_list):
                pass
            else:
                w = t_list[i]
                if w not in self.glove_dict:
                    self.glove_dict[w] = self.nlp(u'%s' %w).vector
                glove_matrix[i] = self.glove_dict[w]
                '''
                if question_or_relation == 'question':
                    if self.vdict.has_key(w) is False:
                        w = ''
                    vec[i] = self.vdict[w]
                elif question_or_relation == 'relation':
                    if self.rdict.has_key(w) is False:
                        w = ''
                    vec[i] = self.rdict[w]
                '''
        return vec, glove_matrix

    def create_batch(self, t_qid_list, datasplit):
        '''
        q_dim = 0
        rb_dim = 0
        rc_dim = 0
        
        for i,qid in enumerate(t_qid_list):
            q_str = self.getQuesStr(qid)
            t_dim = len(VQADataProvider.seq_to_list(q_str))
            if (q_dim < t_dim):
                q_dim = t_dim
                
            if datasplit == "train":
                r_str = self.getRelationStr(qid).split('/')[-2:]
                fr_str = self.getFalseRelationStr(qid).split('/')[-2:]
                
                rb_str = r_str[0]
                rc_str = r_str[1]
                frb_str = fr_str[0]
                frc_str = fr_str[1]
                
                rb_list = VQADataProvider.seq_to_list(rb_str)
                frb_list = VQADataProvider.seq_to_list(frb_str)
                
                rc_list = VQADataProvider.seq_to_list(rc_str)
                frc_list = VQADataProvider.seq_to_list(frc_str)
                
                
                if(rb_dim < len(rb_list)):
                    rb_dim = len(rb_list)
                if(rb_dim < len(frb_list)):
                    rb_dim = len(frb_list)
                
                if(rc_dim < len(rc_list)):
                    rc_dim = len(rc_list)
                if(rc_dim < len(frc_list)):
                    rc_dim = len(frc_list)
          
            if datasplit == "val" or datasplit == "test":
                for er_tuple in self.rdic[qid]['relation_candidate']:
                    relation_list = er_tuple[1]
                    for r_str in relation_list:
                        r_str = r_str.split('/')[-2:]
                        rb_str = r_str[0]
                        rc_str = r_str[1]
                        rb_list = VQADataProvider.seq_to_list(rb_str)
                        rc_list = VQADataProvider.seq_to_list(rc_str)
                        if(rb_dim < len(rb_list)):
                            rb_dim = len(rb_list)
                        if(rc_dim < len(rc_list)):
                            rc_dim = len(rc_list)
        '''   
        qvec = (np.zeros(self.batchsize*self.question_dim)).reshape(self.batchsize, self.question_dim)
        glove_matrix_for_question = np.zeros((self.batchsize, self.question_dim, GLOVE_EMBEDDING_SIZE))
        
        for i,qid in enumerate(t_qid_list):
            q_str = self.getQuesStr(qid)
            
            q_list = VQADataProvider.seq_to_list(q_str)
            '''
            for i in range(q_dim-len(q_list)):
                q_list.append("<pad>")
            '''
            t_qvec, t_glove_qvec_matrix = self.tlist_to_vec(self.question_dim, q_list, 'question')
            qvec[i,...] = t_qvec
            glove_matrix_for_question[i,...] = t_glove_qvec_matrix
        
        if datasplit == "train":
            rbvec = (np.zeros(self.batchsize*self.rb_dim)).reshape(self.batchsize, self.rb_dim)
            frbvec1 = (np.zeros(self.batchsize*self.rb_dim)).reshape(self.batchsize, self.rb_dim)
            frbvec2 = (np.zeros(self.batchsize*self.rb_dim)).reshape(self.batchsize, self.rb_dim)
            frbvec3 = (np.zeros(self.batchsize*self.rb_dim)).reshape(self.batchsize, self.rb_dim)
            glove_matrix_for_rb = np.zeros((self.batchsize, self.rb_dim, GLOVE_EMBEDDING_SIZE))
            glove_matrix_for_frb1 = np.zeros((self.batchsize, self.rb_dim, GLOVE_EMBEDDING_SIZE))
            glove_matrix_for_frb2 = np.zeros((self.batchsize, self.rb_dim, GLOVE_EMBEDDING_SIZE))
            glove_matrix_for_frb3 = np.zeros((self.batchsize, self.rb_dim, GLOVE_EMBEDDING_SIZE))
        
            rcvec = (np.zeros(self.batchsize*self.rc_dim)).reshape(self.batchsize, self.rc_dim)
            frcvec1 = (np.zeros(self.batchsize*self.rc_dim)).reshape(self.batchsize, self.rc_dim)
            frcvec2 = (np.zeros(self.batchsize*self.rc_dim)).reshape(self.batchsize, self.rc_dim)
            frcvec3 = (np.zeros(self.batchsize*self.rc_dim)).reshape(self.batchsize, self.rc_dim)
            glove_matrix_for_rc = np.zeros((self.batchsize, self.rc_dim, GLOVE_EMBEDDING_SIZE))
            glove_matrix_for_frc1 = np.zeros((self.batchsize, self.rc_dim, GLOVE_EMBEDDING_SIZE))
            glove_matrix_for_frc2 = np.zeros((self.batchsize, self.rc_dim, GLOVE_EMBEDDING_SIZE))
            glove_matrix_for_frc3 = np.zeros((self.batchsize, self.rc_dim, GLOVE_EMBEDDING_SIZE))

        
        if datasplit == "val" or datasplit == "test":
            rset = []
            g_rset = []
        
        for i,qid in enumerate(t_qid_list):

            if datasplit == "train":
                
                r_str = self.getRelationStr(qid).split("/")[-2:]
                wrong_b, wrong_c, wrong_r = self.getFalseRelationStr(qid)
                wrong_b = wrong_b.split("/")[-2:]
                wrong_c = wrong_c.split("/")[-2:]
                wrong_r = wrong_r.split("/")[-2:]
                

                rb_str = r_str[0]
                rc_str = r_str[1]
                rb_list = rb_str.split("_")
                rc_list = rc_str.split("_")
                
                frb_str1 = wrong_b[0]
                frc_str1 = wrong_b[1]
                frb_list1 = frb_str1.split("_")
                frc_list1 = frc_str1.split("_")
                frb_str2 = wrong_c[0]
                frc_str2 = wrong_c[1]
                frb_list2 = frb_str2.split("_")
                frc_list2 = frc_str2.split("_")
                frb_str3 = wrong_r[0]
                frc_str3 = wrong_r[1]
                frb_list3 = frb_str3.split("_")
                frc_list3 = frc_str3.split("_")
                '''
                for i in range(rb_dim - len(rb_list)):
                    rb_list.append("<pad>")
                for i in range(rc_dim - len(rc_list)):
                    rc_list.append("<pad>")
                for i in range(rb_dim - len(frb_list)):
                    frb_list.append("<pad>")
                for i in range(rc_dim - len(frc_list)):
                    frc_list.append("<pad>")
                '''

                try:
                    t_rbvec, t_glove_rbvec_matrix = self.tlist_to_vec(self.rb_dim, rb_list, 'relation')
                    t_rcvec, t_glove_rcvec_matrix = self.tlist_to_vec(self.rc_dim, rc_list, 'relation')
                except:
                    print r_list
                    print fr_list
                    exit()

                t_frbvec1, t_glove_frbvec_matrix1 = self.tlist_to_vec(self.rb_dim, frb_list1, 'relation')
                t_frcvec1, t_glove_frcvec_matrix1 = self.tlist_to_vec(self.rc_dim, frc_list1, 'relation')
                t_frbvec2, t_glove_frbvec_matrix2 = self.tlist_to_vec(self.rb_dim, frb_list2, 'relation')
                t_frcvec2, t_glove_frcvec_matrix2 = self.tlist_to_vec(self.rc_dim, frc_list2, 'relation')                
                t_frbvec3, t_glove_frbvec_matrix3 = self.tlist_to_vec(self.rb_dim, frb_list3, 'relation')
                t_frcvec3, t_glove_frcvec_matrix3 = self.tlist_to_vec(self.rc_dim, frc_list3, 'relation')

                rbvec[i,...] = t_rbvec
                glove_matrix_for_rb[i,...] = t_glove_rbvec_matrix               
                rcvec[i,...] = t_rcvec
                glove_matrix_for_rc[i,...] = t_glove_rcvec_matrix

                frbvec1[i,...] = t_frbvec1
                glove_matrix_for_frb1[i,...] = t_glove_frbvec_matrix1
                frcvec1[i,...] = t_frcvec1
                glove_matrix_for_frc1[i,...] = t_glove_frcvec_matrix1
                frbvec2[i,...] = t_frbvec2
                glove_matrix_for_frb2[i,...] = t_glove_frbvec_matrix2
                frcvec2[i,...] = t_frcvec2
                glove_matrix_for_frc2[i,...] = t_glove_frcvec_matrix2  
                frbvec3[i,...] = t_frbvec3
                glove_matrix_for_frb3[i,...] = t_glove_frbvec_matrix3
                frcvec3[i,...] = t_frcvec3
                glove_matrix_for_frc3[i,...] = t_glove_frcvec_matrix3
                

            if datasplit == 'val' or datasplit == 'test':
                entity_number = len(self.rdic[qid]['relation_candidate'])
                enre_list = []
                g_enre_list = []
                #entype_list = []
                #g_entype_list = []
                for er_tuple in self.rdic[qid]['relation_candidate']:
                    # for t_result in result:
                      #  print(t_result)
                    result = er_tuple[0]
                    relation_list = er_tuple[1]
                    relation_number = len(relation_list)
                    
                    rbvec_set = (np.zeros(relation_number*self.rb_dim)).reshape(relation_number, self.rb_dim)
                    glove_rb_set = (np.zeros(relation_number*self.rb_dim*GLOVE_EMBEDDING_SIZE)).reshape(relation_number, self.rb_dim, GLOVE_EMBEDDING_SIZE)
                    rcvec_set = (np.zeros(relation_number*self.rc_dim)).reshape(relation_number, self.rc_dim)
                    glove_rc_set = (np.zeros(relation_number*self.rc_dim*GLOVE_EMBEDDING_SIZE)).reshape(relation_number, self.rc_dim, GLOVE_EMBEDDING_SIZE)
                    
                    r_str_list = relation_list
                    for t in xrange(relation_number):
                        r_str = r_str_list[t].split('/')[-2:]
                        rb_str = r_str[0]
                        rc_str = r_str[1]
                        rb_list = rb_str.split("_")
                        rc_list = rc_str.split("_")
                        '''
                        for i in range(rb_dim - len(rb_list)):
                            rb_list.append("<pad>")
                        for i in range(rc_dim - len(rc_list)):
                            rc_list.append("<pad>")  
                        '''    
                        
                        rbvec_set[t,...], glove_rb_set[t,...] = self.tlist_to_vec(self.rb_dim, rb_list, 'relation')
                        rcvec_set[t,...], glove_rc_set[t,...] = self.tlist_to_vec(self.rc_dim, rc_list, 'relation')
                    enre_list.append((result, rbvec_set, rcvec_set))
                    g_enre_list.append((glove_rb_set, glove_rc_set))
                        
                    # entype_list.append((result, typevec_set))
                    # g_entype_list.append(glove_type_set)
                    
                rset.append(enre_list)
                g_rset.append(g_enre_list)
                
                #typeset.append(entype_list)
                #g_typeset.append(g_entype_list)
                
        if datasplit == 'val' or datasplit == 'test':
            return qvec, glove_matrix_for_question, rset, g_rset
            
        frb_glove_set = [glove_matrix_for_frb1, glove_matrix_for_frb2, glove_matrix_for_frb3]
        frc_glove_set = [glove_matrix_for_frc1, glove_matrix_for_frc2, glove_matrix_for_frc3]
        return qvec, glove_matrix_for_question, rbvec, glove_matrix_for_rb, frbvec1, frb_glove_set, rcvec, glove_matrix_for_rc, frcvec1, frc_glove_set

    def get_batch_vec(self, datasplit):# dataprovider的顶层接口，返回batch张量
        if self.batch_len is None:
            qid_list = self.getQuesIds()
            qid_list.sort()
            if datasplit == "train":
                random.shuffle(qid_list)
            self.qid_list = qid_list
            self.batch_len = len(qid_list)
            self.batch_index = 0
            self.epoch_counter = 0
               
        counter = 0
        t_qid_list = []
        while counter < self.batchsize:
            t_qid = self.qid_list[self.batch_index]
            t_qid_list.append(t_qid)
            counter += 1
            
            if self.batch_index < self.batch_len-1:
                self.batch_index += 1
            else:
                self.epoch_counter += 1
                qid_list = self.getQuesIds()
                random.shuffle(qid_list)
                self.qid_list = qid_list
                self.batch_index = 0
                print("finish epoch[%d] \n" %self.epoch_counter)
        
        t_batch = self.create_batch(t_qid_list, datasplit)
        return t_batch + (t_qid_list, self.epoch_counter)

class VQADataset():
    
    def __init__(self, mode, batchsize, folder, opt):
        self.batchsize = batchsize
        self.mode = mode
        self.folder = folder
        if self.mode == 'val' or self.mode == 'test':
            pass
        else:
            self.dp = VQADataProvider(opt, batchsize=self.batchsize, mode=self.mode, folder=self.folder)
            
    def __iter__(self):
        return self
    
    def next(self):
        if self.mode == 'val' or self.mode == 'test':
            pass
        else:
            question, q_glove_matrix, rb, rb_glove_matrix, frb, glove_matrix_for_frb, rc, rc_glove_matrix, frc, glove_matrix_for_frc, _, epoch = self.dp.get_batch_vec(self.mode)
        # word_length = np.sum(cont, axis=1)
        return question, q_glove_matrix, rb, rb_glove_matrix, frb, glove_matrix_for_frb, rc, rc_glove_matrix, frc, glove_matrix_for_frc, epoch
        
    def __len__(self):
        if self.mode == 'train':
            return 200000
