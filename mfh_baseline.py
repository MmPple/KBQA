# coding=utf-8
from __future__ import absolute_import
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import sys
import numpy as np
#import matplotlib.pyplot as plt
#plt.switch_backend('agg')
import os
import sys
import json
import re
import shutil
import json
import datetime
#from PIL import Image
#from PIL import ImageFont, ImageDraw
sys.path.append("..")
import utils.data_provider as data_provider
from utils.data_provider import VQADataProvider
import config

opt = config.parse_opt()
class RankingLoss(nn.Module):
    def __init__(self, vec_size=opt.BATCH_SIZE, margin=0.75):
        super(RankingLoss, self).__init__()
        self.vec_size = vec_size
        self.margin = margin
        return 
    
    # def forward(self, mix, rela_right1, rela_false1, rela_right2, rela_false2, rela_right3, rela_false3, margin=0.75):
    def forward(self, mix, rela_right1, rela_false1,margin=0.75):   
        # right_item = F.cosine_similarity(mix, rela_right)
        # false_item = F.cosine_similarity(mix, rela_false)
        #self.vec_size = rela_right.shape[0]
        zero1 = torch.zeros(self.vec_size)
        zero1 = Variable(zero1.cuda())
        margin_vec1 = torch.cuda.FloatTensor(self.vec_size)
        margin_vec1.fill_(self.margin)
        margin_vec1 = Variable(margin_vec1)
        #print (right_item.shape)
        margin_vec1 = margin_vec1.sub_(rela_right1)
        loss = torch.max(zero1, margin_vec1.add_(rela_false1))
        '''
        zero2 = torch.zeros(self.vec_size)
        zero2 = Variable(zero2.cuda())
        margin_vec2 = torch.cuda.FloatTensor(self.vec_size)
        margin_vec2.fill_(self.margin)
        margin_vec2 = Variable(margin_vec2)
        #print (right_item.shape)
        margin_vec2 = margin_vec2.sub_(rela_right2)
        loss2 = torch.max(zero2, margin_vec2.add_(rela_false2))

        zero3 = torch.zeros(self.vec_size)
        zero3 = Variable(zero3.cuda())
        margin_vec3 = torch.cuda.FloatTensor(self.vec_size)
        margin_vec3.fill_(self.margin)
        margin_vec3 = Variable(margin_vec3)
        #print (right_item.shape)
        margin_vec3 = margin_vec3.sub_(rela_right3)
        loss3 = torch.max(zero3, margin_vec3.add_(rela_false3))
        #print (loss.shape)
        loss = torch.add(loss1, loss2)
        loss = torch.add(loss, loss3)
        loss = torch.div(loss, 3)
        '''
        loss = torch.mean(loss)
        return loss

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
        # print(final_score.shape)
        # wrb = wrb_vec
        wbglove = wrb_glove
        wbglove = wbglove.permute(1, 0, 2)
        # wrb = torch.transpose(wrb, 1, 0).long()
        # wrb = F.tanh(self.Embedding2(wrb))
        # wrb = torch.cat((wrb, wbglove), 2) # 17*batch*600
        wrb_feat, _ = self.LSTM3(wbglove)
        wrb_feat, _ = torch.max(wrb_feat, 0)
        
        # wrc = wrc_vec
        wcglove = wrc_glove
        wcglove = wcglove.permute(1, 0, 2)
        # wrc = torch.transpose(wrc, 1, 0).long()
        # wrc = F.tanh(self.Embedding2(wrc))
        # wrc = torch.cat((wrc, wcglove), 2) # 17*batch*600
        wrc_feat, _ = self.LSTM4(wcglove)
        wrc_feat, _ = torch.max(wrc_feat, 0)
        

        wab = self.linear(qb_feat.mul(wrb_feat).sum(1).unsqueeze(1))
        wac = self.linear(qc_feat.mul(wrc_feat).sum(1).unsqueeze(1))

        wrb_feat = wab * wrb_feat
        wrc_feat = wac * wrc_feat

        wrb_score = F.cosine_similarity(qb_feat, wrb_feat)
        wrc_score = F.cosine_similarity(qc_feat, wrc_feat)
        
        wfinal_score = wrb_score + wrc_score
        # wfinal_score = torch.sigmoid(wr_score)
        
        return 1, final_score, wfinal_score
    
def adjust_learning_rate(optimizer, decay_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
            
def train(opt, folder):
    train_Data = data_provider.VQADataset(opt.TRAIN_DATA_SPLITS, opt.BATCH_SIZE, folder, opt)
    #train_Data.__getitem__(0)
    #train_Data.__getitem__(1)
    print ('---------------------------train_dataset loads sucessfully !--------------------------------------')
    #train_Loader = torch.utils.data.DataLoader(dataset=train_Data,batch_size=1,  shuffle=True, pin_memory=True, num_workers=1)
    #print train_Loader
    model = mfh_baseline(opt)
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr = opt.INIT_LEARNING_RATE)
    torch.cuda.set_device(opt.TRAIN_GPU_ID)
    criterion = RankingLoss()
    train_loss = np.zeros(opt.MAX_ITERATIONS + 1)
    results = []

    for iter_idx, (data, glove_data, rb, rb_glove, rb_false, frb_glove, rc, rc_glove, rc_false, frc_glove, epoch) in enumerate(train_Data):
        model.train()
        # 将以上的 torch类型变量转为np类型，并将维度值为1的维度删去
        # iter_idx += 1
        data = torch.from_numpy(data)
        glove_data = torch.from_numpy(glove_data)
        

        # rb = torch.from_numpy(rb)
        rb_glove = torch.from_numpy(rb_glove)
        # rb_false = torch.from_numpy(rb_false)        
        
        
        # rc = torch.from_numpy(rc)
        rc_glove = torch.from_numpy(rc_glove)
        # rc_false = torch.from_numpy(rc_false)
        fb1, fb2, fb3 = frb_glove
        fc1, fc2, fc3 = frc_glove

        frb_glove1 = torch.from_numpy(fb1)
        frc_glove1 = torch.from_numpy(fc1)
        # frb_glove2 = torch.from_numpy(fb2)
        # frc_glove2 = torch.from_numpy(fc2)
        # frb_glove3 = torch.from_numpy(fb3)
        # frc_glove3 = torch.from_numpy(fc3)
        
        np.int(epoch)

        # 将需要传入网络的张量用Variable包装起来，为传入网络计算模块做准备，并使用cuda加速运算
        # data = Variable(data).cuda().long()
        glove_data = Variable(glove_data).cuda().float()
        # glove_data2 = glove_data.clone()
        # glove_data3 = glove_data.clone()
        
        # rb = Variable(rb).cuda().float()
        rb_glove = Variable(rb_glove).cuda().float()
        # rb_false = Variable(rb_false).cuda().float()
        # rb_glove2 = rb_glove.clone()
        # rb_glove3 = rb_glove.clone()
        # rc = Variable(rc).cuda().float()
        rc_glove = Variable(rc_glove).cuda().float()
        # rc_glove2 = rc_glove.clone()
        # rc_glove3 = rc_glove.clone()
        # rc_false = Variable(rc_false).cuda().float()
        frb_glove1 = Variable(frb_glove1).cuda().float()
        frc_glove1 = Variable(frc_glove1).cuda().float()
        '''        
        frb_glove2 = Variable(frb_glove2).cuda().float()
        frc_glove2 = Variable(frc_glove2).cuda().float()
        frb_glove3 = Variable(frb_glove3).cuda().float()
        frc_glove3 = Variable(frc_glove3).cuda().float()
        '''
        # 将优化器初始化，并将数据传入网络计算模块中进行计算，得到预测分类类别的概率向量pred: opt.BATCH_SIZE * 3000维的向量
        # forward pass : compute  predicted y by passing x to the model
        optimizer.zero_grad()
        pred, right_score1, wrong_score1 = model(0, glove_data, 0, rb_glove, 0, frb_glove1, 0, rc_glove, 0, frc_glove1, 'train')
        # pred, right_score2, wrong_score2 = model(0, glove_data2, 0, rb_glove2, 0, frb_glove2, 0, rc_glove2, 0, frc_glove2, 'train')
        # pred, right_score3, wrong_score3 = model(0, glove_data3, 0, rb_glove3, 0, frb_glove3, 0, rc_glove3, 0, frc_glove3, 'train')
        
        # right = torch.mean(right_score3)
        # right.backward()
        # print (right_score)
        # print right_score.shape, wrong_score.shape
        # rela_right = model.get_LSTM_vec(relation, r_glove)
        # rela_false = model.get_LSTM_vec(relation_false, fr_glove)
        # compute loss 计算损失
        loss = criterion(pred, right_score1, wrong_score1)
        

        # perform a backward pass and update the weights  反向传播并更新权重
        loss.backward()
        optimizer.step()

        # 以下都是输出和保存迭代中的信息
        train_loss[iter_idx] = loss.data.item()

        if iter_idx % opt.DECAY_STEPS== 0 and iter_idx != 0:
            adjust_learning_rate(optimizer, opt.DECAY_RATE)
        if iter_idx % opt.PRINT_INTERVAL == 0 and iter_idx != 0:
            now = str(datetime.datetime.now())
            c_mean_loss = train_loss[iter_idx - opt.PRINT_INTERVAL:iter_idx].mean()/opt.BATCH_SIZE
            # writer.add_scalar('mfh_baseline_glove/train_loss', c_mean_loss, iter_idx)
            # writer.add_scalar('mfh_baseline_glove/lr', optimizer.param_groups[0]['lr'], iter_idx)
            print('{}\tTrain Epoch : {}\tIter: {}\tLoss: {:.12f}'.format(now, epoch, iter_idx, c_mean_loss))
        if iter_idx % opt.CHECKPOINT_INTERVAL == 0 and iter_idx != 0:
            if not os.path.exists('./{}'.format(opt.pth_path)):
                os.makedirs('./{}'.format(opt.pth_path))
            save_path = './{}/mfh_baseline_glove_iter'.format(opt.pth_path) + str(iter_idx) + '.pth'
            torch.save(model.state_dict(), save_path)
        
        
        

