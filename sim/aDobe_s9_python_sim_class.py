import math
import os

import torch
from torch import nn

from modules import binaryfunction, ir_1w8a
from sim.para_save import sw_save, bn_save, hardtanh_save
from sim.util import util


input_feature = None
def neural_sim_conv(self, input, output):
    global input_feature
    # gen_feature(input, self.name)
    # return
    if input_feature is None:
        gen_feature(input,self.name)
        # exit()
        input_feature = output[0]
    else:
        if input_feature.eq(input[0]).all() is False:
            print("+++++++++++++++++++++++++++++++++++")
            exit()

    print(self.name)
    input_feature = output[0]
    feature = input[0].cpu()
    weight = self.Qweight.clone()
    weight[weight==-1] = torch.tensor(0)

    # resu = []
    # count = 0
    # for col in range(3):
    #     for row in range(3):
    #
    #         str_data = ''
    #         for j in range(16):
    #
    #             for i in range(16):
    #                 str_data =  str_data+str(int(weight[i,j,row,col]))
    #             print(str_data)
    #             count = count + 1
    #             if count %2 == 0:
    #                 print(hex(int(str_data,2)))
    #                 resu.append('0x'+hex(int(str_data,2))[2:].rjust(8,'0'))
    #                 str_data = ''
    # gen_weight(weight=weight[])
    # if weight.size(0) >= 16:
    #     gen_weight(weight=weight[0:16,...] ,name=self.name+'_16_')
    # if weight.size(0) >= 32:
    #     gen_weight(weight=weight[16:32,...],name=self.name+'_32_')
    # if weight.size(0) >= 48:
    #     gen_weight(weight=weight[32:48,...],name=self.name+'_48_')
    # if weight.size(0) >= 64:
    #     gen_weight(weight=weight[48:64,...],name=self.name+'_64_')

    gen_weight(weight=weight, name=self.name)

    return

    gen_conv_mid_data(feature,self.name,self.Qweight,output)

def gen_weight(weight,name):


    for channel_in in range(weight.size(1)//16):
        for channel_out in range(weight.size(0)//16):
            resu = []
            count = 0
            str_data = ''
            for j in range(channel_in*16,(channel_in+1)*16):
                for row in range(3):
                    for col in range(3):
                        for i in range(channel_out*16,(channel_out+1)*16):
                            str_data = str(int(weight[i,j,row,col])) + str_data
                            count = count + 1
                            if count  == 8:

                                resu.append('0x'+hex(int(str_data,2))[2:].rjust(2,'0'))
                                str_data = ''
                                count = 0


            file_path = 'sim/mid_data/'+name+"weight_in"+str((channel_in+1)*16)+"_out"+str((channel_out+1)*16)+".coe"
            with open(file_path, mode='w', encoding='utf-8') as file_obj:
                for i in resu:
                    file_obj.write(i+',\n')


# 将图片保存下来
def gen_feature(input,name):
    resu = []

    for i in range(input[0].size(2)):
        for j in range(input[0].size(3)):
            str = ''
            for c in range(input[0].size(1)):
                val = input[0][0, c, i, j]
                val_int = (int(val)) & 0xF
                val_hex = hex(val_int)
                val_hex = val_hex[2:].rjust(1, '0')
                str =  val_hex+str
            resu.append('0x'+str+',')

    file_path = 'sim/mid_data/'+name+"_input.coe"
    with open(file_path, mode='w', encoding='utf-8') as file_obj:
        file_obj.write( '{')
        for i in resu:
            file_obj.write(i+'\n')
        file_obj.write('},\n')

def neural_sim_bn(self, input, output):
    global input_feature
    if input_feature is None:
        input_feature = output[0]
    else:
        if input_feature.eq(input[0]).all() is False:
            print("+++++++++++++++++++++++++++++++++++")
            exit()
    # print(self.bw)
    bw = torch.zeros(self.bw.size(0))

    sw = input[1]
    for i in range(self.bw.size(0)):
        bw[i] = self.bw[i,i,0,0]
    bb = self.bb
    input_feature = output[0]
    feature = input[0].cpu()
    i = 0

    sw_save(sw)
    bn_save(self.bw,bb,i,1)
    i = i+1
    # gen_bn_mid_data(feature.cpu(),self.name,bw.cpu(),sw.cpu(),bb.cpu(),output[0].cpu())
short_cut = 0
def neural_sim_resi(self, input, output):
    global input_feature
    if input_feature is None:
        input_feature = output
        print("================================================")
    # else:
    #     ...
    #     # if input_feature.eq(input[1]).all() is False:
    #     #     print("+++++++++++++++++++++++++++++++++++")
    #     #     exit()
    #     data   = input[0]
    #     resi   = input[1]
    #     para   = input[2]
    #     change = input[3]
    global short_cut
    data   = input[0]
    resi   = input[1]
    para   = input[2]
    change = input[3]
    short_cut = para
    if not change:
        if data.size(1)>=16:
            print(self.name + '_16')
            util.resi_after_nochange_coe(data[:,0 :16,:,:],resi[:,0 :16,:,:],para,self.name+'_16')
        if data.size(1)>=32:
            print(self.name + '_32')
            util.resi_after_nochange_coe(data[:,16:32,:,:],resi[:,16:32,:,:],para,self.name+'_32')
        if data.size(1)>=64:
            print(self.name + '_48')
            util.resi_after_nochange_coe(data[:,32:48,:,:],resi[:,32:48,:,:],para,self.name+'_48')
        if data.size(1)>=64:
            print(self.name + '_64')
            util.resi_after_nochange_coe(data[:,48:64,:,:],resi[:,48:64,:,:],para,self.name+'_64')
    else:
        resi_new = torch.zeros(resi.size(0), resi.size(1) * 2, resi.size(2) // 2, resi.size(3) // 2)
        # print(resi_new.shape,data.shape,resi.shape,resi_new[:,resi.size(1)//2-1:resi.size(1)+resi.size(1)//2,:,:].shape)
        resi_new[:, resi.size(1) // 2:resi.size(1) + resi.size(1) // 2, :, :] = resi[:, :, ::2, ::2]
        # print((resi.transpose(1, 3).transpose(1, 2))[0,0,:,:])
        # print((resi.transpose(1, 3).transpose(1, 2))[0,1,:,:])
        # print((resi.transpose(1, 3).transpose(1, 2))[0, 2, :, :])
        # print((resi[:, :, ::2, ::2].transpose(1, 3).transpose(1, 2))[0,1,:,:])
        # exit()
        # print(para)
        # print((resi_new*para+data).eq(output[0, 0:16, 0, 0]).all())
        # exit()
        if data.size(1) >= 16:
            util.resi_after_change_coe(data[:, 0:16, :, :],  resi_new[:, 0:16, :, :] , para, self.name + '_16',output)
        if data.size(1) >= 32:
            util.resi_after_change_coe(data[:, 16:32, :, :], resi_new[:, 16:32, :, :], para, self.name + '_32',output)
        if data.size(1) >= 64:
            util.resi_after_change_coe(data[:, 32:48, :, :], resi_new[:, 32:48, :, :], para, self.name + '_48',output)
        if data.size(1) >= 64:
            util.resi_after_change_coe(data[:, 48:64, :, :], resi_new[:, 48:64, :, :], para, self.name + '_64',output)


def neural_sim_act(self, input, output):
    global input_feature
    if input_feature is None:
        input_feature = output[0]
        print("================================================")
    else:
        if input_feature.eq(input[0]).all() is False:
            print("+++++++++++++++++++++++++++++++++++")
            exit()
    compare = []

    input = input[0].round()

    list_for = list(set(output.view(-1).round().tolist()))
    list_for = sorted(list_for)

    for i in list_for:
        # print(input.eq(i))
        compare.append([input[output==i].min(),input[output==i].max(),i])
    global short_cut
    hardtanh_save(input,output,short_cut)
    gen_act_mid_data(input,compare,self.name,output)

def gen_bn_mid_data(input,name,bw,sw,bb,output):

    if input.size(1) >= 16:
        channel1 = input[ :,0:16, :, :]
        print(name + '_16')
        data = util.shift_after_bn_coe(channel1,name+'_16',bw[0:16],sw[:,0:16,:,:],bb[0:16],output[ :,0:16, :, :])
    if input.size(1) >= 32:
        channel1 = input[ :,16:32, :, :]
        print(name + '_32')
        data = util.shift_after_bn_coe(channel1,name+'_32',bw[16:32],sw[:,16:32,:,:],bb[16:32],output[ :,0:16, :, :])

    if input.size(1) >= 64:
        channel1 = input[ :,32:48, :, :]
        print(name + '_48')
        data = util.shift_after_bn_coe(channel1, name + '_48',bw[32:48],sw[:,32:48,:,:],bb[32:48], output[ :,0:16, :, :])
    if input.size(1) >= 64:
        channel1 = input[ :,48:64, :, :]
        print(name + '_64')
        data = util.shift_after_bn_coe(channel1, name + '_64',bw[48:64],sw[:,48:64,:,:],bb[48:64], output[ :,0:16, :, :])

def gen_conv_mid_data(input,name,weight,output):
    gen_out0 = 0
    gen_out1 = 0
    gen_out2 = 0
    gen_out3 = 0
    data_cat = 0
    if input.size(1) >= 16:
        print(name+'_16')

        channel1 = input[:,0:16,:,:]
        data = util.padd_after_coe(channel1[0],name+'_pad_16',True)

        data = util.gen_after_coe(data,name+'_gen_16',True)

        data = util.sender_after_coe(data, name+'_sender_16',True)
        gen_out0 = data
        #
        data_cat = gen_out0

    if input.size(1) >= 32:
        channel1 = input[:,16:32,:,:]
        print(name + '_32')
        data = util.padd_after_coe(channel1[0],name+'_pad_32',True)

        data = util.gen_after_coe(data,name+'_gen_16',True)

        data = util.sender_after_coe(data, name+'_sender_32',True)
        #
        gen_out1 = data
        data_cat = torch.cat((data_cat,gen_out1),0)

    if input.size(1) >= 64:
        print(name + '_48')
        channel1 = input[:,32:48,:,:]
        data = util.padd_after_coe(channel1[0],name+'_pad_48',True)

        data = util.gen_after_coe(data,name+'_gen_48',True)

        data = util.sender_after_coe(data, name+'_sender_48',True)
        #
        gen_out2 = data
        data_cat = torch.cat((data_cat,gen_out2),0)
    if input.size(1) >= 64:
        print(name + '_64')
        channel1 = input[:,48:64,:,:]
        data = util.padd_after_coe(channel1[0],name+'_pad_64',True)

        data = util.gen_after_coe(data,name+'_gen_64',True)

        data = util.sender_after_coe(data, name+'_sender_64',True)
        gen_out3 = data
        data_cat = torch.cat((data_cat, gen_out3), 0)
        #


    print(name,weight.shape,data_cat.shape)
    if weight.size(0)>=16:
        if data_cat.size(0)>=16:
            data = util.shift_after_coe(data_cat[0:16 ,...], name + '_shift_00_16', weight[0:16 ,0:16,:,:], True)
        if data_cat.size(0)>=32:
            data = util.shift_after_coe(data_cat[16:32,...], name + '_shift_10_16', weight[0:16,16:32,:,:], False)
        if data_cat.size(0)>=48:
            data = util.shift_after_coe(data_cat[32:48,...], name + '_shift_20_16', weight[0:16,32:48,:,:], True)
        if data_cat.size(0)>=64:
            data = util.shift_after_coe(data_cat[48:64,...], name + '_shift_30_16', weight[0:16,48:64,:,:], True)

    if weight.size(0) >= 32:
        if data_cat.size(0)>=16:
            data = util.shift_after_coe(data_cat[0:16 ,...], name + '_shift_01_16', weight[16:32, 0:16,:,:], True)
        if data_cat.size(0)>=32:
            data = util.shift_after_coe(data_cat[16:32,...], name + '_shift_11_16', weight[16:32,16:32,:,:], True)
        if data_cat.size(0)>=48:
            data = util.shift_after_coe(data_cat[32:48,...], name + '_shift_21_16', weight[16:32,32:48,:,:], True)
        if data_cat.size(0)>=64:
            data = util.shift_after_coe(data_cat[48:64,...], name + '_shift_31_16', weight[16:32,48:64,:,:], True)
    if weight.size(0) >= 48:
        if data_cat.size(0)>=16:
            data = util.shift_after_coe(data_cat[0:16 ,...], name + '_shift_02_16', weight[32:48,  0:16,:,:], True)
        if data_cat.size(0)>=32:
            data = util.shift_after_coe(data_cat[16:32,...], name + '_shift_12_16', weight[32:48,16:32,:,:], True)
        if data_cat.size(0)>=48:
            data = util.shift_after_coe(data_cat[32:48,...], name + '_shift_22_16', weight[32:48,32:48,:,:], True)
        if data_cat.size(0)>=64:
            data = util.shift_after_coe(data_cat[48:64,...], name + '_shift_32_16', weight[32:48,48:64,:,:], True)
    if weight.size(0) >= 64:
        if data_cat.size(0)>=16:
            data = util.shift_after_coe(data_cat[0:16 ,...], name + '_shift_03_16', weight[48:64,  0:16,:,:], True)
        if data_cat.size(0)>=32:
            data = util.shift_after_coe(data_cat[16:32,...], name + '_shift_13_16', weight[48:64,16:32,:,:], True)
        if data_cat.size(0)>=48:
            data = util.shift_after_coe(data_cat[32:48,...], name + '_shift_23_16', weight[48:64,32:48,:,:], True)
        if data_cat.size(0)>=64:
            data = util.shift_after_coe(data_cat[48:64,...], name + '_shift_33_16', weight[48:64,48:64,:,:], True)

def gen_act_mid_data(input,compare,name,output):

    if input.size(1) >=16:

        print(name + '_16')
        util.act_after_coe(input[:,0:16,:,:],compare,name+'_16',output[:,0:16,:,:])
    if input.size(1) >=32:
        print(name + '_32')
        util.act_after_coe(input[:,16:32,:,:],compare,name+'_32',output[:,16:32,:,:])
    if input.size(1) >=64:
        print(name + '_48')
        util.act_after_coe(input[:,32:48,:,:],compare,name+'_48',output[:,32:48,:,:])
    if input.size(1) >=64:
        print(name + '_64')
        util.act_after_coe(input[:,48:64,:,:],compare,name+'_64',output[:,48:64,:,:])




if __name__ == '__main__':
    test_data = torch.zeros(16,32,32)
    count = 0;
    for i in range(32):
        for j in range(32):
            for k in range(16):
                test_data[k,i,j] = count
                count = count +1
                if count == 15:
                    count = 0

    # print(test_data)
    weight = torch.ones(16 ,16, 3, 3)


    gen_conv_mid_data(test_data[None,...],'test',weight,0)



def write_matrix_weight(input_matrix,filename):
    cout = input_matrix.shape[0]
    if(input_matrix.shape[1]<16):
        return
    bank_num = input_matrix.shape[1]//16

    # weight_matrix = input_matrix.reshape(cout,-1).transpose()
    # np.savetxt(filename, weight_matrix, delimiter=",",fmt='%10.5f')

class AdobeS9CimSim:
    def __init__(self):
        a = 1
        # self.model = model
        self.hook_handle_list = []
        self.models_conv = []
        self.model_name = 'resnet14'
        self.mid_data_hand = {}

    def pth_model_handle(self,model):
        model_name = self.model_name
        if not os.path.exists('./layer_record_' + str(model_name)):
            os.makedirs('./layer_record_' + str(model_name))
        self.layer_handle(model=model,name_pre=model_name)


    def layer_handle(self,model,name_pre):
        name_pre = name_pre + '_'

        for name, module in model._modules.items():
            if len(list(module.children())) > 0:
                # recurse
                self.layer_handle(model=module, name_pre=name_pre+name)
            if isinstance(module,nn.Conv2d):
                if(module.in_channels==3):
                    continue
                module.name = name_pre + name
                # self.models_conv.append(module)

                self.hook_handle_list.append(module.register_forward_hook(neural_sim_conv))
            # if isinstance(module,nn.BatchNorm2d):
            #     # resnet14_layer1_0_
            #
            #     module.name = name_pre+name
            #     if len(name_pre)>=10 :
            #
            #         self.hook_handle_list.append(module.register_forward_hook(neural_sim_bn))
            #         ...
            #
            # if isinstance(module,ir_1w8a.residuals):
            #     module.name = name_pre+name
            #     self.hook_handle_list.append(module.register_forward_hook(neural_sim_resi))
            #
            # if isinstance(module, ir_1w8a.act_hardtanh):
            #     module.name = name_pre + name
            #     self.hook_handle_list.append(module.register_forward_hook(neural_sim_act))
            #     ...

    def remove_hook_list(self):
        for handle in self.hook_handle_list:
            handle.remove()
        self.hook_handle_list = []

    # def static_handle(self):
    #     model_name = self.model_name
    #     if not os.path.exists('./layer_record_' + str(model_name)+'/conv'):
    #         os.makedirs('./layer_record_' + str(model_name)+'/conv')
    #     for conv in self.models_conv:
    #
    #         if conv.in_channels == 3:
    #             continue
    #         print(conv.name)
    #         print(conv.weight.shape)
    #         weight_q,bw = self.weight_quantize(conv)
    #         weight_file_name = './layer_record_' + str(model_name) +'/conv'+ '/weight' + str(conv.name) + '.coe'
    #         with open(weight_file_name, mode='w', encoding='utf-8') as file_obj:
    #             for i in range(len(weight_q)//8):
    #                 tmp = weight_q[i*8:(i+1)*8]
    #                 tmp_str = str(tmp)
    #                 tmp_str = tmp_str[1:-1]
    #                 tmp_str = tmp_str.replace(', ','')
    #                 binary = int(tmp_str,2)
    #                 hex_val = hex(binary)
    #                 file_obj.write(str(hex_val)+'\n')
    #
    # def weight_quantize(self,conv):
    #     w = conv.weight.cuda()
    #     bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
    #     bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
    #
    #     sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
    #                    (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
    #         bw.size(0), 1, 1, 1).detach()
    #
    #     bw = binaryfunction.BinaryQuantize().apply(bw, conv.k, conv.t)
    #     bank_resu = []
    #     for i in range(bw.shape[1]//16):
    #         for j in range(bw.shape[0]//16):
    #             tmp = bw[j*16:(j+1)*16,i*16:(i+1)*16]
    #             tmp = tmp.view(16,16,9)
    #             for x in range(tmp.shape[1]):
    #                 for j in range(tmp.shape[2]):
    #                     for k in range(tmp.shape[0]):
    #                         bank_resu.append(int(tmp[k,x,j]))
    #     bank_resu=[0 if i == -1 else i for i in bank_resu]
    #     #
    #     # for i in range(bw.shape[1]//16):
    #     #     for j in range(bw.shape[0]//16):
    #     #         tmp = bw[j*16:(j+1)*16,i*16:(i+1)*16,::,::]
    #     return bank_resu,bw
    #
    # def weight_q(self,weight,name):
    #     w = weight
    #     bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
    #     bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
    #
    #     sw = torch.pow(torch.tensor([2] * bw.size(0)).cuda().float(),
    #                    (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round().float()).view(
    #         bw.size(0), 1, 1, 1).detach()
    #
    #     bw = binaryfunction.BinaryQuantize().apply(bw, self.k, self.t)

