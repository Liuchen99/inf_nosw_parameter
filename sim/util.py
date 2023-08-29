import numpy as np
import torch



class util():

    @staticmethod
    def convert_hex(data):
        resu = []

        for x in range(data.size(1)):
            for y in range(data.size(2)):
                str = ''
                for z in range(data.size(0)):

                    val = data[z,x, y]
                    val_int = int(val.round())
                    val_int = val_int &(15)
                    val_hex = hex(val_int)
                    val_hex = val_hex[-1:]
                    str = str + val_hex
                resu.append(str)
        return resu

    @staticmethod
    def gen_convert_hex(data):
        resu = []

        for x in range(data.size(1)):
            for y in range(data.size(2)):
                for i in range(data.size(3)):
                    for j in range(data.size(4)):
                        str = ''
                        for z in range(data.size(0)):
                            val = data[z,x,y,i,j]
                            val_int = int(val.round())
                            val_int = val_int &(15)
                            val_hex = hex(val_int)
                            val_hex = val_hex[-1:]
                            str = str + val_hex
                        resu.append(str)
        return resu

    @staticmethod
    def sender_convert_hex(data):
        resu = []
        for x in range(data.size(1)):
            for y in range(data.size(2)):
                for point in range(data.size(3)):
                    str = ''
                    for z in range(data.size(0)):
                        for i in range(data.size(4)):
                            for j in range(data.size(5)):
                                val = data[z,x,y,point,i,j]
                                val_int = int(val.round())
                                val_hex = hex(val_int)
                                val_hex = val_hex[-1:]
                                str = val_hex + str

                    val_2_int = int(str,2)

                    val_2_hex = hex(val_2_int)[2:].rjust(36,'0')
                    resu.append(val_2_hex)
        return resu

    @staticmethod
    def adc_out_hex(data):
        resu = []
        for heigh in range(data.size(1)):
            for width in range(data.size(2)):
                for clk in range(data.size(3)):
                    for bank in range(data.size(4)):
                        str = ''
                        for channel in range(data.size(0)):
                            val = data[channel, heigh, width, clk, bank]

                            val_int = int(val.round())&0xFF
                            val_hex = hex(val_int)

                            val_hex = val_hex[2:].rjust(2,'0')

                            str=val_hex+str

                        resu.append(str)



        return resu

    @staticmethod
    def shift_out_hex(data):
        resu = []

        for heigh in range(data.size(1)):
            for width in range(data.size(2)):
                str = ''
                for channel in range(data.size(0)):
                    val = data[channel,heigh,width]

                    val_int = (int(val)) & 0xFFFF
                    val_hex = hex(val_int)
                    val_hex = val_hex[2:].rjust(4, '0')
                    str = str+val_hex

                resu.append(str)
        return resu

    @staticmethod
    def padd_after_coe(data,name,write_file):
        new_data = torch.zeros(data.size(0),data.size(1)+2,data.size(2)+2)

        new_data[:,1:-1,1:-1] = data

        if write_file:
            resu = util.convert_hex(new_data)

            file_path = 'sim/mid_data/'+name+".coe"
            with open(file_path, mode='w', encoding='utf-8') as file_obj:
                for i in resu:
                    file_obj.write(i[::-1]+'\n')
        return new_data
        #
        #     for i in len(resu)
        # return data_to_next_layer

    def act_after_coe(input,compare,name,output):
        resu = torch.zeros_like(input)
        for i in range(input.size(2)):
            for j in range(input.size(3)):
                for c in range(input.size(1)):
                    for left, right, value in compare:
                        if input[0, c, i, j] >= left and input[0, c, i, j] <= right:
                            resu[0, c, i, j] = value
                        # if i==0 and j==1 and c ==7 and input[0, c, i, j] >= left and input[0, c, i, j] <= right:
                        #     print(input[0, c, i, j],left,right,value)

        resu_str = []
        for i in range(input.size(2)):
            for j in range(input.size(3)):
                str = ''
                flag = 0
                for c in range(input.size(1)):
                    val = resu[0, c, i, j]
                    val_int = (int(val.round())) & 0xF
                    if int(val.round()) != int(output[0, c, i, j].round()):
                        flag = 1
                    val_hex = hex(val_int)
                    val_hex = val_hex[2:].rjust(1, '0')
                    str = val_hex+str
                resu_str.append(str)
                if(flag == 1):
                    print(resu_str[len(resu_str)-1])
                    flag = 0

        file_path = 'sim/mid_data/' + name + ".coe"
        with open(file_path, mode='a', encoding='utf-8') as file_obj:
            for i in resu_str:
                file_obj.write(i + '\n')




    @staticmethod
    def gen_after_coe(data,name,write_file):
        new_data = torch.zeros(data.size(0),data.size(1)-2,data.size(2)-2,3,3)
        for i in range(data.size(1)-2):
            for j in range(data.size(2)-2):
                for k in range(data.size(0)):
                    new_data[k,i,j,...]=data[k,i:i+3,j:j+3]



        if write_file:
            resu = util.gen_convert_hex(new_data)

            file_path = 'sim/mid_data/'+name+".coe"
            with open(file_path, mode='w', encoding='utf-8') as file_obj:
                for i in resu:
                    file_obj.write(i[::-1]+'\n')
        return new_data

    @staticmethod
    def sender_after_coe(data,name,wirte_file):
        new_data = torch.zeros(data.size(0),data.size(1),data.size(2),4,3,3)

        for i in range(data.size(1)):
            for j in range(data.size(2)):
                for k in range(data.size(0)):
                    for x in range(4):
                        new_data[k,i,j,x,...]=(data[k,i,j,...].int()>>x)&1

        if wirte_file:
            resu = util.sender_convert_hex(new_data)
            file_path = 'sim/mid_data/'+name+".coe"
            with open(file_path, mode='w', encoding='utf-8') as file_obj:
                for i in resu:
                    file_obj.write(i+'\n')

        return new_data

    @staticmethod
    def shift_after_coe(data,name,weight,wirte_file):
        #            通道   尺寸   bit 33卷积
        #torch.Size([16, 32, 32, 4, 3, 3])

        #                      输出通道             尺寸                 时钟    bank
        if wirte_file == False:
            print(weight)
            exit()
        adc_out = torch.zeros(data.size(0),data.size(1),data.size(2),4,    weight.size(0)//16 )

        for i in range(adc_out.size(1)):
            for j in range(adc_out.size(2)):
                for k in range(adc_out.size(0)):
                    # 时钟
                    for x in range(adc_out.size(3)):
                        # bank
                        for y in range(adc_out.size(4)):
                            adc_out[k, i, j, x, y] = (data[y*16:(y+1)*16,i,j,x] * weight[k,:,:,:].cpu()).sum()

        adc_out = torch.clamp(adc_out,-128,127)
        adc_coe = util.adc_out_hex(adc_out)
        file_path = 'sim/mid_data/'+name+"_adc.coe"
        with open(file_path, mode='w', encoding='utf-8') as file_obj:
            for i in adc_coe:
                file_obj.write(i+'\n')

        shift_mid = adc_out.sum(4)
        shift_out = shift_mid[:,:,:,0] + shift_mid[:,:,:,1] * 2 + shift_mid[:,:,:,2] * 4 + shift_mid[:,:,:,3] * -8

        shift_resu = util.shift_out_hex(shift_out)
        file_path = 'sim/mid_data/'+name+"_shift.coe"
        with open(file_path, mode='w', encoding='utf-8') as file_obj:
            for i in shift_resu:
                file_obj.write(i+'\n')
        return shift_out
        # print(adc_out)



        # resu = util.sender_convert_hex(new_data)
        # file_path = 'mid_data/'+name+".coe"
        # with open(file_path, mode='w', encoding='utf-8') as file_obj:
        #     for i in resu:
        #         file_obj.write(i+'\n')
        # return new_data
    def bn_conv_coe(name,data):


        ...

    def shift_after_bn_coe(channel1,name,bw,sw,bb,output):
        # sw = torch.ones_like(sw)

        resu = torch.zeros_like(channel1).cpu()
        resu_sw = torch.zeros_like(channel1).cpu()
        for i in range(channel1.size(1)):
            for j in range(channel1.size(2)):
                for k in range(channel1.size(3)):
                    # print((int(channel1[:,i,k,j]) & 0x7 == 0x6 or  int(channel1[:,i,k,j]) & 0x3 == 3),(sw[0,i,0,0] == 0.5))
                    # exit()

                    if (int(channel1[:,i,k,j]) & 0x7 == 0x6 or  int(channel1[:,i,k,j]) & 0x3 == 3):
                        if sw[0,i,0,0] == 0.5:
                            resu[:,i,k,j]    = ((((int(channel1[:,i,k,j])>>2) + 1) )>>1)*bw[i].round()+bb[i].round()
                            resu_sw[:, i, k, j] = ((((int(channel1[:, i, k, j]) >> 2) + 1)) >> 1)
                        else :
                            resu[:, i, k, j] = ((((int(channel1[:,i,k,j])>> 2) + 1)   ))*bw[i].round()+bb[i].round()
                            resu_sw[:, i, k, j] = ((((int(channel1[:,i,k,j])>> 2) + 1)   ))
                    else :
                        if sw[0,i,0,0] == 0.5:
                            resu[:,i,k,j]    = (((int(channel1[:,i,k,j])>> 2)    )>>1)*bw[i].round()+bb[i].round()
                            resu_sw[:, i, k, j] =(((int(channel1[:,i,k,j])>> 2)    )>>1)
                        else :
                            resu[:, i, k, j] = (((int(channel1[:,i,k,j])>> 2)    ))*bw[i].round()+bb[i].round()
                            resu_sw[:, i, k, j] = (((int(channel1[:,i,k,j])>> 2)    ))
        if resu.eq(output).all() is False:
            print("检测到了错误",name)
            exit()

        file_path = 'sim/mid_data/'+name+"_sw.coe"
        with open(file_path, mode='w', encoding='utf-8') as file_obj:
            #     for i in resu:
            #         file_obj.write(i+'\n')
            # return new_data
            for i in range(resu.size(2)):
                for j in range(resu.size(3)):
                    str = ''
                    for k in range(resu.size(1)):
                        val = resu_sw[0,k,i,j]
                        val_int = (int(val.round())) & 0xFFFF
                        val_hex = hex(val_int)
                        val_hex = val_hex[2:].rjust(4, '0')
                        str = val_hex+str
                    file_obj.write(str + '\n')



        file_path = 'sim/mid_data/'+name+"_bn.coe"
        with open(file_path, mode='w', encoding='utf-8') as file_obj:
            #     for i in resu:
            #         file_obj.write(i+'\n')
            # return new_data
            for i in range(resu.size(2)):
                for j in range(resu.size(3)):
                    str = ''
                    for k in range(resu.size(1)):
                        val = resu[0,k,i,j]

                        val_int = (int(val.round())) & 0xFFFF

                        val_hex = hex(val_int)

                        val_hex = val_hex[2:].rjust(4, '0')
                        str = val_hex+str
                    file_obj.write(str + '\n')
            # with open(file_path, mode='w', encoding='utf-8') as file_obj:
            #     for i in resu:
            #         file_obj.write(i+'\n')
            # return new_data
        # print(resu)
        # print(sw)
        #     val = data[channel, heigh, width]

    def resi_after_nochange_coe(data,resi,para,name):
        mid_data = torch.zeros_like(data)

        mid_data = resi * para + data

        resu = []

        for i in range(mid_data.size(2)):
            for j in range(mid_data.size(3)):
                str = ''
                for c in range(mid_data.size(1)):
                    val = mid_data[0,c,i,j]
                    val_int = (int(val.round())) & 0xFFFFF
                    val_hex = hex(val_int)
                    val_hex = val_hex[2:].rjust(5, '0')
                    str =  val_hex + str
                resu.append(str)

        file_path = 'sim/mid_data/'+name+".coe"
        with open(file_path, mode='w', encoding='utf-8') as file_obj:
            for i in resu:
                file_obj.write(i+'\n')

    def resi_after_change_coe(data,resi,para,name,output):

        mid_data = torch.zeros_like(data)
        # print((resi.transpose(1, 3).transpose(1, 2))[0, 1, :, :].round())
        # print((data.transpose(1, 3).transpose(1, 2))[0, 1, :, :].round())
        # print((mid_data.transpose(1, 3).transpose(1, 2))[0, 1, :, :].round())
        # print(para)
        # exit()

        mid_data = resi.cuda() * para.cuda() + data
        # print(name)
        #
        # print(mid_data[0, 0:16, 1, 1].round())
        # print((mid_data.transpose(1,3).transpose(1,2))[0, 1,:,:].round())
        # # print(output[0, 0:16, 0, :].round())
        # exit()
        resu = []
        for i in range(mid_data.size(2)):
            for j in range(mid_data.size(3)):
                str = ''
                for c in range(mid_data.size(1)):
                    val = mid_data[0,c,i,j]
                    val_int = (int(val.round())) & 0xFFFFF
                    val_hex = hex(val_int)
                    val_hex = val_hex[2:].rjust(5, '0')
                    str = val_hex+str
                resu.append(str)
        file_path = 'sim/mid_data/'+name+".coe"
        with open(file_path, mode='w', encoding='utf-8') as file_obj:
            for i in resu:
                file_obj.write(i+'\n')



def convert_coe(data, data_bit, row_bit):
    one_row_data = row_bit // data_bit
    resu = []
    mask = 0xF
    if (data_bit == 8):
        mask = 0xFF
    for i in range(len(data) // one_row_data):
        tmp = data[i * one_row_data:(i + 1) * one_row_data].cpu()
        str = ''

        for i in tmp:
            # if i.abs()>15 and data_bit==4 or i.abs()>64 and data_bit==8:
            #     print(i.abs())
            #     raise IndexError
            val = int(i) & mask
            val_hex = hex(val)
            val_hex = val_hex[2:].rjust(data_bit // 4, '0')
            str = val_hex + str
        resu.append(str)
    return resu


def bin32ToInt(self, s):
    return int(s[1:], 2) - int(s[0]) * (1 << 31)


def dim4_to_str(data):
    len = data.size(3)
    resu = torch.zeros(data.size(0), data.size(1), data.size(2))
    resu_list = []
    for i in range(data.size(0)):
        row = []
        for j in range(data.size(1)):
            col = []
            for k in range(data.size(2)):
                one_data = ''
                for x in range(data.size(3)):
                    one_data = one_data + str(int(data[i, j, k, x]))
                col.append(one_data)
            row.append(col)
        resu_list.append(row)
    return (resu_list)




# def image2coe(image, name=1):
#     data_to_next_layer = image
#     row = image.shape[2]
#     channel = image.shape[1]
#     image = image.transpose(1, 3)
#     image = image.contiguous().view(-1)
#     resu = convert_coe(image, 4, 64)
#     file_path = 'mid_data/pad_before_' + str(channel) + '_' + str(row) + '_' + str(row) + '_channel_' + str(
#         name) + '_sim.coe'
#     with open(file_path, mode='w', encoding='utf-8') as file_obj:
#         for i in resu:
#             file_obj.write(i + '\n')
#     return data_to_next_layer
#
#
# def padd_after_coe(data, name):
#     row = data.shape[2]
#     channel = data.shape[1]
#
#     input_q = torch.zeros(data.size(0), data.size(1), data.size(2) + 2, data.size(3) + 2)
#     input_q[:, :, 1:-1, 1:-1] = data
#     image = input_q.transpose(1, 3)
#     image = image.contiguous().view(-1)
#
#     resu = convert_coe(image, 4, 64)
#     file_path = 'mid_data/pad_after_' + str(channel) + '_' + str(row) + '_' + str(row) + '_channel_' + str(
#         name) + '_sim.coe'
#     with open(file_path, mode='w', encoding='utf-8') as file_obj:
#         for i in resu:
#             file_obj.write(i + '\n')
#     return input_q
#
#
# def gen_after_coe(data, name):
#     row = data.shape[2]
#     channel = data.shape[1]
#
#     input_q_33 = torch.empty(data.shape[0], data.shape[1], row - 2, row - 2, 9, dtype=torch.int8).cuda()
#     for i in range(row - 2):
#         for j in range(row - 2):
#             input_q_33[:, :, i, j, :] = data[:, :, i:i + 3, j:j + 3].transpose(2, 3).contiguous().view(data.size(0),
#                                                                                                        data.size(1), 9)
#
#     input_q_33 = input_q_33.transpose(1, 3)
#     input_q_33 = input_q_33.transpose(3, 4)
#     resu_33 = input_q_33
#     input_q_33 = input_q_33.contiguous().view(-1)
#     resu = convert_coe(input_q_33, 4, 64)
#     file_path = 'mid_data/gen_after_' + str(channel) + '_' + str(row) + '_' + str(row) + '_channel_' + str(
#         name) + '_sim.coe'
#     with open(file_path, mode='w', encoding='utf-8') as file_obj:
#         for i in resu:
#             file_obj.write(i + '\n')
#     return resu_33
#
#
# def sender_after_coe(data, name):
#     row = data.shape[2]
#     channel = data.shape[1]
#
#     data = data.contiguous().view(data.size(0), data.size(1), data.size(2), 9 * 16)
#
#     print(data.shape)
#
#     one = data % 2
#     two = (data - one) // 2 % 2
#     three = (data - one - 2 * two) // 4 % 2
#     four = (data - one - 2 * two - 4 * three) // 8 % 2
#
#     one_str = dim4_to_str(one)
#     two_str = dim4_to_str(two)
#     three_str = dim4_to_str(three)
#     four_str = dim4_to_str(four)
#     exit()
#     resu1 = convert_coe(one.view(-1), 144, 144)
#     print(resu1)
#     return resu1
#
#
# def shift_add_after_coe(data, name):
#     data_to_next_layer = data
#     row = data.shape[2]
#     channel = data.shape[1]
#     image = data.transpose(1, 3)
#     image = image.contiguous().view(-1)
#     resu = convert_coe(image, 8, 128)
#     file_path = 'mid_data/shift_add_' + str(channel) + '_' + str(row) + '_' + str(row) + '_channel_' + str(
#         name) + '_sim.coe'
#     with open(file_path, mode='w', encoding='utf-8') as file_obj:
#         for i in resu:
#             file_obj.write(i + '\n')
#     return data_to_next_layer
# img = torch.arange(0,32*32*16)
# img = img % 16
# img = img.view(-1)
# print(img)
# resu = convert_coe(img,4,64)
# print(len(resu))
