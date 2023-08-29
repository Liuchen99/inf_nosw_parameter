import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

torch.set_printoptions(precision=20, sci_mode=False)


def intToBin(number, index, feature=True):
    """index为该数据位宽,number为待转换数据,
    feature为True则进行十进制转二进制，为False则进行二进制转十进制。"""

    if feature:  # 十进制转换为二进制
        if (number >= 0):
            b = bin(number)
            b = '0' * (index + 2 - len(b)) + b
        else:
            b = 2 ** (index) + number
            b = bin(b)
            b = '1' * (index + 2 - len(b)) + b  # 注意这里算出来的结果是补码
        b = b.replace("0b", '')
        b = b.replace('-', '')

        return b
    elif feature == False:  # 二进制转换为十进制
        i = int(str(number), 2)
        if i >= 2 ** (index - 1):  # 如果是负数
            i = -(2 ** index - i)
            return i
        else:
            return i


def lut_all_save(lut, layer_idx):
    base_h1 = ["" for x in range(448)]
    base_h = ["" for x in range(152)]
    for j in range(448):
        base_h1[j] = intToBin(int(lut[j]), 10, 1)
    for j in range(37):
        base_h[j] = base_h1[3 * j + 2] + base_h1[3 * j + 1] + base_h1[3 * j]
        base_h[j] = '0x' + hex(int(base_h[j], 2))[2:].rjust(8, '0')
    base_h[37] = '0x' + hex(int(base_h1[111], 2))[2:].rjust(8, '0')
    for j in range(37, 74):
        base_h[j + 1] = base_h1[3 * j + 3] + base_h1[3 * j + 2] + base_h1[3 * j + 1]
        base_h[j + 1] = '0x' + hex(int(base_h[j + 1], 2))[2:].rjust(8, '0')
    base_h[75] = '0x' + hex(int(base_h1[223], 2))[2:].rjust(8, '0')
    for j in range(74, 111):
        base_h[j + 2] = base_h1[3 * j + 4] + base_h1[3 * j + 3] + base_h1[3 * j + 2]
        base_h[j + 2] = '0x' + hex(int(base_h[j + 2], 2))[2:].rjust(8, '0')
    base_h[113] = '0x' + hex(int(base_h1[335], 2))[2:].rjust(8, '0')
    for j in range(111, 148):
        base_h[j + 3] = base_h1[3 * j + 5] + base_h1[3 * j + 4] + base_h1[3 * j + 3]
        base_h[j + 3] = '0x' + hex(int(base_h[j + 3], 2))[2:].rjust(8, '0')
    base_h[151] = '0x' + hex(int(base_h1[447], 2))[2:].rjust(8, '0')

    f = open("./para_resnet.txt", "a")
    f.write('\n//layer_idx is  ' + str(layer_idx + 1) + '\n')
    for i in range(152):
        base_h[i] = base_h[i] + ',' + '\n'
        f.write(base_h[i])
    f.write('\n')
    f.close()
