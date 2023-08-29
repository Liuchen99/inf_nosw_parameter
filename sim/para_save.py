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


def diff_base_save(lut_base, lut_diff, layer_idx):
    # lut_base
    base_h1 = ["" for x in range(64)]
    base_h = ["" for x in range(24)]
    for j in range(64):
        base_h1[j] = intToBin(int(lut_base[j]), 9, 1)
    for j in range(5):
        base_h[j] = base_h1[3 * j + 2] + base_h1[3 * j + 1] + base_h1[3 * j]
        base_h[j] = '0x' + hex(int(base_h[j], 2))[2:].rjust(8, '0')
    base_h[5] = '0x' + hex(int(base_h1[15], 2))[2:].rjust(8, '0')
    for j in range(5, 10):
        base_h[j + 1] = base_h1[3 * j + 3] + base_h1[3 * j + 2] + base_h1[3 * j + 1]
        base_h[j + 1] = '0x' + hex(int(base_h[j + 1], 2))[2:].rjust(8, '0')
    base_h[11] = '0x' + hex(int(base_h1[31], 2))[2:].rjust(8, '0')
    for j in range(10, 15):
        base_h[j + 2] = base_h1[3 * j + 4] + base_h1[3 * j + 3] + base_h1[3 * j + 2]
        base_h[j + 2] = '0x' + hex(int(base_h[j + 2], 2))[2:].rjust(8, '0')
    base_h[17] = '0x' + hex(int(base_h1[47], 2))[2:].rjust(8, '0')
    for j in range(15, 20):
        base_h[j + 3] = base_h1[3 * j + 5] + base_h1[3 * j + 4] + base_h1[3 * j + 3]
        base_h[j + 3] = '0x' + hex(int(base_h[j + 3], 2))[2:].rjust(8, '0')
    base_h[23] = '0x' + hex(int(base_h1[63], 2))[2:].rjust(8, '0')

    f = open("./para_resnet.txt", "a")
    f.write('\n//layer_idx is  ' + str(layer_idx + 1) + '\n' + '//base:' + '\n')
    for i in range(24):
        base_h[i] = base_h[i] + ',' + '\n'
        f.write(base_h[i])
    f.write('\n')
    f.close()

    # lut_diff
    diff_h1 = ["" for x in range(64)]
    diff_h = ["" for x in range(16)]
    for j in range(64):
        diff_h1[j] = intToBin(int(lut_diff[j]), 6, 1)
    for j in range(3):
        diff_h[j] = diff_h1[5 * j + 4] + diff_h1[5 * j + 3] + diff_h1[5 * j + 2] + diff_h1[5 * j + 1] + diff_h1[5 * j]
        diff_h[j] = '0x' + hex(int(diff_h[j], 2))[2:].rjust(8, '0')
    diff_h[3] = '0x' + hex(int(diff_h1[15], 2))[2:].rjust(8, '0')
    for j in range(3, 6):
        diff_h[j + 1] = diff_h1[5 * j + 5] + diff_h1[5 * j + 4] + diff_h1[5 * j + 3] + diff_h1[5 * j + 2] + diff_h1[
            5 * j + 1]
        diff_h[j + 1] = '0x' + hex(int(diff_h[j + 1], 2))[2:].rjust(8, '0')
    diff_h[7] = '0x' + hex(int(diff_h1[31], 2))[2:].rjust(8, '0')
    for j in range(6, 9):
        diff_h[j + 2] = diff_h1[5 * j + 6] + diff_h1[5 * j + 5] + diff_h1[5 * j + 4] + diff_h1[5 * j + 3] + diff_h1[
            5 * j + 2]
        diff_h[j + 2] = '0x' + hex(int(diff_h[j + 2], 2))[2:].rjust(8, '0')
    diff_h[11] = '0x' + hex(int(diff_h1[47], 2))[2:].rjust(8, '0')
    for j in range(9, 12):
        diff_h[j + 3] = diff_h1[5 * j + 7] + diff_h1[5 * j + 6] + diff_h1[5 * j + 5] + diff_h1[5 * j + 4] + diff_h1[
            5 * j + 3]
        diff_h[j + 3] = '0x' + hex(int(diff_h[j + 3], 2))[2:].rjust(8, '0')
    diff_h[15] = '0x' + hex(int(diff_h1[63], 2))[2:].rjust(8, '0')

    f = open("./para_resnet.txt", "a")
    f.write('//diff:' + '\n')
    for i in range(16):
        if i < 15:
            diff_h[i] = diff_h[i] + ',' + '\n'
        f.write(diff_h[i])
    f.write('\n')
    f.close()
def lut_all_save(lut, layer_idx):
    # lut_base
    base_h1 = ["" for x in range(64)]
    base_h = ["" for x in range(24)]
    for j in range(64):
        base_h1[j] = intToBin(int(lut[j]), 9, 1)
    for j in range(5):
        base_h[j] = base_h1[3 * j + 2] + base_h1[3 * j + 1] + base_h1[3 * j]
        base_h[j] = '0x' + hex(int(base_h[j], 2))[2:].rjust(8, '0')
    base_h[5] = '0x' + hex(int(base_h1[15], 2))[2:].rjust(8, '0')
    for j in range(5, 10):
        base_h[j + 1] = base_h1[3 * j + 3] + base_h1[3 * j + 2] + base_h1[3 * j + 1]
        base_h[j + 1] = '0x' + hex(int(base_h[j + 1], 2))[2:].rjust(8, '0')
    base_h[11] = '0x' + hex(int(base_h1[31], 2))[2:].rjust(8, '0')
    for j in range(10, 15):
        base_h[j + 2] = base_h1[3 * j + 4] + base_h1[3 * j + 3] + base_h1[3 * j + 2]
        base_h[j + 2] = '0x' + hex(int(base_h[j + 2], 2))[2:].rjust(8, '0')
    base_h[17] = '0x' + hex(int(base_h1[47], 2))[2:].rjust(8, '0')
    for j in range(15, 20):
        base_h[j + 3] = base_h1[3 * j + 5] + base_h1[3 * j + 4] + base_h1[3 * j + 3]
        base_h[j + 3] = '0x' + hex(int(base_h[j + 3], 2))[2:].rjust(8, '0')
    base_h[23] = '0x' + hex(int(base_h1[63], 2))[2:].rjust(8, '0')

    f = open("./para_resnet.txt", "a")
    f.write('\n//layer_idx is  ' + str(layer_idx + 1) + '\n' + '//base:' + '\n')
    for i in range(24):
        base_h[i] = base_h[i] + ',' + '\n'
        f.write(base_h[i])
    f.write('\n')
    f.close()
