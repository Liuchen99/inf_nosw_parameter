# 2023/6/21
# 此文件用来提取ResNet14中间12层卷积层的权值，一键生成c数组。
# 1. 运行程序
# 2. 将每一个数组最后的,}替换为}即可
import os


def findAllFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f


base = 'sim/weight/'
cnt = 0
for i in findAllFile(base):
    if cnt < 4:
        print('uint8_t weight_layer' + str(cnt // 2 + 1) + '_conv' + str(cnt % 2 + 1) + '[288] = {')
    if 4 <= cnt <= 5:
        print('uint8_t weight_layer3_conv1_bank' + str(cnt % 2) + '[288] = {')
    if 6 <= cnt <= 9:
        print('uint8_t weight_layer3_conv2_bank' + str((cnt + 2) % 4) + '[288] = {')
    if 10 <= cnt <= 13:
        print('uint8_t weight_layer4_conv1_bank' + str((cnt + 2) % 4) + '[288] = {')
    if 14 <= cnt <= 17:
        print('uint8_t weight_layer4_conv2_bank' + str((cnt + 2) % 4) + '[288] = {')
    if 18 <= cnt <= 25:
        print('uint8_t weight_layer5_conv1_bank' + str((cnt - 2) % 8) + '[288] = {')
    if 26 <= cnt <= 41:
        print('uint8_t weight_layer5_conv2_bank' + str((cnt - 10) % 16) + '[288] = {')
    if 42 <= cnt <= 57:
        print('uint8_t weight_layer6_conv1_bank' + str((cnt - 10) % 16) + '[288] = {')
    if 58 <= cnt <= 73:
        print('uint8_t weight_layer6_conv2_bank' + str((cnt - 10) % 16) + '[288] = {')
    cnt += 1

    with open('sim/weight/' + i, "r", encoding='utf-8') as f:  # 打开文本
        print(f.read())
    #
    print('};')
