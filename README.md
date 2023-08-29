# cmds

```python
# train FP32
python train.py --opt=options/experiment_resnet14_lut/resnet14_fp32_64ki_cifar10.yml

# train 1w4a; requires pretrained FP32 model
python train.py --opt=options/experiment_resnet14_lut/resnet14_1w4a_64ki_cifar10.yml

# test 1w4a; BN->LUT
python test.py --opt=options/experiment_resnet14_lut/resnet14_1w4a_LUT_64ki_cifar10.yml 
```