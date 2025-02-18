# 使用deepspeed实现单机多卡分布式训练
1 启动脚本：train__ds.sh

2 deepspeed的配置文件：ds_config.json


优化器、学习率调度器可直接在Deepspeed配置文件中指定由Deepspeed生成
也可以直接将数据集交给Deepspeed管理，Deepspeed会自动包装DDP的dataloader

参考：

DeepSpeed：https://github.com/deepspeedai/DeepSpeed

DeepSpeed Examples: https://github.com/deepspeedai/DeepSpeedExamples

https://github.com/deepspeedai/DeepSpeedExamples/blob/master/training/cifar/cifar10_deepspeed.py
