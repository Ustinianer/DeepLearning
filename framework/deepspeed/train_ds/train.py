
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist

import deepspeed
from deepspeed.accelerator import get_accelerator

# 简单的模型
def get_model():
    return nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )


def get_dataset():
    x = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(x, y)
    return dataset

# 生成随机数据
def get_dataloader(dataset):
    
    return DataLoader(dataset, batch_size=32, shuffle=True)



# 训练逻辑
def train():
    # 1 Deepspeed初始化
    deepspeed.init_distributed()  # 初始化分布式训练
    local_rank = args.local_rank  # 从命令行参数中读取 local_rank
    get_accelerator().set_device(local_rank)  # 设置设备

    is_main_process = accelerator.is_local_main_process
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    seed = config.run.seed + global_rank
    set_seed(seed)

    device = torch.device(local_rank)
    
    model = get_model()

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    loss_fn = nn.MSELoss()
    
    train_dataset = get_dataset()

    # 2 Deepspeed
    model_engine, optimizer, train_dataloader, __ = deepspeed.initialize(
                                                                            args=args,
                                                                            model=model,
                                                                            model_parameters=trainable_params,
                                                                            training_data=train_dataset,
                                                                            # config=ds_config,
                                                                        )
    
    # 恢复训练
    if args.resume_ckpt_dir and os.path.exists(args.resume_ckpt_dir): # deepspeed恢复训练
        logger.info("===== deepspeed resume training from checkpoint: {} =====".format(args.resume_ckpt_dir))
        _, client_state = model_engine.load_checkpoint(load_dir=args.resume_ckpt_dir)
        checkpoint_step = client_state['checkpoint_step']
        resume_global_step = client_state['global_step']

    device = model_engine.device

    model.train()
    for epoch in range(5):
        total_loss = 0
        for x_batch, y_batch in train_dataloader:
            
            # deepspeed会根据配置文件自动进行混合精度训练
            # outputs = model(x_batch.to(device))
            outputs = model_engine(x_batch.to(device))

            loss = loss_fn(outputs, y_batch.to(device))
            
            # 更新梯度
            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()

            # 保存训练状态、权重 应该在所有进程中都执行，deepspeed格式可通过Deepspeed的脚本直接转换为torch的权重
            if global_step % config.ckpt.save_ckpt_steps == 0:
                exp_dir = os.path.join(output_dir, f"checkpoints/global_step-{global_step}")
                print(f"Saving checkpoint at step {global_step}: {output_dir}\n{exp_dir}")
                model_engine.save_checkpoint(save_dir=exp_dir,
                                             client_state={ 
                                                            "global_step": global_step, 
                                                            'checkpoint_step': step
                                                            })  


        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Config file path
    parser.add_argument("--local_rank", type=int, default=0, help="Local rank for distributed training")
    # 添加Deepspeed的配置参数
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    train()

