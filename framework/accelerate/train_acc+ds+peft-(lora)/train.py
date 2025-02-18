
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed, DummyOptim, DummyScheduler

# 简单的模型
def get_model():
    return nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )

# 生成随机数据
def get_dataloader():
    x = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=32, shuffle=True)



# 训练逻辑
def train():
    # 1 实例化Accelerator
    accelerator = Accelerator()
    accelerator.print("Accelerator Configuration:")
    accelerator.print(accelerator.state)

    is_main_process = accelerator.is_local_main_process
    global_rank = dist.get_rank()
    num_processes = dist.get_world_size()
    seed = config.run.seed + global_rank
    set_seed(seed)

    device = accelerator.device
    
    model = get_model()

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # 优化器设置--如果deepspeed中配置了优化器，acclerator中需要先设置一个虚拟优化器占位
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    optimizer_cls = (
        torch.optim.AdamW
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(trainable_params, lr=config.optimizer.lr)

    # 调度器设置--如果deepspeed中配置了学习率调度器，acclerator中需要先设置一个虚拟调度器占位
    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `args.lr_scheduler_type` Scheduler
    if (accelerator.state.deepspeed_plugin is None
        or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config):
        # Scheduler
        lr_scheduler = get_scheduler(
            config.optimizer.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=config.optimizer.lr_warmup_steps,
            num_training_steps=config.run.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=config.run.max_train_steps, warmup_num_steps=100)
        

    loss_fn = nn.MSELoss()
    train_dataloader = get_dataloader()
    
    # 2 加速器包装-传参顺序可随机，与参数接收对应即可
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)
    
    # 恢复训练
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)   # 与accelerator.save_state(output_dir)成对使用
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        path = os.path.basename(args.resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // num_update_steps_per_epoch
            resume_step -= starting_epoch * num_update_steps_per_epoch
            completed_steps = resume_step


    model.train()
    for epoch in range(5):
        total_loss = 0
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()

            # 3 使用accelerator进行混合精度训练
            with accelerator.autocast():
                outputs = model(x_batch.to(device))

            loss = loss_fn(outputs, y_batch.to(device))
            
            # 4 使用 accelerator 处理梯度
            accelerator.backward(loss)

            optimizer.step()
            total_loss += loss.item()

            # 保存训练状态、权重、以及accelerator.register_for_checkpointing(my_scheduler)方法注册的类的状态
            accelerator.save_state(output_dir)  # 若使用了deepspeed保存的是deepspeed格式的权重，且应该在所有进程中都执行，deepspeed格式可通过Deepspeed的脚本直接转换为torch的权重


        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")

if __name__ == "__main__":
    train()

