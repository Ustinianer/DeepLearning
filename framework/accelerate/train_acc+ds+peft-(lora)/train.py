
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed, DummyOptim, DummyScheduler

# from simple_lora import add_lora
from peft import LoraConfig, get_peft_model, PeftModel


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

    # Injecting LoRA layers into unet
    if config.lora.use_lora:
        if config.lora.resume_lora_path:
            model = PeftModel.from_pretrained(model, config.lora.resume_lora_path, is_trainable=True)
            resume_global_step = torch.load(os.path.join(config.lora.resume_lora_path, "state_dict.pt"), map_location=device)["global_step"]
        else:
            lora_config = LoraConfig(
                r=config.lora.r,                      
                lora_alpha=config.lora.lora_alpha,            
                target_modules=list(config.lora.target_modules),  
                bias=config.lora.bias,
                lora_dropout=config.lora.lora_dropout,
                task_type=config.lora.task_type,      
            )
            model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        # # 自定义lora
        # add_lora(model)
        # print(model) if is_main_process else print("lora added")
    else:
        model.requires_grad_(True)
        trainable_params = list(model.parameters())

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

            # 保存权重 使用 accelerate + Deepspeed + PEFT 保存权重方式
            # 方式一（可恢复训练）：使用accelerator.save_state(model_save_path) 统一保存，包括完整模型、优化器、学习率调度器等
            #    保存的为Deepspeed模式的权重，可使用Deepspeed提供的zero_to_fp32.py脚本转为peft包装后的torch权重
            #    转换思路 Deepspeed -> peft -> 原model.pth
            # 方式二（无法恢复训练）：先解包到peft模型，然后手动保存lora权重、优化器和学习率调度器状态，accelerte没有提供优化器状态加载接口，实际上无法恢复优化器状态，故只需保存lora权重即可舍弃训练状态
            #   1 使用 accelerator.unwrap_model(model) 获取PEFT模型
            #   2 使用 accelerator.get_state_dict(optimizer) 获取优化器状态  （无法恢复优化器状态）
            #   3 使用 lr_scheduler.state_dict() 获取学习率调度器状态  
            if global_step % config.ckpt.save_ckpt_steps == 0:
                accelerator.wait_for_everyone()  # 确保所有进程同步
                if is_main_process:
                    # 只保存lora权重
                    model_save_path = os.path.join(output_dir, f"checkpoints/checkpoint-{global_step}")
                    lora_model = accelerator.unwrap_model(unet)  # unwrap DeepspeedEngine aimed to get peft model (返回的是一个新的引用，不会影响训练)
                    lora_model.save_pretrained(model_save_path, save_adapter=True)   

                    if hasattr(unet, 'lr_scheduler'):
                        lr_scheduler_state = unet.lr_scheduler.state_dict()  # 从deepspeed模型中获取学习率调度器状态
                    elif hasattr(unet, 'optimizer') and hasattr(unet.optimizer, 'lr_scheduler'):
                        lr_scheduler_state = unet.optimizer.lr_scheduler.state_dict()  # 从deepspeed模型的optimizer中获取学习率调度器状态

                    
                    # 保存优化器和学习率调度器状态
                    torch.save(
                        {
                            "global_step": global_step,
                            "optimizer": accelerator.get_state_dict(optimizer),  # DeepSpeed 兼容
                            "lr_scheduler": lr_scheduler_state,
                        },
                        os.path.join(model_save_path, "state_dict.pt")
                    )

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}")

if __name__ == "__main__":
    train()

