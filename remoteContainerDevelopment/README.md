# 深度学习环境搭建与远程开发

## 1. 参考文章
**文章：** [从硬件到软件零起步搭建深度学习环境 | 配置深度学习环境: Docker + Conda + Pytorch + SSH + VS Code - 知乎](https://zhuanlan.zhihu.com/p/xxxxxx)

**代码地址：** [DeepLearning/remoteContainerDevelopment at main · Ustinianer/DeepLearning](https://github.com/Ustinianer/DeepLearning)

**关键词：** 深度学习训练、VSCode，远程开发，容器开发、远程容器开发

---

## 2. 应用场景
**VPN + 跳板机 + 目标机 + 容器开发**

---

## 3. 基础概念理解

### **基础远程开发**
1. 需要先开启 VPN 连接跳板机
2. 通过跳板机连接到目标主机

### **容器远程开发**
1. 需要先开启 VPN 连接跳板机
2. 通过跳板机连接到目标主机内的开发容器

### **主要区别**
1. 对于目标机的端口号配置不同：连接容器时使用 目标主机 IP + 容器 `22` 端口映射到目标主机上的端口。
2. 用户名和密码不同。

---

## 4. 配置开发容器流程

### **脚本地址：** [DeepLearning/remoteContainerDevelopment at main · Ustinianer/DeepLearning](https://github.com/Ustinianer/DeepLearning)

### **1. 拉取 Pytorch CUDA 基础镜像**
> **目的：** 省去手动安装 CUDA 及相关配置

```bash
# docker-pytorch 官方镜像网址  https://hub.docker.com/r/pytorch/pytorch/tags
docker pull pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel
```

### **2. 构建基础开发镜像**
在拉取的镜像基础上构建开发镜像，主要安装 SSH、常用 Linux 工具以及 Miniconda（通常 Pytorch 官方镜像会预装 Miniconda）。

### **3. 启动容器**
> **关注点：** `-p`（端口映射）、`--gpus`（指定 GPU）、`--mount`（挂载目录）

```bash
#!/bin/bash
c=1  

host_name=dev_base
which_image=dev_base:v1
new_user=wmj
new_pwd=123456
container_name=wmj_dev
local_mount_dir=/nvme1/wmj
container_mount_dir=/mnt/wmj

docker run -it \
    -h $host_name \
    -p "$((50401+c)):80" \
    -p "$((49448+10*c)):443" \
    -p "$((10000+c)):6006" \
    -p "$((20000+c)):6007"  \
    -p "$((32868+c)):22" \
    -p "$((40100+10*c)):8000" \
    -p "$((41200+10*c)):8080" \
    -p "$((42300+10*c)):8081" \
    -p "$((43400+10*c)):8082" \
    -p "$((2201+c)):$((2201+c))" \
    --gpus all \
    --name $container_name \
    --ipc=host \
    --mount type=bind,source=$local_mount_dir,target=$container_mount_dir \
    -e NEW_USER=$new_user \
    -e NEW_PWD=$new_pwd \
    $which_image /bin/bash
```

---

## 5. VSCode 远程连接配置

### **基础远程开发配置**
```ini
Host 自定义
  HostName 目标机IP
  Port 22
  User 目标机用户名
  ProxyJump 跳板机用户名@跳板机IP:跳板机端口  
  IdentityFile ~/.ssh/id_rsa
```

### **容器远程开发配置**
```ini
Host 自定义
  HostName 目标机IP
  Port 容器内22端口映射到目标机上的端口
  User 容器内用户名
  ProxyJump 跳板机用户名@跳板机IP:跳板机端口  
  IdentityFile ~/.ssh/id_rsa
```

---

## 6. 远程工具 SSH 连接配置

...

