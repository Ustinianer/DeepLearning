#!/bin/bash
c=1;  

host_name=dev_base
which_image=dev_base:v1
new_user=wmj
new_pwd=123456
container_name=wmj_dev
local_mount_dir=/nvme1/wmj
container_mount_dir=/mnt/wmj

# mkdir -p /data/cont_space/"$new_user"_"$c"
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