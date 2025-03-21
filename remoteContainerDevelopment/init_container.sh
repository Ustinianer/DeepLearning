#!/bin/bash
# 初始化文件的功能是根据容器启动时输入的用户名和密码来新建用户，并开启SSH服务。

# Fail if NEW_USER or NEW_PWD is not given
if [ -z "$NEW_USER" ]; then
    echo 'ERROR: Missing -e NEW_USER="username" in docker run command.'
    exit 1
fi
if [ -z "$NEW_PWD" ]; then
    echo 'ERROR: Missing -e NEW_PWD="password" in docker run command.'
    exit 1
fi

# Create a sudo user account with a password and a /bin/bash shell
useradd -m -G sudo -s /bin/bash $NEW_USER &>/dev/null
echo "$NEW_USER:$NEW_PWD" | chpasswd

# Start SSH service
service ssh start

# Switch to the new user
su - $NEW_USER