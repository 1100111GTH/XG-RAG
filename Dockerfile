FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
# 宿主设备安装 NVIDIA CUDA Toolkit
# 下方指引仅适用于 Ubuntu 22.04，其它版本或系统请移步查看（ 可参考配置环境变量 ）：https://developer.nvidia.com/cuda-11-8-0-download-archive
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
# sudo dpkg -i cuda-keyring_1.0-1_all.deb
# sudo apt-get update
# sudo apt-get install cuda=11.8.0-1
# reboot
# echo 'PATH="/usr/local/cuda/bin:$PATH"' | sudo tee -a /root/.bashrc
# echo 'LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"' | sudo tee -a /root/.bashrc
# source /root/.bashrc
# nvcc -V 或 nvcc --version 查看为 release 11.8 或 cuda_11.8 即为成功。
## nvidia-smi 查看到的 Cuda Version 仅为驱动兼容的 Cuda 版本，而非实际安装版本。
# ---
# 宿主设备安装 NVIDIA Container Toolkit
# curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
# apt-get update
# apt-get install nvidia-container-toolkit
## 如果在安装前 Docker 已经启动，请重启 Docker（ systemctl restart docker ）
# ---
# 宿主设备安装 Docker Engine（ https://docs.docker.com/engine/install/ ）
# systemctl start docker
# docker build -t img_name:tag -f /path/Dockerfile /path/上下文路径（ 建议查看 README ）
# docker run -itd --name container_name:tag --gpus all -p 2223:2222 -p 2032:2031 -p 6007:6006 img_name:tag


LABEL authors="internethat@tutamail.com"

# 显示 build 详情
ARG BUILDKIT_PROGRESS=plain

# 防止出现选择时区而变相打断 build
ARG DEBIAN_FRONTEND=noninteractive

# 似乎 FastTokenizer 被分叉启用并行性时会导致大语言模型理解能力下降并警告（ 可能导致死锁 ）
ENV TOKENIZERS_PARALLELISM=false

RUN apt-get update && \
    apt-get install -y \
    wget \
    apt-transport-https \
    software-properties-common \
    lsb-release \
    curl \
    gpg \
    # 安装 git
    git-all \
    # 安装 screen
    screen \
    # 安装 nano
    nano \
    # 安装 SSH
    openssh-server && \
    # 安装 Redis
    curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list && \
    apt-get update && \
    apt-get install -y redis && \
    # 安装 Grafana, Prometheus
    ## mkdir -p /etc/apt/keyrings/ && \
    ## wget -q -O - https://apt.grafana.com/gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/grafana.gpg > /dev/null && \
    ## echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://apt.grafana.com stable main" | sudo tee -a /etc/apt/sources.list.d/grafana.list && \
    ## apt-get update && \
    ## apt-get install -y \
    ## prometheus \
    ## grafana && \
    # 清除数据
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安装 Miniconda
RUN mkdir -p /root/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda3/miniconda.sh && \
    bash /root/miniconda3/miniconda.sh -b -u -p /root/miniconda3 && \
    rm -rf /root/miniconda3/miniconda.sh
ARG PATH="/root/miniconda3/bin:$PATH"
RUN /root/miniconda3/bin/conda init bash

# 安装 Python 3.9.18（ 3.8 有 bug，因环境依赖，建议安装 3.9 ）
RUN conda install python==3.9.18 -y && conda clean --all -y

# 创建以项目名命名的目录
RUN --mount=type=bind,source=./packages/config,target=/packages/config \
    python_var=$(python3 /packages/config/config.py) && \
    project_name=$(echo "$python_var" | grep 'project_name=' | cut -d'=' -f2) && \
    mkdir -p "/$project_name" && \
    ln -s "/$project_name" /project_name

WORKDIR /project_name

COPY . /project_name/

# 完成 SSH, Grafana, Prometheus 配置
## SSH 私钥在 packages/sources/keys 中（ 建议自行生成一套公私钥使用，防止 Hack ）
RUN mkdir -p /var/run/sshd && \
    # 如使用密码登陆：echo 'root:xconnectg' | chpasswd（ 外加 PermitRootLogin yes ）
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    # 防止与宿主机 SSH 端口冲突
    sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    mkdir -p /root/.ssh && \
    chmod 700 /root/.ssh && \
    chown root:root /root/.ssh && \
    mv -f /project_name/packages/sources/xg_rag.pub /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys && \
    chown root:root /root/.ssh/authorized_keys && \
    sed -i '/PubkeyAuthentication yes/a\AuthorizedKeysFile /root/.ssh/authorized_keys' /etc/ssh/sshd_config
    # echo -e "\n\n  - job_name: 'vllm'\n    static_configs:\n    - targets: ['127.0.0.1:8000']" >> /etc/prometheus/prometheus.yml && \
    # mv -f /project_name/packages/sources/datasource.yaml /etc/grafana/provisioning/datasources/

RUN pip3 install --upgrade pip && \
    # 安装 Pytorch
    # pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 xformers --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu118 && \
    pip3 install \
    einops \
    scipy \
    poetry \
    pytest \
    # 调度器
    apscheduler \
    # 安装 Redis
    redis \
    # Hugging Face
    accelerate \
    sentence-transformers \
    "transformers>=4.37.0" \
    ## 安装模型权重下载加速器
    hf-transfer \
    # 安装 Tokenizer（ 大语言模型分词器 ）
    tiktoken \
    # 安装 LangChain
    langchain \
    langchain-cli \
    langchain-openai \
    langchain-community \
    langchain-experimental \
    langchain_cohere \
    langsmith \
    langgraph \
    ## 安装 LangServe 含 FastAPI
    "langserve[all]" \
    # 安装 Gradio
    # gradio>=4.26.0" \
    gradio \
    # 安装 Arize Phoenix
    "arize-phoenix[evals]" \
    # 安装 Jupyter 组件
    ## ipykernel 可由 VSCode 提示安装
    ## ipykernel \
    ## ipywidgets \
    # 安装 FlashAttention（ 用于推理加速 ）
    flash-attn --no-build-isolation \
    # 安装 ms-swift
    ## 如 PyPI 库中新版本存在 bug，可尝试下方代码安装 dev 版本
    ## git clone https://github.com/modelscope/swift.git && \
    ## cd /project_name/swift && pip3 install .[llm] && \
    ## cd /project_name && \
    ## rm -rf /project_name/swift && \
    ## 如需更改成特定版本可以执行以下代码
    ### cd /project_name/swift && git checkout 哈希值
    ms-swift==2.0.4 \
    # 安装 NumPy
    numpy \
    # 安装 Faiss
    faiss-cpu \
    # 安装 vLLM（ 用于推理加速 ）
    # https://github.com/vllm-project/vllm/releases/download/v0.4.0.post1/vllm-0.4.0.post1+cu118-cp39-cp39-manylinux1_x86_64.whl && \
    # 若提示 “No matching distribution found for ray ...” 请手动安装一下
    https://github.com/vllm-project/vllm/releases/download/v0.4.2/vllm-0.4.2+cu118-cp39-cp39-manylinux1_x86_64.whl \
    # 安装 AutoAWQ
    autoawq \
    # 用于密码验证
    cryptography \
    # 知识图谱
    # neo4j && \
    # 清除数据
    pip3 cache purge

# 删除文件夹软连接
WORKDIR /
RUN rm -rf /project_name

# Faiss 官方推荐 conda 安装
# 若出现 OMP 错误则建议不使用 mkl
# RUN conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl -y && conda clean --all -y
# gpu 版本能提供 GPU 加速
# RUN conda install -c pytorch -c nvidia -c rapidsai -c conda-forge faiss-gpu-raft=1.8.0 -y && conda clean --all -y

# 替换并配置第三方 systemctl（ 非提权下运行 systemctl ）
## https://github.com/gdraheim/docker-systemctl-replacement
RUN wget https://raw.githubusercontent.com/gdraheim/docker-systemctl-replacement/master/files/docker/systemctl3.py -O /bin/systemctl && \
    sed -i 's|#! /usr/bin/python3|#! /root/miniconda3/bin/python3|' /bin/systemctl && \
    chmod a+x /bin/systemctl

EXPOSE 2222 2031 7007 6006

# ENTRYPOINT 无法通过参数覆盖指令，CMD 可以（ 跟随 img_name:tag 之后的便是新的指令 ）
# CMD 可以用作 ENTRYPOINT 的参数
# 均可使用 exec 或 shell 形式执行命令
# 如果不执行 bash，则 -t 伪终端将自动连接至 sh
ENTRYPOINT systemctl start ssh && bash
# ENTRYPOINT systemctl start grafana-server && systemctl start prometheus && systemctl start prometheus-node-exporter && systemctl start ssh && bash
# ENTRYPOINT bash ./launch_all.sh


# 未标明版本安装的软件，处于当前依赖环境内产生问题几率较小，但当某一环出现重大更新时请注意检查！


# @ Hugging Face 模型权重下载指令
# ENV export HF_HUB_ENABLE_HF_TRANSFER=1（ 设置环境变量开启多线程下载 ）
# ENV export HF_ENDPOINT=https://hf-mirror.com（ 替换为国内镜像站 ）
# RUN huggingface-cli download --resume-download --local-dir-use-symlinks False 项目名/模型名 --local-dir 路径
## 下载下来的模型文件不存在父级目录，请提前创建
## 若下载卡住，请 kill pid 后重试或开启代理，从原始站点下载
# RUN huggingface-cli scan-cache 此命令可查看默认缓存（ 下载 ）目录

# @ docker 常用指令
# 启动 Docker：systemctl start docker
# 启动 Docker：systemctl enable docker --now
# 关闭 Docker：systemctl stop docker
# 重连容器：docker attach container_name:tag
# 暂离容器：Ctrl + p + q
# 离开容器：exit
# ---
# 创建镜像：docker build -t img_name:tag -f /path/Dockerfile /path/上下文路径
## 上下文路径是相对 COPY ADD 等命令的路径
# 镜像启动创建容器：docker run -itd --name container_name:tag --gpus all --memory=80g -p 2223:2222 -p 2032:2031 -p 6007:6006 img_name:tag
## -i 保持容器的标准输入打开、-t 分配一个伪终端、-d 后台运行、 --gpus 选择 GPU、--memory 分配内存、-p 配置转发端口
# 修改容器现有配置：docker update --memory=8g... container_name:tag
# 容器启动：docker start container_name:tag
# 容器启动并临时替换指令：docker start container_name:tag -p 2226:2222
# 停止运行容器：docker stop container_name:tag
# ---
# 删除构建缓存：docker builder prune --all
# 查看所有镜像: docker images
# 查看所有容器：docker ps -a
# 查看所有容器：docker container ls -a（ 不使用用 -a 则仅显示正在运行的容器 ）
# 删除容器：docker rm container_name
# 删除镜像：docker rmi img_name:tag
# ---
# 容器导出镜像：docker commit container_name:tag img_name:tag
# 镜像导出 tar：docker save -o /path/img_name.tar img_name:tag
## 仅可保存成 tar 格式
# 加载镜像：docker load -i /path/img_name.tar
# ---
# 查看日志：docker logs container_name:tag
# ---
# 复制文件至容器：docker cp /path/file container_name:tag:/path
# 复制文件至宿主机：docker cp container_name:tag:/path/file /path

# @ ssh 常用指令
# 生成用于连接的公私钥匙：ssh-keygen
# 常规连接：ssh root@ip -p 安全外壳协议端口
# 公私钥连接：ssh -i /path/id_rsa root@ip -p 安全外壳协议端口
# X11 转发：ssh -i /path/id_rsa -X root@ip -p 安全外壳协议端口
# 端口转发（ 不可执行命令 ）：ssh -i /path/id_rsa -L 端口:localhost:端口 root@ip -p 安全外壳协议端口 -N


