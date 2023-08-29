
# usage: docker build -t yuanfangqiao/ubuntu-python38:0.3 .
# ubuntu 20.04 镜像源携带3.8
FROM ubuntu:20.04

#设置环境变量
ENV TZ Asia/Shanghai
ENV LANG C.UTF-8

#执行命令
#替换为阿里源
RUN sed -i 's#http://archive.ubuntu.com/#http://mirrors.aliyun.com/#' /etc/apt/sources.list \
    && sed -i 's#http://security.ubuntu.com/#http://mirrors.aliyun.com/#' /etc/apt/sources.list

# 基础依赖，因为某些python包需要
RUN apt-get -qq -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
        gcc \
        g++ \
        zlib1g-dev \
        libssl-dev \
        libbz2-dev \
        libsqlite3-dev \
        libncurses5-dev \
        libgdbm-dev \
        libgdbm-compat-dev \
        liblzma-dev \
        libreadline-dev \
        uuid-dev \
        libffi-dev \
        tk-dev \
        wget \
        curl \
        git \
        make \
        cmake \
        sudo \
        bash-completion \
        tree \
        vim \
        software-properties-common \
        python3-pip \
        python3-wheel \
        build-essential \
        && mv /usr/bin/lsb_release /usr/bin/lsb_release.bak && \
        apt-get -y autoclean && \
        apt-get -y autoremove && \
        rm -rf /var/lib/apt/lists/*

CMD [ "/bin/bash" ]
