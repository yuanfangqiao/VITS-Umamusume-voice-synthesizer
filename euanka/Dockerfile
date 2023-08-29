FROM yuanfangqiao/ubuntu-python38:0.2 

RUN git config --global http.proxy 'socks5://192.168.1.6:7890'  && \
    git config --global https.proxy 'socks5://192.168.1.6:7890' && \
    git config --global http.proxy http://192.168.1.6:7890  && \
    git config --global https.proxy https://192.168.1.6:7890 && \
    git clone --recursive https://github.com/yuanfangqiao/VITS-Umamusume-voice-synthesizer.git 

RUN cd /VITS-Umamusume-voice-synthesizer/ && \
 pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
 pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ 

WORKDIR /VITS-Umamusume-voice-synthesizer/

CMD ["python3","-u", "app.py"]
