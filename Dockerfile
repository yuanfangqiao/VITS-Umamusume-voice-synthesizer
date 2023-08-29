FROM yuanfangqiao/ubuntu-python38:0.3


COPY ./ /VITS-Umamusume-voice-synthesizer/

RUN cd /VITS-Umamusume-voice-synthesizer/ && \
 pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
 pip3 install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ &&\
 rm -rf /root/.cache/pip/

WORKDIR /VITS-Umamusume-voice-synthesizer/

CMD ["python3","-u", "app.py"]
