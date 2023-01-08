import argparse
import gradio as gr
import torch
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import numpy as np
import os
import translators.server as tss
def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_info()
    memory = info.rss / 1024.0 / 1024
    print("{} 内存占用: {} MB".format(hint, memory))


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


hps = utils.get_hparams_from_file("./configs/uma87.json")
net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model)
_ = net_g.eval()

_ = utils.load_checkpoint("pretrained_models/uma_1153000.pth", net_g, None)

def infer(text, character, language, duration, noise_scale, noise_scale_w):
    show_memory_info("infer调用前")
    if language == '日本語':
        pass
    elif language == '简体中文':
        text = tss.google(text, from_language='zh', to_language='ja')
    elif language == 'English':
        text = tss.google(text, from_language='en', to_language='ja')
    char_id = int(character.split(':')[0])
    stn_tst = get_text(text, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        sid = torch.LongTensor([char_id])
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                            length_scale=duration)[0][0, 0].data.cpu().float().numpy()
    del stn_tst, x_tst, x_tst_lengths, sid
    show_memory_info("infer调用后")
    return (text, (22050, audio))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
    args = parser.parse_args()
    app = gr.Blocks()
    with app:
        gr.Markdown("# Umamusume voice synthesizer 赛马娘语音合成器\n\n"
                    "![visitor badge](https://visitor-badge.glitch.me/badge?page_id=Plachta.VITS-Umamusume-voice-synthesizer)\n\n"
                    "This synthesizer is created based on [VITS](https://arxiv.org/abs/2106.06103) model, trained on voice data extracted from mobile game Umamusume Pretty Derby \n\n"
                    "这个合成器是基于VITS文本到语音模型，在从手游《賽馬娘：Pretty Derby》解包的语音数据上训练得到。\n\n"
                    "[introduction video / 模型介绍视频](https://www.bilibili.com/video/BV1T84y1e7p5/?vd_source=6d5c00c796eff1cbbe25f1ae722c2f9f#reply607277701)\n\n"
                    "Due to some unknown reason, VITS inference on CPU results in accumulative memory leakage, resulting in Runtime error：Memory limit exceeded.\n\n"
                    "In case of space crash, you may duplicate this space or [open in Colab](https://colab.research.google.com/drive/1J2Vm5dczTF99ckyNLXV0K-hQTxLwEaj5?usp=sharing) to run it privately and without any queue.\n\n"
                    "由于未知原因，VITS模型在CPU上执行推理时会有逐步累积的内存泄漏，最终导致空间报错Runtime error：Memory limit exceeded，目前正在排查。\n\n"
                    "以防该空间崩溃，您可以复制该空间至私人空间运行或打开[Google Colab](https://colab.research.google.com/drive/1J2Vm5dczTF99ckyNLXV0K-hQTxLwEaj5?usp=sharing)在线运行。\n\n"
                    "If your input language is not Japanese, it will be translated to Japanese by Google translator, but accuracy is not guaranteed.\n\n"
                    "如果您的输入语言不是日语，则会由谷歌翻译自动翻译为日语，但是准确性不能保证。\n\n"
                    )
        with gr.Row():
            with gr.Column():
                # We instantiate the Textbox class
                textbox = gr.Textbox(label="Text", placeholder="Type your sentence here", lines=2)
                # select character
                char_dropdown = gr.Dropdown(choices=['0:特别周', '1:无声铃鹿', '2:东海帝王', '3:丸善斯基',
                                              '4:富士奇迹', '5:小栗帽', '6:黄金船', '7:伏特加',
                                              '8:大和赤骥', '9:大树快车', '10:草上飞', '11:菱亚马逊',
                                              '12:目白麦昆', '13:神鹰', '14:好歌剧', '15:成田白仁',
                                              '16:鲁道夫象征', '17:气槽', '18:爱丽数码', '19:青云天空',
                                              '20:玉藻十字', '21:美妙姿势', '22:琵琶晨光', '23:重炮',
                                              '24:曼城茶座', '25:美普波旁', '26:目白雷恩', '27:菱曙',
                                              '28:雪之美人', '29:米浴', '30:艾尼斯风神', '31:爱丽速子',
                                              '32:爱慕织姬', '33:稻荷一', '34:胜利奖券', '35:空中神宫',
                                              '36:荣进闪耀', '37:真机伶', '38:川上公主', '39:黄金城市',
                                              '40:樱花进王', '41:采珠', '42:新光风', '43:东商变革',
                                              '44:超级小溪', '45:醒目飞鹰', '46:荒漠英雄', '47:东瀛佐敦',
                                              '48:中山庆典', '49:成田大进', '50:西野花', '51:春乌拉拉',
                                              '52:青竹回忆', '53:微光飞驹', '54:美丽周日', '55:待兼福来',
                                              '56:Mr.C.B', '57:名将怒涛', '58:目白多伯', '59:优秀素质',
                                              '60:帝王光环', '61:待兼诗歌剧', '62:生野狄杜斯', '63:目白善信',
                                              '64:大拓太阳神', '65:双涡轮', '66:里见光钻', '67:北部玄驹',
                                              '68:樱花千代王', '69:天狼星象征', '70:目白阿尔丹', '71:八重无敌',
                                              '72:鹤丸刚志', '73:目白光明', '74:樱花桂冠', '75:成田路',
                                              '76:也文摄辉', '77:吉兆', '78:谷野美酒', '79:第一红宝石',
                                              '80:真弓快车', '81:骏川手纲', '82:凯斯奇迹', '83:小林历奇',
                                              '84:北港火山', '85:奇锐骏', '86:秋川理事长'], label='character')
                language_dropdown = gr.Dropdown(choices=['日本語', '简体中文', 'English'], label='language')


                duration_slider = gr.Slider(minimum=0.1, maximum=5, value=1, step=0.1, label='时长 Duration')
                noise_scale_slider = gr.Slider(minimum=0.1, maximum=5, value=0.667, step=0.001, label='噪声比例 noise_scale')
                noise_scale_w_slider = gr.Slider(minimum=0.1, maximum=5, value=0.8, step=0.1, label='噪声偏差 noise_scale_w')
            with gr.Column():
                text_output = gr.Textbox(label="Output Text")
                audio_output = gr.Audio(label="Output Voice")
        btn = gr.Button("Generate!")
        btn.click(infer, inputs=[textbox, char_dropdown, language_dropdown,
                                 duration_slider, noise_scale_slider, noise_scale_w_slider],
                  outputs=[text_output, audio_output])
        examples = [['お疲れ様です，トレーナーさん。', '1:无声铃鹿', '日本語', 1, 0.667, 0.8],
                    ['張り切っていこう！', '67:北部玄驹', '日本語', 1, 0.667, 0.8],
                    ['何でこんなに慣れでんのよ，私のほが先に好きだっだのに。', '10:草上飞', '日本語', 1, 0.667, 0.8],
                    ['授業中に出しだら，学校生活終わるですわ。', '12:目白麦昆', '日本語', 1, 0.667, 0.8],
                    ['お帰りなさい，お兄様！', '29:米浴', '日本語', 1, 0.667, 0.8],
                    ['私の処女をもらっでください！', '29:米浴', '日本語', 1, 0.667, 0.8]]
        gr.Examples(
            examples=examples,
            inputs=[textbox, char_dropdown, language_dropdown,
                    duration_slider, noise_scale_slider,noise_scale_w_slider],
            outputs=[text_output, audio_output],
            fn=infer
        )
    app.queue(concurrency_count=3).launch(show_api=False, share=args.share)