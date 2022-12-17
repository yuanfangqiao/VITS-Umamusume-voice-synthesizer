import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import translators.server as tss

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import gradio as gr

from scipy.io.wavfile import write


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

_ = utils.load_checkpoint("pretrained_models/uma87_639000.pth", net_g, None)

title = "Umamusume voice synthesizer \n 赛马娘语音合成器"
description = """
This synthesizer is created based on VITS (https://arxiv.org/abs/2106.06103) model, trained on voice data extracted from mobile game <Umamusume Pretty Derby>\n
这个合成器是基于VITS文本到语音模型，在从手游《賽馬娘：Pretty Derby》解包的语音数据上训练得到。
"""
article = """
If your input language is not Japanese, it will be translated to Japanese by Google translator, but accuracy is not guaranteed.\n
如果您的输入语言不是日语，则会由谷歌翻译自动翻译为日语，但是准确性不能保证。
"""
def infer(text, character, language):
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
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    return (text,(22050, audio))

# We instantiate the Textbox class
textbox = gr.Textbox(label="Type your sentence here:", placeholder="お疲れ様です，トレーナーさん。", lines=2)
# select character
char_dropdown = gr.Dropdown(['0:特别周','1:无声铃鹿','2:东海帝王','3:丸善斯基',
                            '4:富士奇迹','5:小栗帽','6:黄金船','7:伏特加',
                            '8:大和赤骥','9:大树快车','10:草上飞','11:菱亚马逊',
                            '12:目白麦昆','13:神鹰','14:好歌剧','15:成田白仁',
                            '16:鲁道夫象征','17:气槽','18:爱丽数码','19:青云天空',
                            '20:玉藻十字','21:美妙姿势','22:琵琶晨光','23:重炮',
                            '24:曼城茶座','25:美普波旁','26:目白雷恩','27:菱曙',
                            '28:雪之美人','29:米浴','30:艾尼斯风神','31:爱丽速子',
                            '32:爱慕织姬','33:稻荷一','34:胜利奖券','35:空中神宫',
                            '36:荣进闪耀','37:真机伶','38:川上公主','39:黄金城市',
                            '40:樱花进王','41:采珠','42:新光风','43:东商变革',
                            '44:超级小溪','45:醒目飞鹰','46:荒漠英雄','47:东瀛佐敦',
                            '48:中山庆典','49:成田大进','50:西野花','51:春乌拉拉',
                            '52:青竹回忆','53:微光飞驹','54:美丽周日','55:待兼福来',
                            '56:Mr.C.B','57:名将怒涛','58:目白多伯','59:优秀素质',
                            '60:帝王光环','61:待兼诗歌剧','62:生野狄杜斯','63:目白善信',
                            '64:大拓太阳神','65:双涡轮','66:里见光钻','67:北部玄驹',
                            '68:樱花千代王','69:天狼星象征','70:目白阿尔丹','71:八重无敌',
                            '72:鹤丸刚志','73:目白光明','74:樱花桂冠','75:成田路',
                            '76:也文摄辉','77:吉兆','78:谷野美酒','79:第一红宝石',
                            '80:真弓快车','81:骏川手纲','82:凯斯奇迹','83:小林历奇',
                            '84:北港火山','85:奇锐骏','86:秋川理事长'])
language_dropdown = gr.Dropdown(['日本語','简体中文','English'])
examples = [['このデモを使用していただき，ありがとうございます!', '12:目白麦昆', '日本語'],
            ['トレーナーさん，今日はなにお食べます？', '2:东海帝王', '日本語'],
            ['おにいさまは，ライスのこと，好きですか，それども嫌い？', '29:米浴','日本語'],
            ['トレーナーさんは，本当に優しい人ですね。', '24:曼城茶座','日本語']]
gr.Interface(fn=infer, inputs=[textbox, char_dropdown, language_dropdown], outputs=["text","audio"],
            title=title, description=description, article=article, examples = examples).launch()