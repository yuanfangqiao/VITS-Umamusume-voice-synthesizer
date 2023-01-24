import argparse
import json
import os
import re
import tempfile
import logging
logging.getLogger('numba').setLevel(logging.WARNING)
import ONNXVITS_infer
import librosa
import numpy as np
import torch
from torch import no_grad, LongTensor
import commons
import utils
import gradio as gr
import gradio.utils as gr_utils
import gradio.processing_utils as gr_processing_utils
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text
from text.symbols import symbols
from mel_processing import spectrogram_torch
import translators.server as tss
import psutil
from datetime import datetime
from text.cleaners import japanese_cleaners

def audio_postprocess(self, y):
    if y is None:
        return None

    if gr_utils.validate_url(y):
        file = gr_processing_utils.download_to_file(y, dir=self.temp_dir)
    elif isinstance(y, tuple):
        sample_rate, data = y
        file = tempfile.NamedTemporaryFile(
            suffix=".wav", dir=self.temp_dir, delete=False
        )
        gr_processing_utils.audio_to_file(sample_rate, data, file.name)
    else:
        file = gr_processing_utils.create_tmp_copy_of_file(y, dir=self.temp_dir)

    return gr_processing_utils.encode_url_or_file_to_base64(file.name)


gr.Audio.postprocess = audio_postprocess

limitation = os.getenv("SYSTEM") == "spaces"  # limit text and audio length in huggingface spaces
languages = ['日本語', '简体中文', 'English', 'English2Katakana']
characters = ['0:特别周', '1:无声铃鹿', '2:东海帝王', '3:丸善斯基',
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
              '84:北港火山', '85:奇锐骏', '86:秋川理事长']
def show_memory_info(hint):
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_info()
    memory = info.rss / 1024.0 / 1024
    print("{} 内存占用: {} MB".format(hint, memory))


def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

hps = utils.get_hparams_from_file("./configs/uma87.json")
symbols = hps.symbols
net_g = ONNXVITS_infer.SynthesizerTrn(
    len(hps.symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model)
_ = net_g.eval()

_ = utils.load_checkpoint("pretrained_models/G_1153000.pth", net_g)

def to_symbol_fn(is_symbol_input, input_text, temp_text):
    return (_clean_text(input_text, hps.data.text_cleaners), input_text) if is_symbol_input \
        else (temp_text, temp_text)

def infer(text_raw, character, language, duration, noise_scale, noise_scale_w, is_symbol):
    # check character & duraction parameter
    if language not in languages:
        print("Error: No such language\n")
        return "Error: No such language", None
    if character not in characters:
        print("Error: No such character\n")
        return "Error: No such character", None
    # check text length
    if limitation:
        text_len = len(text_raw) if is_symbol else len(re.sub("\[([A-Z]{2})\]", "", text_raw))
        max_len = 150
        if is_symbol:
            max_len *= 3
        if text_len > max_len:
            print(f"Refused: Text too long ({text_len}).")
            return "Error: Text is too long", None
        if text_len == 0:
            print("Refused: Text length is zero.")
            return "Error: Please input text!", None
    if is_symbol:
        text = text_raw
    elif language == '日本語':
        text = text_raw
    elif language == '简体中文':
        text = tss.google(text_raw, from_language='zh', to_language='ja')
    elif language == 'English':
        text = tss.google(text_raw, from_language='en', to_language='ja')
    elif language == "English2Katakana":
        text = romajitable.to_kana(text_raw).katakana
    char_id = int(character.split(':')[0])
    stn_tst = get_text(text, hps, is_symbol)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        sid = torch.LongTensor([char_id])
        jp2phoneme = japanese_cleaners(text)
        durations = net_g.predict_duration(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                                           noise_scale_w=noise_scale_w, length_scale=duration)
        char_dur_list = []
        for i, char in enumerate(jp2phoneme):
            char_pos = i * 2 + 1
            char_dur = durations[char_pos]
            char_dur_list.append(char_dur)
        char_spacing_dur_list = []
        char_spacings = []
        for i in range(len(durations)):
            if i % 2 == 0:  # spacing
                char_spacings.append("spacing")
            elif i % 2 == 1:  # char
                char_spacings.append(jp2phoneme[int((i - 1) / 2)])
            char_spacing_dur_list.append(int(durations[i]))
        # convert duration information to string
        duration_info_str = ""
        for i in range(len(char_spacings)):
            if char_spacings[i] == "spacing":
                duration_info_str += str(char_spacing_dur_list[i])
            else:
                duration_info_str += "{" + char_spacings[i] + ":" + str(char_spacing_dur_list[i]) + "}"
            if i != len(char_spacings)-1:
                duration_info_str += ", "
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=duration)[0][0,0].data.float().numpy()
    currentDateAndTime = datetime.now()
    print(f"Character {character} inference successful: {text}\n")
    if language != '日本語':
        print(f"translate from {language}: {text_raw}")
    show_memory_info(str(currentDateAndTime) + " infer调用后")
    return (text,(22050, audio), jp2phoneme, duration_info_str)

def infer_from_phoneme_dur(duration_info_str, character, duration, noise_scale, noise_scale_w):
    try:
        phonemes = duration_info_str.split(", ")
        recons_durs = []
        recons_phonemes = ""
        for item in phonemes:
            if "{" not in item:  # spacing
                recons_durs.append(int(item))
            else:
                recons_phonemes += item.strip("{}").split(":")[0]
                recons_durs.append(int(item.strip("{}").split(":")[1]))
    except ValueError:
        return ("Error: Format must not be changed!", None)
    except AssertionError:
        return ("Error: Format must not be changed!", None)
    char_id = int(character.split(':')[0])
    stn_tst = get_text(recons_phonemes, hps, is_symbol=True)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        sid = torch.LongTensor([char_id])
        print(len(recons_durs))
        print(x_tst.shape[1])
        audio = net_g.infer_with_duration(x_tst, x_tst_lengths, w_ceil=recons_durs, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                            length_scale=duration)[0][0, 0].data.cpu().float().numpy()
    return (recons_phonemes, (22050, audio))

download_audio_js = """
() =>{{
    let root = document.querySelector("body > gradio-app");
    if (root.shadowRoot != null)
        root = root.shadowRoot;
    let audio = root.querySelector("#{audio_id}").querySelector("audio");
    if (audio == undefined)
        return;
    audio = audio.src;
    let oA = document.createElement("a");
    oA.download = Math.floor(Math.random()*100000000)+'.wav';
    oA.href = audio;
    document.body.appendChild(oA);
    oA.click();
    oA.remove();
}}
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
    args = parser.parse_args()
    app = gr.Blocks()
    with app:
        gr.Markdown("# Umamusume voice synthesizer 赛马娘语音合成器\n\n"
                    "![visitor badge](https://visitor-badge.glitch.me/badge?page_id=Plachta.VITS-Umamusume-voice-synthesizer)\n\n"
                    "This synthesizer is created based on [VITS](https://arxiv.org/abs/2106.06103) model, trained on voice data extracted from mobile game Umamusume Pretty Derby \n\n"
                    "这个合成器是基于VITS文本到语音模型，在从手游《賽馬娘：Pretty Derby》解包的语音数据上训练得到。[Dataset Link](https://huggingface.co/datasets/Plachta/Umamusume-voice-text-pairs/tree/main)\n\n"
                    "[introduction video / 模型介绍视频](https://www.bilibili.com/video/BV1T84y1e7p5/?vd_source=6d5c00c796eff1cbbe25f1ae722c2f9f#reply607277701)\n\n"
                    "You may duplicate this space or [open in Colab](https://colab.research.google.com/drive/1J2Vm5dczTF99ckyNLXV0K-hQTxLwEaj5?usp=sharing) to run it privately and without any queue.\n\n"
                    "您可以复制该空间至私人空间运行或打开[Google Colab](https://colab.research.google.com/drive/1J2Vm5dczTF99ckyNLXV0K-hQTxLwEaj5?usp=sharing)在线运行。\n\n"
                    "This model has been integrated to the model collections of [Moe-tts](https://huggingface.co/spaces/skytnt/moe-tts).\n\n"
                    "现已加入[Moe-tts](https://huggingface.co/spaces/skytnt/moe-tts)模型大全。\n\n"
                    "If you have any suggestions or bug reports, feel free to open discussion in Community.\n\n"
                    "若有bug反馈或建议，请在Community下开启一个新的Discussion。 \n\n"
                    "If your input language is not Japanese, it will be translated to Japanese by Google translator, but accuracy is not guaranteed.\n\n"
                    "如果您的输入语言不是日语，则会由谷歌翻译自动翻译为日语，但是准确性不能保证。\n\n"
                    )
        with gr.Row():
            with gr.Column():
                # We instantiate the Textbox class
                textbox = gr.TextArea(label="Text", placeholder="Type your sentence here (Maximum 150 words)", value="こんにちわ。", elem_id=f"tts-input")
                with gr.Accordion(label="Phoneme Input", open=False):
                    temp_text_var = gr.Variable()
                    symbol_input = gr.Checkbox(value=False, label="Symbol input")
                    symbol_list = gr.Dataset(label="Symbol list", components=[textbox],
                                             samples=[[x] for x in symbols],
                                             elem_id=f"symbol-list")
                    symbol_list_json = gr.Json(value=symbols, visible=False)
                symbol_input.change(to_symbol_fn,
                                    [symbol_input, textbox, temp_text_var],
                                    [textbox, temp_text_var])
                symbol_list.click(None, [symbol_list, symbol_list_json], [],
                                  _js=f"""
                (i, symbols) => {{
                    let root = document.querySelector("body > gradio-app");
                    if (root.shadowRoot != null)
                        root = root.shadowRoot;
                    let text_input = root.querySelector("#tts-input").querySelector("textarea");
                    let startPos = text_input.selectionStart;
                    let endPos = text_input.selectionEnd;
                    let oldTxt = text_input.value;
                    let result = oldTxt.substring(0, startPos) + symbols[i] + oldTxt.substring(endPos);
                    text_input.value = result;
                    let x = window.scrollX, y = window.scrollY;
                    text_input.focus();
                    text_input.selectionStart = startPos + symbols[i].length;
                    text_input.selectionEnd = startPos + symbols[i].length;
                    text_input.blur();
                    window.scrollTo(x, y);
                    return [];
                }}""")
                # select character
                char_dropdown = gr.Dropdown(choices=characters, value = "0:特别周", label='character')
                language_dropdown = gr.Dropdown(choices=languages, value = "日本語", label='language')


                duration_slider = gr.Slider(minimum=0.1, maximum=5, value=1, step=0.1, label='时长 Duration')
                noise_scale_slider = gr.Slider(minimum=0.1, maximum=5, value=0.667, step=0.001, label='噪声比例 noise_scale')
                noise_scale_w_slider = gr.Slider(minimum=0.1, maximum=5, value=0.8, step=0.1, label='噪声偏差 noise_scale_w')

                
                
            with gr.Column():
                text_output = gr.Textbox(label="Output Text")
                audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                btn = gr.Button("Generate!")
                cus_dur_gn_btn = gr.Button("Regenerate with custom phoneme durations")
                btn.click(infer, inputs=[textbox, char_dropdown, language_dropdown, duration_slider, noise_scale_slider, noise_scale_w_slider, symbol_input], 
                  outputs=[text_output, audio_output, phoneme_output, duration_output])
                download = gr.Button("Download Audio")
                download.click(None, [], [], _js=download_audio_js.format(audio_id="tts-audio"))
                with gr.Accordion(label="Speaking Pace Control", open=True):
                    phoneme_output = gr.Textbox(label="Output Phonemes", interactive=False)
                    duration_output = gr.Textbox(label="Duration of each phoneme", placeholder="After you generate a sentence, the detailed information of each phoneme's duration will be presented here.",
                                                interactive = True)
                    cus_dur_gn_btn.click(infer_from_phoneme_dur, inputs=[duration_output, char_dropdown, duration_slider, noise_scale_slider, noise_scale_w_slider], 
                          outputs=[phoneme_output, audio_output])
                    gr.Markdown(
                        "\{ \}内的数字代表每个音素在生成的音频中的长度，\{ \}外的数字代表音素之间间隔的长度。"
                        "您可以手动修改这些数字来控制每个音素以及间隔的长度，从而完全控制合成音频的说话节奏。"
                        "注意这些数字只能是整数。 \n\n(1 代表 0.01161 秒的长度)\n\n"
                        "The numbers inside \{ \} represent the length for each phoneme in the generated audio, while the numbers out of \{ \} represent the length of spacings between phonemes."
                        "You can manually change the numbers to adjust the length of each phoneme, so that speaking pace can be completely controlled."
                        "Note that these numbers should be integers only. \n\n(1 represents a length of 0.01161 seconds)\n\n"
                    )
                
        examples = [['haa\u2193......haa\u2193......haa\u2193......haa\u2193......haa\u2193......haa\u2193......haa\u2193......haa\u2193......haa\u2193......haa\u2193......haa\u2193......haa\u2193......', '29:米浴', '日本語', 1, 0.667, 0.8, True],
                    ['お疲れ様です，トレーナーさん。', '1:无声铃鹿', '日本語', 1, 0.667, 0.8, False],
                    ['張り切っていこう！', '67:北部玄驹', '日本語', 1, 0.667, 0.8, False],
                    ['何でこんなに慣れでんのよ，私のほが先に好きだっだのに。', '10:草上飞', '日本語', 1, 0.667, 0.8, False],
                    ['授業中に出しだら，学校生活終わるですわ。', '12:目白麦昆', '日本語', 1, 0.667, 0.8, False],
                    ['お帰りなさい，お兄様！', '29:米浴', '日本語', 1, 0.667, 0.8, False],
                    ['私の処女をもらっでください！', '29:米浴', '日本語', 1, 0.667, 0.8, False]]
        gr.Examples(
            examples=examples,
            inputs=[textbox, char_dropdown, language_dropdown,
                    duration_slider, noise_scale_slider,noise_scale_w_slider, symbol_input],
            outputs=[text_output, audio_output],
            fn=infer
        )
        gr.Markdown("# Updates Logs 更新日志：\n\n"
                   "2023/1/24：\n\n"
                   "增加了对说话节奏的音素级控制。\n\n"
                   "Added more precise control on pace of speaking by modifying the duration of each phoneme.\n\n"
                   "2023/1/13：\n\n"
                   "增加了音素输入的example（米浴喘气）\n\n"
                   "Added one example of phoneme input.\n\n"
                   "2023/1/12：\n\n"
                   "增加了音素输入的功能，可以对语气和语调做到一定程度的精细控制。\n\n"
                   "Added phoneme input, which enables more precise control on output audio.\n\n"
                   "调整了UI的布局。\n\n"
                   "Adjusted UI arrangements.\n\n"
                   "2023/1/10：\n\n"
                   "数据集已上传，您可以在[这里](https://huggingface.co/datasets/Plachta/Umamusume-voice-text-pairs/tree/main)下载。\n\n"
                   "Dataset used for training is now uploaded to [here](https://huggingface.co/datasets/Plachta/Umamusume-voice-text-pairs/tree/main)\n\n"
                   "2023/1/9：\n\n"
                   "模型推理已全面转为onnxruntime，现在不会出现Runtime Error: Memory Limit Exceeded了。\n\n"
                   "Model inference has been fully converted to onnxruntime. There will be no more Runtime Error: Memory Limit Exceeded\n\n"
                   "现已加入[Moe-tts](https://huggingface.co/spaces/skytnt/moe-tts)模型大全。\n\n"
                   "Now integrated to [Moe-tts](https://huggingface.co/spaces/skytnt/moe-tts) collection.\n\n"
                   )
    app.queue(concurrency_count=3).launch(show_api=False, share=args.share)