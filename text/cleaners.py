import re
from text.japanese import japanese_to_romaji_with_accent, japanese_to_ipa, japanese_to_ipa2, japanese_to_ipa3

# from text.shanghainese import shanghainese_to_ipa
# from text.cantonese import cantonese_to_ipa
# from text.ngu_dialect import ngu_dialect_to_ipa


def japanese_cleaners(text):
    text = japanese_to_romaji_with_accent(text)
    text = re.sub(r'([A-Za-z])$', r'\1.', text)
    return text


def japanese_cleaners2(text):
    return japanese_cleaners(text).replace('ts', 'ʦ').replace('...', '…')





# def shanghainese_cleaners(text):
#     text = shanghainese_to_ipa(text)
#     text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
#     return text


# def chinese_dialect_cleaners(text):
#     text = re.sub(r'\[ZH\](.*?)\[ZH\]',
#                   lambda x: chinese_to_ipa2(x.group(1))+' ', text)
#     text = re.sub(r'\[JA\](.*?)\[JA\]',
#                   lambda x: japanese_to_ipa3(x.group(1)).replace('Q', 'ʔ')+' ', text)
#     text = re.sub(r'\[SH\](.*?)\[SH\]', lambda x: shanghainese_to_ipa(x.group(1)).replace('1', '˥˧').replace('5',
#                   '˧˧˦').replace('6', '˩˩˧').replace('7', '˥').replace('8', '˩˨').replace('ᴀ', 'ɐ').replace('ᴇ', 'e')+' ', text)
#     text = re.sub(r'\[GD\](.*?)\[GD\]',
#                   lambda x: cantonese_to_ipa(x.group(1))+' ', text)
#     text = re.sub(r'\[EN\](.*?)\[EN\]',
#                   lambda x: english_to_lazy_ipa2(x.group(1))+' ', text)
#     text = re.sub(r'\[([A-Z]{2})\](.*?)\[\1\]', lambda x: ngu_dialect_to_ipa(x.group(2), x.group(
#         1)).replace('ʣ', 'dz').replace('ʥ', 'dʑ').replace('ʦ', 'ts').replace('ʨ', 'tɕ')+' ', text)
#     text = re.sub(r'\s+$', '', text)
#     text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
#     return text
