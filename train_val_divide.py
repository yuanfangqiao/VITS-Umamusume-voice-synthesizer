import os
import numpy as np
filename = 'E:/uma_voice/output.txt'
split ='|'
with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f]

train_filename = filename.split('.')[0] + '_train' + '.txt'
val_filename = filename.split('.')[0] + '_val' + '.txt'

train_split_ratio = 0.99
train_f = open(train_filename, 'w', encoding='utf-8')
val_f = open(val_filename, 'w', encoding='utf-8')
for i in range(len(filepaths_and_text)):
    if np.random.rand() < train_split_ratio:
        train_f.writelines('|'.join(filepaths_and_text[i]) + '\n')
    else:
        val_f.writelines('|'.join(filepaths_and_text[i]) + '\n')