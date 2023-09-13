
import os
import sys
import numpy as np

exp_dict = {}
exp_list = []

path = f"./{sys.argv[1]}"
log_path = os.path.join(path, 'log.txt')
with open(log_path, 'r') as f:
    lines = f.read().split('\n')

lines = [line for line in lines if line[:4] == '>>>>']
line3, line2, line1 =lines[-3], lines[-2], lines[-1]
if '>>>> EP 1000, train' in line3:

    test_acc_idx = line2.index('acc:')
    test_acc = float(line2[test_acc_idx+4:test_acc_idx+10])

    t = line3.split(',')
    dsp = [(e.split(':')[0]).strip().replace('-- ', '') for e in t[1:]] + ['test acc']
    dsp = ["{:>10}".format(e) for e in dsp]
    value = [float(e.split(':')[1]) for e in t[1:]] + [test_acc]
    value = ["{:>10}".format(str(e)) for e in value]

    print(','.join(dsp))
    print(','.join(value))
    print(line3)
    print(line2)
    print(line1)