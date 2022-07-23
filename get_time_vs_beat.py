import numpy as np
import matplotlib.pyplot as plt
import os

onset_time = []
onset_beat = []

with open('onset_time.txt') as time:
    for t in time:
        onset_time.append(t.replace('\n',''))

with open('onset_beat.txt') as beat:
    for b in beat:
        label = b.split(' ')[1].replace('\n','')
        label = int(label) / 60
        onset_beat.append(label)

count_8_beat = 0
is_start = False
time_vs_beat = []
for t,b in zip(onset_time,onset_beat):
    print(t,b)
    item = str(t) + ' ' + str(b) + '\n'
    # if b - int(b) == 0.0:
    time_vs_beat.append(item)

time_vs_beat_file = open('time_vs_beat.txt',mode='w')
time_vs_beat_file.writelines(time_vs_beat)
time_vs_beat_file.close()

# plt.plot(onset_time,onset_beat)
# plt.xlabel('onset time (10ms)')
# plt.ylabel('onset beat')
# plt.savefig('time_vs_beat.jpg', dpi=2000)