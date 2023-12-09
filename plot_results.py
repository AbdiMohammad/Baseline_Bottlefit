import re
import numpy as np
import matplotlib.pyplot as plt

acc_drop_vs_bn_size = []
with open("logs/vgg19.txt") as f:
    for line in f:
        match_bn_size = re.search("bn_size=[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", line)
        if match_bn_size:
            bn_size = int(match_bn_size.group().split("=")[1])
        match_acc = re.search("Acc@1 [-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?", line)
        if match_acc:
            acc = float(match_acc.group().split(" ")[1])
            acc_drop_vs_bn_size.append([bn_size, 93.92 - acc])
acc_drop_vs_bn_size = np.array(acc_drop_vs_bn_size)
print(acc_drop_vs_bn_size[:, 1] * 0.01)

acc_drop_vs_bn_size_plt = plt.figure()
plt.plot(acc_drop_vs_bn_size[:, 0], acc_drop_vs_bn_size[:, 1], '.-')
# plt.xlim((-21, 31))
# plt.ylim((0, 100))
plt.grid(True)
plt.xlabel("BN Size", fontsize=12)
plt.ylabel("Accuracy Drop (\%)", fontsize=12)
# plt.legend(["Naive", "Static BN", "K = 5", "K = 8"], fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
# plt.savefig("/home/mohammad/research/dynamic-nn/repo/SNR-Adaptive-Bottlefit/result/backup1/AccuracyvsSNR.pdf")

print(np.geomspace(0.01, 0.3, num=10))