import numpy as np
from scipy.stats import wilcoxon

# 原始数据
X = np.load('/data0/home/luyizhuo/ICLR2025/Rebuttal/Reviewer_2/Our_results/sub3/EPE.npy')#.squeeze()
Y = np.load('/data0/home/luyizhuo/ICLR2025/Rebuttal/Reviewer_2/Chen_results/sub3/EPE.npy')#.squeeze()

print(np.mean(X))
print(np.mean(Y))
# Bootstrap抽样次数
B = 100  # bootstrap样本数
n = len(X)

# 存储 bootstrap 计算的均值
mean_X = []
mean_Y = []

# 进行bootstrap抽样
for _ in range(B):
    X_star = np.random.choice(X, size=n, replace=True)
    Y_star = np.random.choice(Y, size=n, replace=True)

    mean_X.append(np.mean(X_star))
    mean_Y.append(np.mean(Y_star))

# 将均值数据转换为numpy数组
mean_X = np.array(mean_X)
mean_Y = np.array(mean_Y)

# 使用Wilcoxon符号秩检验
stat, p_value = wilcoxon(mean_X, mean_Y)

# 输出结果
print("Wilcoxon statistic:", stat)
print("Wilcoxon p-value:", p_value)
