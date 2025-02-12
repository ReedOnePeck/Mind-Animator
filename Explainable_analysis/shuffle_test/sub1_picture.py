import matplotlib.pyplot as plt
import numpy as np

# 切换到ggplot风格
#plt.style.use('ggplot')

# 数据：X轴上的类别标签，第一组和第二组的数值，以及它们的误差
categories = ['CLIP-pcc' , 'EPE']
values1 = [0.13, 0.012]
values2 = [0.97, 0.87]
values3 = [0.09,0.005]
errors1 = [0.013, 0.011]  # 误差
errors2 = [0.02, 0.06]  # 误差
errors3 = [0.01, 0.004]  # 误差
# 每个类别的位置
ind = np.array([0,1.5])

R = (175/255.0,60/255.0,60/255.0,1)
B = (78/255.0,90/255.0,145/255.0,1)

# 绘制第一组柱状图


bars1 = plt.bar(ind, values1, width=0.3, color=B, label='w/ Consistency Motion Generator')  # 稍亮的蓝色

# 绘制第二组柱状图
bars2 = plt.bar(ind + 0.4, values2, width=0.3, color=R, label='w/o Consistency Motion Generator')  # 稍亮的红色

bars3 = plt.bar(ind - 0.4, values3, width=0.3, color='green', label='Noise ceiling')  # 稍亮的红色

# 设置X轴标签
plt.xticks(ind, categories, fontsize=15)

# 添加误差条
for bar in bars1:
    yval = bar.get_height()
    plt.errorbar(bar.get_x() + bar.get_width() / 2, yval, yerr=errors1[bars1.index(bar)], fmt='.', ecolor=B, capsize=5, capthick=2)

for bar in bars2:
    yval = bar.get_height()
    plt.errorbar(bar.get_x() + bar.get_width() / 2, yval, yerr=errors2[bars2.index(bar)], fmt='.', ecolor=R, capsize=5, capthick=2)

for bar in bars3:
    yval = bar.get_height()
    plt.errorbar(bar.get_x() + bar.get_width() / 2, yval, yerr=errors3[bars3.index(bar)], fmt='.', ecolor='green', capsize=5, capthick=2)

plt.axhline(y=0.05, color='black', linestyle='--', linewidth=1.5, label='p=0.05')

plt.ylim(top=1.0)  # 设置Y轴的最大值为60
plt.gca().set_ylim(bottom=0, top=1.2)  # 绘图区域范围最大值为1.4

# 添加图例
plt.legend(fontsize=12)

# 设置标题和轴标签
#plt.title('Bar Chart with Two Groups and Error Bars')
plt.ylabel('P-values of shuffle test', fontsize=14)
plt.gca().yaxis.set_tick_params(labelsize=14)
# 显示图表
plt.show()