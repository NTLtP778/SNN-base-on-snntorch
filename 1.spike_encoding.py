import snntorch as snn
from snntorch import utils
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import spikegen
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
# Training Parameters
batch_size=128
data_path='./data'
num_classes = 10  # MNIST has 10 output classes
# Torch Variables
dtype = torch.float
# Define a transform
transform = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])
mnist_train = datasets.MNIST(root = './', train=True, download=True, transform=transform)
'''
- 在我们真正开始训练网络之前，我们不需要大型数据集。
    snntorch.utils包含一些用于修改数据集的有用函数。
    应用data_subset将数据集缩减为子集中定义的因子。例如，对于子集=10：60000的训练集将减少到6000。
'''
subset = 10
mnist_train = utils.data_subset(mnist_train, subset)
print(f"The size of mnist_train is {len(mnist_train)}")
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)

'''
## Rate Coding 速率编码
- spikegen.rate 将输入数据进行 速率编码。输入数据的值被解释为发生脉冲的概率。
- 参数 num_steps 表示时间步数，控制生成脉冲数据的时间维度。
- 生成的数据形状:
    100 → 时间步数 num_steps。
    128 → 批次大小 batch_size。
    1 → 可能表示通道数（例如灰度图像的单通道）。
    28, 28 → 输入数据的空间维度（MNIST 图像尺寸）。
- 通过 gain 参数 来控制速率编码（Rate Coding）的脉冲频率
'''
print('Rate Coding……')
#next() 获取当前批次的输入数据 data_it 和目标标签 targets_it。
data = iter(train_loader)
data_it, targets_it = next(data)
num_steps = 100

spike_data1 = spikegen.rate(data_it, num_steps=num_steps)
print(spike_data1.size()) #torch.Size([100, 128, 1, 28, 28])

## 可视化
spike_data_sample1 = spike_data1[:, 0, 0]
print(spike_data_sample1.size()) #torch.Size([100, 28, 28])
fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample1, fig, ax)
anim.save('spike_animation1.mp4')
print(f"The corresponding target is: {targets_it[0]}")

spike_data2 = spikegen.rate(data_it, num_steps=num_steps, gain=0.25) #通过 gain 参数 来控制速率编码（Rate Coding）的脉冲频率
print(spike_data2.size()) #torch.Size([100, 128, 1, 28, 28])

spike_data_sample2 = spike_data2[:, 0, 0]
print(spike_data_sample2.size()) #torch.Size([100, 28, 28])
fig, ax = plt.subplots()
anim = splt.animator(spike_data_sample2, fig, ax)
anim.save('spike_animation2.mp4')
print(f"The corresponding target is: {targets_it[0]}")

#对比不同Gain下的脉冲激活
plt.figure(facecolor="w")
plt.subplot(1,2,1)
plt.imshow(spike_data_sample1.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
plt.axis('off')
plt.title('Gain = 1')
plt.subplot(1,2,2)
plt.imshow(spike_data_sample2.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
plt.axis('off')
plt.title('Gain = 0.25')
plt.show()

'''
这段代码展示了如何使用 snnTorch 的 spikeplot.raster 方法生成栅格图（Raster Plot），用于可视化脉冲数据的时间步和神经元激活情况。
栅格图的用途:栅格图 是展示脉冲神经网络中 神经元发放模式 的标准可视化工具。
           每个点表示在特定时间步，特定神经元产生了脉冲。
           图中的 时间步 和 神经元编号 帮助直观理解神经元激活的时间与空间分布。
'''

spike_data_sample2 = spike_data_sample2.reshape((num_steps, -1))
fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data_sample2, ax, s=1.5, c="black")
plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()
#单个神经元
idx = 400  # index into 210th neuron
fig = plt.figure(facecolor="w", figsize=(8, 1))
ax = fig.add_subplot(111)
splt.raster(spike_data_sample2.reshape(num_steps, -1)[:, idx].unsqueeze(1), ax, s=100, c="black", marker="|")
plt.title("Input Neuron")
plt.xlabel("Time step")
plt.yticks([])
plt.show()
'''
延迟编码
spikegen.latency```函数用于实现延迟编码。输入特征根据其大小决定发放脉冲的时间
    -tau（时间常数）较大的tau脉冲发放时间更晚，较小的tau会使得脉冲更快触发。
    -threshold（阈值）：定义了最小输入特征值，低于阈值的输入无法产生脉冲（电流不足以充电到触发阈值）。小于此阈值的输入被直接裁剪，并被分配到最后一个时间步。
'''
print('Latency Coding……')
spike_data = spikegen.latency(data_it, num_steps=100, tau=5, threshold=0.01, linear= True ,normalize= True , clip= True)

fig = plt.figure(facecolor="w", figsize=(10, 5))
ax = fig.add_subplot(111)
splt.raster(spike_data[:, 0].view(num_steps, -1), ax, s=25, c="black")

plt.title("Input Layer")
plt.xlabel("Time step")
plt.ylabel("Neuron Number")
plt.show()

# optional save
# fig.savefig('destination_path.png', format='png', dpi=300)
