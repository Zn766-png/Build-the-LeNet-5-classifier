# Experiment: Building the LeNet-5 classifier

## Experiment Objectives:
1. Master the PyTorch CNN components and proficiently use nn.Conv2d (convolution), nn.MaxPool2d (pooling), and nn.Flatten (flattening).
2. Understand the changes in feature map dimensions and be able to calculate the (N, C, H, W) variations of each layer Tensor, especially when transitioning from the convolutional layer to the fully connected layer, including the calculation of the number of nodes.

## Experiment Data:
Use torchvision.datasets.FashionMNIST (10 classes, 28x28 grayscale images). Preprocessing: transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]).

## Experiment Content:
1. Define the LeNet-5 model class (nn.Module): - Layer 1:  Conv2d(in=1, out=6, kernel_size=5, stride=1, padding=2) → ReLU → MaxPool2d(kernel_size=2, stride=2)
- Layer 2:  Conv2d(in=6, out=16, kernel_size=5, stride=1, padding=0) → ReLU → MaxPool2d(kernel_size=2, stride=2)
- Layer 3 (Flatten):  Transform (N, 16, 5, 5) into (N, 16 × 5 × 5), which is (N, 400).
- Layer 4 (FC):  Linear(400 → 120) → ReLU → Linear(120 → 84) → ReLU → Linear(84 → 10).
2. Dimension Check:  Write a function called check_shape(), input random tensor x = torch.randn(1, 1, 28, 28), print and assert the shape of each layer's output.
3. Write the training Pipeline:
- Detecting the device:  device = "cuda" if torch.cuda.is_available() else "cpu".
- Model and data transfer:  model.to(device), inputs, labels = inputs.to(device), labels.to(device).
- Optimizer:  optim.Adam(model.parameters(), lr=0.001).
- Loss function:  nn.CrossEntropyLoss().
4. Print the train loss and test accuracy of each Epoch.

---

# 实验 搭建LeNet-5分类器

## 实验目标：
1. 掌握 PyTorch CNN 组件，熟练使用 nn.Conv2d（卷积）、nn.MaxPool2d（池化）和 nn.Flatten（展平）。
2. 理解特征图维度变化，能够计算每一层 Tensor 的 (N, C, H, W) 变化，特别是从卷积层过渡到全连接层时的节点数计算。

## 实验数据：
使用 torchvision.datasets.FashionMNIST（10类，28x28 灰度图）。预处理：transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])。

## 实验内容：
1. 定义 LeNet-5 模型类 (nn.Module)：
- Layer 1:  Conv2d(in=1, out=6, kernel_size=5, stride=1, padding=2) → ReLU → MaxPool2d(kernel_size=2, stride=2)
- Layer 2:  Conv2d(in=6, out=16, kernel_size=5, stride=1, padding=0) → ReLU → MaxPool2d(kernel_size=2, stride=2)
- Layer 3 (Flatten):  将(N,16,5,5)展平为(N,16×5×5)即(N,400)。
- Layer 4 (FC):  Linear(400 → 120) → ReLU → Linear(120 → 84) → ReLU → Linear(84 → 10)。
2. 维度自检：编写一个 check_shape() 函数，输入随机张量 x = torch.randn(1,1,28,28)，打印并断言每一层输出 shape。
3. 编写训练Pipeline：
- 检测设备：device = "cuda" if torch.cuda.is_available() else "cpu"。
- 模型与数据迁移：model.to(device), inputs, labels = inputs.to(device), labels.to(device)。
- 优化器：optim.Adam(model.parameters(), lr=0.001)。
- 损失函数：nn.CrossEntropyLoss()。
4. 打印每个Epoch的train loss 和 test accuracy