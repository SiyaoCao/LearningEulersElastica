# ContinuousNetwork_main.py 代码逐行解释

## 代码目的概述

本脚本实现了一套基于**连续神经网络算子**（Continuous Neural Operator）的**欧拉弹性曲线（Euler's Elastica）轨迹预测**框架。与离散版本不同，本框架将曲线视为弧长参数 $s \in [0,1]$ 的连续函数，训练一个神经网络 $\mathcal{N}_\theta(s, \mathbf{q}_1, \mathbf{q}_2, \mathbf{v}_1, \mathbf{v}_2)$，使其在给定两端边界条件（位置 $\mathbf{q}_{1,2}$ 与切向 $\mathbf{v}_{1,2}$）后，能对任意弧长坐标 $s$ 输出曲线上对应的位置向量 $\mathbf{q}(s) \in \mathbb{R}^2$。主要流程包括：①读取并预处理数据集；②用 `optuna` 进行贝叶斯超参数搜索；③训练或加载最终模型；④对训练/验证/测试集评估并可视化结果。

---

## 目录

- [一、环境准备与依赖导入（第 1–28 行）](#一环境准备与依赖导入第-128-行)
- [二、全局配置与随机种子（第 35–49 行）](#二全局配置与随机种子第-3549-行)
- [三、用户输入与数据加载（第 55–86 行）](#三用户输入与数据加载第-5586-行)
  - [3.1 辅助函数 `loadData`（GetData.py）](#31-辅助函数-loaddatagetdatapy)
  - [3.2 辅助函数 `getDataLoaders`（GetData.py）](#32-辅助函数-getdataloadersdatagetdatapy)
  - [3.3 数据集类 `dataset`（GetData.py）](#33-数据集类-datasetgetdatapy)
- [四、工具函数 `getBCs`（Utils.py）](#四工具函数-getbcsutilspy)
- [五、索引辅助函数 `flatten_chain`（第 119–125 行）](#五索引辅助函数-flatten_chain第-119125-行)
- [六、神经网络类 `approximate_curve`（Network.py）](#六神经网络类-approximate_curvenetworkpy)
  - [6.1 `__init__` 构造方法](#61-__init__-构造方法)
  - [6.2 `normalize` 归一化方法](#62-normalize-归一化方法)
  - [6.3 `parametric_part` 参数化网络主体](#63-parametric_part-参数化网络主体)
  - [6.4 `get_coefficients` 边界修正系数计算](#64-get_coefficients-边界修正系数计算)
  - [6.5 `correction_bcs` 边界修正多项式](#65-correction_bcs-边界修正多项式)
  - [6.6 `forward` 前向传播方法](#66-forward-前向传播方法)
  - [6.7 `derivative` 一阶导数方法](#67-derivative-一阶导数方法)
  - [6.8 `second_derivative` 二阶导数方法](#68-second_derivative-二阶导数方法)
- [七、超参数搜索模型构建函数 `define_model`（第 92–113 行）](#七超参数搜索模型构建函数-define_model第-92113-行)
- [八、Optuna 目标函数 `objective`（第 131–225 行）](#八optuna-目标函数-objective第-131225-行)
- [九、Optuna 超参数搜索（第 231–242 行）](#九optuna-超参数搜索第-231242-行)
- [十、超参数选取逻辑（第 248–266 行）](#十超参数选取逻辑第-248266-行)
  - [10.1 辅助函数 `hyperparams`（SavedParameters.py）](#101-辅助函数-hyperparamssavedparameterspy)
- [十一、最优模型构建函数 `define_best_model`（第 278–288 行）](#十一最优模型构建函数-define_best_model第-278288-行)
- [十二、训练配置与模型训练（第 294–341 行）](#十二训练配置与模型训练第-294341-行)
  - [12.1 训练函数 `trainModel`（Training.py）](#121-训练函数-trainmodeltrainingpy)
- [十三、结果评估与可视化（第 349–351 行）](#十三结果评估与可视化第-349351-行)
  - [13.1 辅助函数 `plotTestResults`（PlotResults.py）](#131-辅助函数-plottestresultsplotresultspy)
- [附录：网络结构总览](#附录网络结构总览)

---

> 本文档按主函数的运行顺序，逐段解释 `ContinuousNetwork_main.py` 中的每一行代码。涉及计算的部分均附有对应的数学公式，行号在前，公式在后，可以分段对应。

---

## 一、环境准备与依赖导入（第 1–28 行）

```python
1.  #!/usr/bin/env python
2.  # coding: utf-8
```

- **第 1 行**：Shebang 行，声明使用系统 Python 解释器运行脚本。
- **第 2 行**：声明源文件编码为 UTF-8，保证中文等非 ASCII 字符可正常处理。

```python
9.  import torch
10. import torch.nn as nn
11. import matplotlib.pyplot as plt
12. import numpy as np
13. from torch.autograd.functional import jacobian as jac
14. from torch.func import jacfwd, vmap
15. from csv import writer
16. import os
17. import optuna
```

- **第 9–10 行**：导入 PyTorch 核心库及其神经网络模块。
- **第 11 行**：导入 `matplotlib.pyplot`，用于绘制曲线与误差图。
- **第 12 行**：导入 NumPy，用于数值计算和数组操作。
- **第 13 行**：从 `torch.autograd.functional` 导入 `jacobian`（别名 `jac`），可计算函数的雅可比矩阵（备用接口）。
- **第 14 行**：导入 `jacfwd`（基于前向模式自动微分的雅可比计算）和 `vmap`（向量化映射），二者配合用于批量高效计算一阶/二阶导数。
- **第 15 行**：导入 CSV 写入工具（保存超参数搜索结果，当前代码中已注释）。
- **第 16 行**：导入操作系统接口（用于路径切换）。
- **第 17 行**：导入 `optuna`，贝叶斯超参数优化框架。

```python
23. from Scripts.GetData import getDataLoaders, loadData
24. from Scripts.Utils import getBCs
25. from Scripts.Network import approximate_curve
26. from Scripts.Training import trainModel
27. from Scripts.PlotResults import plotTestResults
28. from Scripts.SavedParameters import hyperparams
```

- **第 23–28 行**：从本地 `Scripts` 包导入自定义模块：
  - `getDataLoaders`：将原始轨迹数据构建为训练/验证/测试 `DataLoader`。
  - `loadData`：从磁盘读取数据文件，返回节点数与轨迹矩阵。
  - `getBCs`：从轨迹矩阵中提取两端边界条件（位置与切向量）。
  - `approximate_curve`：连续神经网络模型类，核心建模组件。
  - `trainModel`：执行完整的训练循环。
  - `plotTestResults`：对测试集进行预测并绘图、打印误差。
  - `hyperparams`：根据训练数据比例返回预存的最优超参数字典。

---

## 二、全局配置与随机种子（第 35–49 行）

```python
35. torch.set_default_dtype(torch.float32)
```

- **第 35 行**：将 PyTorch 全局默认张量数据类型设为单精度浮点数（`float32`），在计算精度与内存效率之间取得平衡。

```python
41. device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
42. print(device)
```

- **第 41–42 行**：自动检测 CUDA GPU 是否可用。若可用，则使用第 0 号 GPU（`cuda:0`）以加速张量运算；否则退而使用 CPU。

```python
48. torch.manual_seed(1)
49. np.random.seed(1)
```

- **第 48–49 行**：分别为 PyTorch 和 NumPy 固定随机种子为 1，保证每次运行结果可重复（包括数据划分、模型权重初始化等）。

---

## 三、用户输入与数据加载（第 55–86 行）

```python
55. percentage_train = input("Choose percentage of training data between 80, 40, 20, and 10: ")
56. percentage_train = int(percentage_train)/100
57. percentage_train
```

- **第 55–56 行**：读取用户输入的训练集比例（可选 80、40、20、10），将整数除以 100 转换为小数形式（如 `80 → 0.8`），即

$$p_{\text{train}} = \frac{\text{input}}{100}, \quad \text{input} \in \{10, 20, 40, 80\}$$

```python
63. batch_size = 1024
64. epochs = 100
```

- **第 63 行**：设置每个训练批次（mini-batch）包含的样本数为 1024。
- **第 64 行**：设置训练总轮数（epoch）为 100。

```python
70. num_nodes, trajectories_train, test_traj = loadData()
71. shuffle_idx_train = np.random.permutation(len(trajectories_train))
72. trajectories_train = trajectories_train[shuffle_idx_train]
73. test_traj = test_traj[shuffle_idx_train]
```

- **第 70 行**：调用 `loadData()` 从磁盘加载欧拉弹性曲线数据集，返回节点数 `num_nodes` 以及训练/测试轨迹矩阵（每行一条轨迹，包含所有节点的位置与切向量）。详见[第 3.1 节](#31-辅助函数-loaddatagetdatapy)。
- **第 71–73 行**：生成随机排列索引，对训练轨迹矩阵和对应的测试轨迹矩阵同步洗牌，防止数据顺序引入偏差：

$$\boldsymbol{\pi} = \text{randperm}(N_{\text{total}}), \quad \mathbf{T}_{\text{train}} \leftarrow \mathbf{T}_{\text{train}}[\boldsymbol{\pi}], \quad \mathbf{T}_{\text{test}} \leftarrow \mathbf{T}_{\text{test}}[\boldsymbol{\pi}]$$

```python
75. number_samples_train, number_components = trajectories_train.shape
76. indices_train = np.random.permutation(len(trajectories_train))
77. trajectories_train = trajectories_train[indices_train]
78. number_samples_test, _ = test_traj.shape
79. test_traj = test_traj[indices_train]
```

- **第 75 行**：从轨迹矩阵形状中提取总样本数 $N_{\text{total}}$ 和每条轨迹的分量数 $D$。每条轨迹有 $D = 4 \times N_{\text{nodes}}$ 个分量（每个节点包含位置 $(q_x, q_y)$ 与切向量 $(q'_x, q'_y)$ 共 4 个值）：

$$\mathbf{T} \in \mathbb{R}^{N_{\text{total}} \times D}, \quad D = 4 \times N_{\text{nodes}}$$

- **第 76–79 行**：再次随机排列（第二次洗牌），进一步打乱数据顺序，并同步更新测试轨迹矩阵。

```python
81. number_elements = int(number_components/4)-1
82. data_train, data_test, data_val, x_train, x_test, y_train, y_test, x_val, y_val, trainloader, testloader, valloader = getDataLoaders(batch_size, number_elements,number_samples_train, number_samples_test,trajectories_train, test_traj, percentage_train)
```

- **第 81 行**：计算有限元素数（即节点间区间数）：

$$N_{\text{elem}} = \frac{D}{4} - 1 = N_{\text{nodes}} - 1$$

- **第 82 行**：调用 `getDataLoaders` 构建数据加载器（详见[第 3.2 节](#32-辅助函数-getdataloadersdatagetdatapy)），返回数据字典、特征矩阵、标签矩阵及三个 `DataLoader`。

```python
84. training_trajectories = np.concatenate((x_train[:,:4],y_train,x_train[:,-4:]),axis=1)
85. test_trajectories = np.concatenate((x_test[:,:4],y_test,x_test[:,-4:]),axis=1)
86. val_trajectories = np.concatenate((x_val[:,:4],y_val,x_val[:,-4:]),axis=1)
```

- **第 84–86 行**：将被拆分的特征和标签重新拼接成完整轨迹矩阵，用于后续评估和可视化。每条完整轨迹的结构为：

$$\mathbf{t} = [\underbrace{\mathbf{q}_1, \mathbf{v}_1}_{\text{左端 BC（4 维）}},\ \underbrace{\mathbf{q}(s_1), \ldots, \mathbf{q}(s_{N-2})}_{\text{内部节点（y）}},\ \underbrace{\mathbf{q}_2, \mathbf{v}_2}_{\text{右端 BC（4 维）}}]$$

---

### 3.1 辅助函数 `loadData`（GetData.py）

```python
# GetData.py 第 7–21 行
def loadData():
    original_dir = os.getcwd()
    root_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    os.chdir(root_dir+"/DataSets")
    
    both_ends_360_sol = open("both_ends.txt", "r")
    trajectoriesload_b_360 = np.loadtxt(both_ends_360_sol)

    trajectories_train = trajectoriesload_b_360
    trajectories_test = trajectories_train 
    
    num_nodes = trajectories_train.shape[1]//4
    os.chdir(original_dir)
    return num_nodes, trajectories_train, trajectories_test
```

- **`os.getcwd()` / `os.chdir()`**：保存并切换当前工作目录，确保数据文件路径正确（数据集位于上级目录的 `DataSets/` 子目录中）。
- **`open("both_ends.txt")`**：打开包含欧拉弹性曲线数值解的文本文件，每行为一条完整曲线轨迹。数据集名称 "both_ends" 表示两端边界条件均已知（位置与切向量）。
- **`np.loadtxt(...)`**：将文本文件解析为 NumPy 矩阵 $\mathbf{T} \in \mathbb{R}^{N_{\text{total}} \times 4N_{\text{nodes}}}$。
- **`num_nodes = shape[1] // 4`**：节点总数等于列数除以 4（每个节点 4 个分量）：

$$N_{\text{nodes}} = \lfloor D / 4 \rfloor$$

- 训练集与测试集初始指向同一矩阵（后续在 `getDataLoaders` 中按比例划分）。

---

### 3.2 辅助函数 `getDataLoaders`（GetData.py）

```python
# GetData.py 第 23–186 行
def getDataLoaders(batch_size, number_elements, number_samples,
                   number_samples_test, trajectories_train,
                   trajectories_test, percentage_train):
```

该函数完成以下操作：

**提取特征与标签**

```python
x_full_train = np.concatenate((trajectories_train[:,:4], trajectories_train[:,-4:]), axis=1)
y_full_train = trajectories_train[:,4:-4]
```

- 特征 $\mathbf{x}$（输入）为每条轨迹的两端边界条件（左端 4 维 + 右端 4 维），共 8 维：

$$\mathbf{x} = [\mathbf{q}_1,\ \mathbf{v}_1,\ \mathbf{q}_2,\ \mathbf{v}_2] \in \mathbb{R}^8$$

- 标签 $\mathbf{y}$（输出）为去除两端边界条件后的内部节点值：

$$\mathbf{y} = [\mathbf{q}(s_1), \mathbf{v}(s_1), \ldots, \mathbf{q}(s_{N-2}), \mathbf{v}(s_{N-2})] \in \mathbb{R}^{4(N_{\text{nodes}}-2)}$$

**按比例划分数据集**

```python
Ntrain = int(percentage_train * number_samples)
```

$$N_{\text{train}} = \lfloor p_{\text{train}} \times N_{\text{total}} \rfloor$$

训练集取前 $N_{\text{train}}$ 条轨迹；测试集和验证集各取 $N_{\text{test}} = \lfloor 0.1 \times N_{\text{total}} \rfloor$ 条。

**节点采样策略**

```python
x_range = np.linspace(0, 1, number_elements+1)
list_nodes_boundary1 = np.array([0,1,2,3,5])
list_nodes_boundary2 = np.array([number_elements-4,...,number_elements])
list_nodes_others = np.arange(6, number_elements-4, 2)
list_nodes = np.concatenate((list_nodes_boundary1, list_nodes_others, list_nodes_boundary2))
```

- 弧长参数 $s$ 被均匀离散化到 $[0,1]$ 上：

$$s_k = \frac{k}{N_{\text{elem}}}, \quad k = 0, 1, \ldots, N_{\text{elem}}$$

- 采样策略：两端附近的节点（边界区域）全部采样，内部节点每隔一个采样，以在保证边界精度的同时减少训练数据量。

**构建数据字典**

对每条训练/测试轨迹的每个采样节点，将以下信息存入字典：

| 键 | 含义 | 维度 |
|---|---|---|
| `s` | 弧长坐标 $s_k$ | 标量 |
| `q1` | 左端位置 $\mathbf{q}_1$ | $\mathbb{R}^2$ |
| `v1` | 左端切向量 $\mathbf{v}_1$ | $\mathbb{R}^2$ |
| `q2` | 右端位置 $\mathbf{q}_2$ | $\mathbb{R}^2$ |
| `v2` | 右端切向量 $\mathbf{v}_2$ | $\mathbb{R}^2$ |
| `qs` | 节点 $k$ 处的真实位置 $\mathbf{q}(s_k)$ | $\mathbb{R}^2$ |
| `vs` | 节点 $k$ 处的真实切向量 $\mathbf{v}(s_k)$ | $\mathbb{R}^2$ |

**构建 DataLoader**

```python
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader  = DataLoader(testset,  batch_size=len(x_test),  shuffle=True)
valloader   = DataLoader(valset,   batch_size=len(x_val),   shuffle=True)
```

- 训练集使用指定 `batch_size` 且随机打乱，测试集和验证集一次性全部加载（batch_size 等于数据集大小）。

---

### 3.3 数据集类 `dataset`（GetData.py）

```python
# GetData.py 第 188–203 行
class dataset(Dataset):
    def __init__(self, data):
        self.q1  = torch.from_numpy(data["q1"].astype(np.float32))
        self.q2  = torch.from_numpy(data["q2"].astype(np.float32))
        self.v1  = torch.from_numpy(data["v1"].astype(np.float32))
        self.v2  = torch.from_numpy(data["v2"].astype(np.float32))
        self.s   = torch.from_numpy(data["s"].astype(np.float32)).unsqueeze(1)
        self.sample = torch.from_numpy(data["sample_number"].astype(np.float32))
        self.qs  = torch.from_numpy(data["qs"].astype(np.float32))
        self.vs  = torch.from_numpy(data["vs"].astype(np.float32))
        self.length = self.q1.shape[0]

    def __getitem__(self, idx):
        return self.q1[idx], self.q2[idx], self.v1[idx], self.v2[idx],
               self.s[idx], self.sample[idx], self.qs[idx], self.vs[idx]

    def __len__(self):
        return self.length
```

- **`__init__`**：将 NumPy 数组转换为 `float32` 类型的 PyTorch 张量。`s` 需要 `unsqueeze(1)` 增加一个维度，使其形状从 $(N,)$ 变为 $(N, 1)$，以便与其他二维张量拼接。
- **`__getitem__`**：按索引返回一条训练样本，包含 8 项数据。
- **`__len__`**：返回数据集大小，供 `DataLoader` 计算批次数。

---

## 四、工具函数 `getBCs`（Utils.py）

```python
# Utils.py 第 5–9 行
def getBCs(trajectories):
    bcs = {"q1": trajectories[:, :2],
           "q2": trajectories[:, -4:-2],
           "v1": trajectories[:, 2:4],
           "v2": trajectories[:, -2:]}
    return bcs
```

- 从完整轨迹矩阵 $\mathbf{T} \in \mathbb{R}^{B \times D}$ 中提取两端边界条件：
  - $\mathbf{q}_1 = \mathbf{T}[:, 0:2]$：左端位置向量（$x$ 分量与 $y$ 分量）
  - $\mathbf{v}_1 = \mathbf{T}[:, 2:4]$：左端切向量
  - $\mathbf{q}_2 = \mathbf{T}[:, -4:-2]$：右端位置向量
  - $\mathbf{v}_2 = \mathbf{T}[:, -2:]$：右端切向量

这四个量构成连续神经网络的条件输入，描述了弹性曲线的边界约束：

$$\text{BCs} = (\mathbf{q}_1,\ \mathbf{v}_1,\ \mathbf{q}_2,\ \mathbf{v}_2) \in \mathbb{R}^2 \times \mathbb{R}^2 \times \mathbb{R}^2 \times \mathbb{R}^2$$

---

## 五、索引辅助函数 `flatten_chain`（第 119–125 行）

```python
119. from itertools import chain

121. def flatten_chain(matrix):
122.     return list(chain.from_iterable(matrix))

124. q_idx  = flatten_chain([[i,   i+1] for i in np.arange(0, number_components, 4)])
125. qp_idx = flatten_chain([[i+2, i+3] for i in np.arange(0, number_components, 4)])
```

- **`flatten_chain`**：将嵌套列表展开为一维列表，例如 `[[0,1],[4,5]]` → `[0,1,4,5]`。
- **第 124 行**：构造位置分量的列索引。每个节点的数据排列为 $(q_x, q_y, q'_x, q'_y)$，因此位置分量的索引为：

$$\text{q\_idx} = [0, 1, 4, 5, 8, 9, \ldots, 4(N-1), 4(N-1)+1]$$

- **第 125 行**：构造切向量分量的列索引：

$$\text{qp\_idx} = [2, 3, 6, 7, 10, 11, \ldots, 4(N-1)+2, 4(N-1)+3]$$

这两个索引用于从预测的完整轨迹矩阵中分别提取位置预测值和切向量预测值。

---

## 六、神经网络类 `approximate_curve`（Network.py）

`approximate_curve` 是本框架的核心模型，继承自 `torch.nn.Module`，实现了一个以弧长 $s$ 和边界条件 $(\mathbf{q}_1, \mathbf{q}_2, \mathbf{v}_1, \mathbf{v}_2)$ 为输入、以曲线上位置向量 $\mathbf{q}(s) \in \mathbb{R}^2$ 为输出的连续神经算子。

### 6.1 `__init__` 构造方法

```python
# Network.py 第 6–51 行
class approximate_curve(nn.Module):
    def __init__(self, normalize=True, act_name='tanh', nlayers=3,
                 hidden_nodes=50, correct_functional=True,
                 is_res=True, is_mult=False, both=True):
        super().__init__()
        torch.manual_seed(1)
        np.random.seed(1)
```

- **`normalize`**：是否对输入边界条件做归一化预处理。
- **`act_name`**：激活函数类型（`'tanh'`、`'sin'`、`'swish'`、其他则为 `sigmoid`）。
- **`nlayers`**：隐藏层数量 $L$。
- **`hidden_nodes`**：每个隐藏层的神经元数 $H$。
- **`correct_functional`**：是否在网络输出上叠加多项式边界修正项（本框架中设为 `False`，即无修正）。
- **`is_res`**：是否使用残差连接（ResNet 风格）。
- **`is_mult`**：是否使用乘法门控网络（MULT 结构）。
- **`both`**：是否同时强制两端边界条件（位置与切向量），为 `True` 时修正多项式为三次 Hermite，为 `False` 时为一次线性。

```python
        if act_name=='tanh':
            self.act = lambda x: torch.tanh(x)
        elif act_name=="sin":
            self.act = lambda x: torch.sin(x)
        elif act_name=="swish":
            self.act = lambda x: x * torch.sigmoid(x)
        else:
            self.act = lambda x: torch.sigmoid(x)
```

激活函数选项的数学定义（其中 $\sigma(x) = \frac{1}{1+e^{-x}}$ 为标准 sigmoid 函数）：

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}, \quad \sin(x), \quad \text{swish}(x) = x \cdot \sigma(x), \quad \sigma(x) = \frac{1}{1+e^{-x}}$$

```python
        self.embed   = nn.Linear(9, self.hidden_nodes)
        self.lift    = nn.Linear(self.hidden_nodes, self.hidden_nodes)
        ll = []
        for it in range(self.nlayers):
            ll.append(nn.Linear(self.hidden_nodes, self.hidden_nodes))
        self.linears  = nn.ModuleList([ll[i] for i in range(self.nlayers)])
        self.linearsO = nn.ModuleList([nn.Linear(self.hidden_nodes, self.hidden_nodes)
                                       for i in range(self.nlayers)])
        self.lift_U   = nn.Linear(self.hidden_nodes, self.hidden_nodes)
        self.lift_V   = nn.Linear(self.hidden_nodes, self.hidden_nodes)
        self.lift_H   = nn.Linear(self.hidden_nodes, self.hidden_nodes)
        self.linears_Z = nn.ModuleList([nn.Linear(self.hidden_nodes, self.hidden_nodes)
                                        for i in range(self.nlayers)])
        self.proj     = nn.Linear(self.hidden_nodes, 2)
```

各层的参数矩阵形状（格式为权重矩阵 $\mathbf{W}$ 的行列数，即 输出维度 × 输入维度）：

| 层 | 权重形状 $\mathbf{W}$ | 用途 |
|---|---|---|
| `embed` | $H \times 9$ | 将 9 维输入映射到隐空间，再经 Fourier 特征编码 |
| `lift` | $H \times H$ | MLP/ResNet 模式的提升层 |
| `linears[l]` | $H \times H$ | 每层的内部变换矩阵 |
| `linearsO[l]` | $H \times H$ | 每层的输出变换矩阵（MULT 模式中构成 $Z^{(l)}$） |
| `lift_U`, `lift_V`, `lift_H` | $H \times H$ | MULT 网络的门控提升层（各一个） |
| `linears_Z[l]` | $H \times H$ | MULT 网络中 $Z^{(l)}$ 的内部变换 |
| `proj` | $2 \times H$ | 将隐向量映射为 $\mathbb{R}^2$ 输出 |

---

### 6.2 `normalize` 归一化方法

```python
# Network.py 第 162–167 行
def normalize(self, q1, q2, v1, v2):
    q1n = (q1 - 1.5) / 1.5
    q2n = q2
    v1n = v1
    v2n = v2
    return q1n, q2n, v1n, v2n
```

- 仅对左端位置 $\mathbf{q}_1$ 做中心化与缩放，将数值范围从 $[0, 3]$ 映射到 $[-1, 1]$：

$$\tilde{\mathbf{q}}_1 = \frac{\mathbf{q}_1 - 1.5}{1.5}$$

- $\mathbf{q}_2$、$\mathbf{v}_1$、$\mathbf{v}_2$ 保持不变。此归一化有助于改善梯度流动，加速收敛。

---

### 6.3 `parametric_part` 参数化网络主体

```python
# Network.py 第 77–104 行
def parametric_part(self, s, q1, q2, v1, v2):
    input = torch.cat((s, q1, q2, v1, v2), dim=1)
    input = torch.sin(2 * torch.pi * self.embed(input))
```

**输入拼接与 Fourier 特征编码**

将弧长参数与边界条件拼接为 9 维向量，再通过嵌入层和正弦函数进行 Fourier 特征编码：

$$\mathbf{z} = [s,\ \mathbf{q}_1,\ \mathbf{q}_2,\ \mathbf{v}_1,\ \mathbf{v}_2] \in \mathbb{R}^9$$

$$\mathbf{h}^{(0)} = \sin\!\left(2\pi \left(\mathbf{W}_{\text{emb}}\, \mathbf{z} + \mathbf{b}_{\text{emb}}\right)\right) \in \mathbb{R}^H$$

Fourier 特征编码利用正弦函数的周期性，使网络更容易拟合高频函数，相当于将输入映射到随机 Fourier 特征空间。

**分支一：乘法门控网络（MULT，`is_mult=True`）**

```python
    if self.is_mult:
        U = self.act(self.lift_U(input))
        V = self.act(self.lift_V(input))
        H = self.act(self.lift_H(input))
        for i in range(self.nlayers):
            Z = self.linearsO[i](self.act(self.linears_Z[i](H)))
            H = U * (1 - Z) + V * Z
        input = H
```

MULT 网络通过门控机制混合两个固定向量 $U$ 和 $V$，其更新规则如下：

$$U = \sigma\!\left(\mathbf{W}_U \mathbf{h}^{(0)} + \mathbf{b}_U\right), \quad V = \sigma\!\left(\mathbf{W}_V \mathbf{h}^{(0)} + \mathbf{b}_V\right)$$

$$H^{(0)} = \sigma\!\left(\mathbf{W}_H \mathbf{h}^{(0)} + \mathbf{b}_H\right)$$

对每一层 $l = 0, 1, \ldots, L-1$：

$$Z^{(l)} = \mathbf{W}_{O}^{(l)}\, \sigma\!\left(\mathbf{W}_{Z}^{(l)} H^{(l)} + \mathbf{b}_{Z}^{(l)}\right)$$

$$H^{(l+1)} = U \odot (1 - Z^{(l)}) + V \odot Z^{(l)}$$

其中 $\odot$ 表示逐元素乘积。$Z^{(l)}$ 起到软性插值门控的作用：当 $Z^{(l)} \to \mathbf{0}$ 时输出接近 $U$，当 $Z^{(l)} \to \mathbf{1}$ 时输出接近 $V$。

**分支二：残差网络（ResNet，`is_res=True`）与普通 MLP（`is_res=False`）**

```python
    else:
        input = self.act(self.lift(input))
        for i in range(self.nlayers):
            if self.is_res:
                input = input + self.linearsO[i](self.act(self.linears[i](input)))
            else:
                input = self.act(self.linears[i](input))
```

普通 MLP（`is_res=False`）：

$$\mathbf{h}^{(l+1)} = \sigma\!\left(\mathbf{W}^{(l+1)} \mathbf{h}^{(l)} + \mathbf{b}^{(l+1)}\right), \quad l = 0, \ldots, L-1$$

残差网络（`is_res=True`，每层加恒等跳跃连接以缓解梯度消失）：

$$\mathbf{h}^{(l+1)} = \mathbf{h}^{(l)} + \mathbf{W}_{O}^{(l+1)}\, \sigma\!\left(\mathbf{W}^{(l+1)} \mathbf{h}^{(l)} + \mathbf{b}^{(l+1)}\right), \quad l = 0, \ldots, L-1$$

**输出投影**

```python
    output = self.proj(input)
    return output
```

将最终隐向量线性投影到 $\mathbb{R}^2$（曲线上的二维位置）：

$$g(s) = \mathbf{W}_{\text{proj}}\, H^{(L)} + \mathbf{b}_{\text{proj}} \in \mathbb{R}^2$$

---

### 6.4 `get_coefficients` 边界修正系数计算

```python
# Network.py 第 107–149 行
def get_coefficients(self, q1, q2, v1, v2):
    B = len(q1)
    left_node  = torch.zeros((B,1), dtype=torch.float32).to(q1.device)
    right_node = torch.ones((B,1),  dtype=torch.float32).to(q1.device)

    q = lambda s,a,b,c,d: self.parametric_part(
            s.reshape(-1,1), a.reshape(-1,2), b.reshape(-1,2),
            c.reshape(-1,2), d.reshape(-1,2))
```

- 设 $g(s) = \texttt{parametric\_part}(s, \cdot)$ 为未修正的网络输出（不满足边界条件）。
- 在 `impose_both_bcs=True`（同时强制两端位置与切向量）时，需要求 Hermite 三次多项式系数 $a_0, a_1, a_2, a_3$，使得：

$$q(s) = g(s) + a_0 + a_1 s + a_2 s^2 + a_3 s^3$$

满足四个边界条件：

$$q(0) = \mathbf{q}_1, \quad q(1) = \mathbf{q}_2, \quad q'(0) = \mathbf{v}_1, \quad q'(1) = \mathbf{v}_2$$

```python
    if self.impose_both_bcs:
        g_left  = q(left_node, ...)       # g(0)
        g_right = q(right_node, ...)      # g(1)
        gp_left  = vmap(jacfwd(q, argnums=0))(left_node,  ...)[:,0,:,0]  # g'(0)
        gp_right = vmap(jacfwd(q, argnums=0))(right_node, ...)[:,0,:,0]  # g'(1)

        a0 = q1 - g_left
        a1 = v1 - gp_left
        a2 = 2*gp_left + gp_right - 3*g_right + 3*g_left - 3*q1 + 3*q2 - 2*v1 - v2
        a3 = -gp_right + 2*g_right - 2*g_left + 2*q1 - gp_left - 2*q2 + v1 + v2
```

由四个边界条件联立方程组（令 $c(s) = a_0 + a_1 s + a_2 s^2 + a_3 s^3$ 为修正量）：

$$c(0) = a_0 = \mathbf{q}_1 - g(0)$$

$$c'(0) = a_1 = \mathbf{v}_1 - g'(0)$$

$$c(1) = a_0 + a_1 + a_2 + a_3 = \mathbf{q}_2 - g(1)$$

$$c'(1) = a_1 + 2a_2 + 3a_3 = \mathbf{v}_2 - g'(1)$$

解出 $a_2$ 和 $a_3$（以 $\delta_q = \mathbf{q}_2 - \mathbf{q}_1$，$\delta_g = g(1) - g(0)$，$\delta_v = \mathbf{v}_2 - g'(1)$ 为辅助量）：

- 来自网络输出的项：$+2g'(0)$、$+g'(1)$、$-3g(1)$、$+3g(0)$
- 来自边界条件的项：$-3\mathbf{q}_1$、$+3\mathbf{q}_2$、$-2\mathbf{v}_1$、$-\mathbf{v}_2$

$$a_2 = \underbrace{2g'(0) + g'(1) - 3g(1) + 3g(0)}_{\text{来自网络输出}} + \underbrace{3(\mathbf{q}_2 - \mathbf{q}_1) - 2\mathbf{v}_1 - \mathbf{v}_2}_{\text{来自边界条件}}$$

$$a_3 = \underbrace{-g'(1) + 2g(1) - 2g(0) - g'(0)}_{\text{来自网络输出}} + \underbrace{2(\mathbf{q}_1 - \mathbf{q}_2) + \mathbf{v}_1 + \mathbf{v}_2}_{\text{来自边界条件}}$$

在 `impose_both_bcs=False`（仅强制两端位置，不强制切向量）时，修正多项式退化为一次：

$$c(s) = a_0 + a_1 s, \quad a_0 = \mathbf{q}_1 - g(0), \quad a_1 = \mathbf{q}_2 - \mathbf{q}_1 + g(0) - g(1)$$

---

### 6.5 `correction_bcs` 边界修正多项式

```python
# Network.py 第 151–160 行
def correction_bcs(self, s, q1, q2, v1, v2):
    s = s.reshape(-1, 1)
    if self.impose_both_bcs:
        a0, a1, a2, a3 = self.get_coefficients(q1, q2, v1, v2)
        return a0 + a1*s + a2*s**2 + a3*s**3
    else:
        a0, a1 = self.get_coefficients(q1, q2, v1, v2)
        return a0 + a1*s
```

根据 `get_coefficients` 计算出的系数构造修正多项式：

**四端点 Hermite 三次修正（`impose_both_bcs=True`）**：

$$c(s) = a_0 + a_1 s + a_2 s^2 + a_3 s^3$$

**两端点线性修正（`impose_both_bcs=False`）**：

$$c(s) = a_0 + a_1 s$$

---

### 6.6 `forward` 前向传播方法

```python
# Network.py 第 171–181 行
def forward(self, s, q1, q2, v1, v2):
    if self.is_norm:
        q1n, q2n, v1n, v2n = self.normalize(q1, q2, v1, v2)
        net_part = self.parametric_part(s, q1n, q2n, v1n, v2n)
    else:
        net_part = self.parametric_part(s, q1, q2, v1, v2)
    if self.correct_functional:
        correction_bcs = self.correction_bcs(s, q1, q2, v1, v2)
        return net_part + correction_bcs
    else:
        return net_part
```

前向传播的完整计算图：

1. **归一化**（若 `is_norm=True`）：对 $\mathbf{q}_1$ 做归一化，得 $(\tilde{\mathbf{q}}_1, \mathbf{q}_2, \mathbf{v}_1, \mathbf{v}_2)$。
2. **参数化网络**：计算 $g(s) = \texttt{parametric\_part}(s, \ldots)$。
3. **边界修正**（若 `correct_functional=True`）：叠加修正多项式，得最终预测：

$$\hat{\mathbf{q}}(s) = g(s) + c(s)$$

  其中 $c(s)$ 确保 $\hat{\mathbf{q}}$ 精确满足给定的边界条件。当 `correct_functional=False` 时（本主函数中的设置），网络不做硬性修正，边界条件完全由训练学习：

$$\hat{\mathbf{q}}(s) = g(s)$$

---

### 6.7 `derivative` 一阶导数方法

```python
# Network.py 第 53–63 行
def derivative(self, s, q1, q2, v1, v2):
    B = len(q1)
    q = lambda s,a,b,c,d: self.forward(
            s.reshape(-1,1), a.reshape(-1,2), b.reshape(-1,2),
            c.reshape(-1,2), d.reshape(-1,2))
    if self.is_norm:
        q1n, q2n, v1n, v2n = self.normalize(q1, q2, v1, v2)
        return vmap(jacfwd(q, argnums=0))(s, q1n, q2n, v1n, v2n)[:,0,:,0]
    else:
        return vmap(jacfwd(q, argnums=0))(s, q1, q2, v1, v2)[:,0,:,0]
```

利用前向模式自动微分（`jacfwd`）和向量化映射（`vmap`）批量计算曲线对弧长参数的一阶导数（即切向量）：

$$\hat{\mathbf{q}}'(s) = \frac{\partial \hat{\mathbf{q}}}{\partial s}(s, \mathbf{q}_1, \mathbf{q}_2, \mathbf{v}_1, \mathbf{v}_2)$$

- `jacfwd(q, argnums=0)` 对函数 $q$ 关于第 0 个参数 $s$ 求雅可比矩阵（在本例中等价于一维导数，因为 $s$ 是标量）。
- `vmap(...)` 对批次维度进行向量化映射，相当于 for 循环但速度更快。
- `[:,0,:,0]` 的索引操作提取雅可比矩阵中 $\partial q / \partial s$ 的形状为 $(B, 2)$ 的部分。

---

### 6.8 `second_derivative` 二阶导数方法

```python
# Network.py 第 65–75 行
def second_derivative(self, s, q1, q2, v1, v2):
    q_p = lambda s,a,b,c,d: self.derivative(
            s.reshape(-1,1), a.reshape(-1,2), b.reshape(-1,2),
            c.reshape(-1,2), d.reshape(-1,2))
    if self.is_norm:
        q1n, q2n, v1n, v2n = self.normalize(q1, q2, v1, v2)
        return vmap(jacfwd(q_p, argnums=0))(s, q1n, q2n, v1n, v2n)[:,0,:,0]
    else:
        return vmap(jacfwd(q_p, argnums=0))(s, q1, q2, v1, v2)[:,0,:,0]
```

类似地，对一阶导数方法 `derivative` 再次应用 `jacfwd`，得到曲线对弧长的二阶导数（即曲率相关量）：

$$\hat{\mathbf{q}}''(s) = \frac{\partial^2 \hat{\mathbf{q}}}{\partial s^2}(s, \mathbf{q}_1, \mathbf{q}_2, \mathbf{v}_1, \mathbf{v}_2)$$

二阶导数的模的平方与曲线曲率 $\kappa$ 密切相关（对弧长参数化的曲线）：

$$\kappa(s) = \|\hat{\mathbf{q}}''(s)\|$$

---

## 七、超参数搜索模型构建函数 `define_model`（第 92–113 行）

```python
92.  def define_model(trial):
93.      torch.manual_seed(1)
94.      np.random.seed(1)
95.
96.      normalize = True
97.      netarch   = 0
98.
99.      if netarch == 0:
100.         is_mult = True
101.         is_res  = False
102.     elif netarch == 1:
103.         is_mult = False
104.         is_res  = True
105.     else:
106.         is_mult = False
107.         is_res  = False
108.     act = 'tanh'
109.     nlayers      = trial.suggest_int("n_layers",     3,   8)
110.     hidden_nodes = trial.suggest_int("hidden_nodes", 10, 200)
111.     model = approximate_curve(normalize, act, nlayers, hidden_nodes,
112.                               correct_functional=False, is_res=is_res,
113.                               is_mult=is_mult, both=False)
114.     return model
```

- **第 92 行**：接收 `trial`（Optuna 试验对象）作为参数，用于从搜索空间中采样超参数。
- **第 94–95 行**：固定随机种子，确保每次超参数试验的模型初始化一致。
- **第 96–108 行**：固定部分超参数：启用归一化（`normalize=True`），使用 MULT 网络结构（`netarch=0`），激活函数为 `tanh`。
- **第 109 行**：在 $[3, 8]$ 范围内搜索隐藏层数 $L$，Optuna 使用贝叶斯优化（树结构 Parzen 估计器，TPE）建议试验值。
- **第 110 行**：在 $[10, 200]$ 范围内搜索每层神经元数 $H$。
- **第 111–113 行**：用采样的超参数实例化模型，`correct_functional=False`（直接学习 BC，不用多项式修正），`both=False`（单端 BC 模式）。

---

## 八、Optuna 目标函数 `objective`（第 131–225 行）

```python
131. def objective(trial):
132.     torch.manual_seed(1)
133.     np.random.seed(1)
134.
135.     model = define_model(trial)
136.     model.to(device)
137.
138.     lr           = 1e-3
139.     weight_decay = 0
140.     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
141.     criterion = nn.MSELoss()
142.     _, _, _, _, _, _, _, _, _, trainloader, testloader, valloader = getDataLoaders(...)
```

- **第 135–136 行**：为当前 trial 构建并部署模型。
- **第 138–141 行**：配置 Adam 优化器（学习率 $\eta = 10^{-3}$，无权重衰减）和均方误差损失函数：

$$\eta = 10^{-3}, \quad \mathcal{L}_{\text{MSE}}(\hat{\mathbf{q}}, \mathbf{q}) = \frac{1}{N} \sum_{i=1}^{N} \|\hat{\mathbf{q}}_i - \mathbf{q}_i\|^2$$

```python
143.     loss = trainModel(number_elements, device, model, criterion, optimizer,
                          epochs, trainloader, valloader, ...)
144.     val_error = 100
145.     if not torch.isnan(loss):
```

- **第 143 行**：调用 `trainModel` 执行训练（详见[第 12.1 节](#121-训练函数-trainmodeltrainingpy)），返回最终损失值。
- **第 144 行**：将验证误差初始化为极大值（100），若训练产生 NaN 则直接以此作为 trial 结果（被 Optuna 视为失败的超参数组合）。

**批量评估验证集**（第 172–200 行）：

```python
172.     bcs = getBCs(val_trajectories)
...
186.     xx = torch.linspace(0,1,number_elements+1).unsqueeze(1).repeat(len(q1),1).to(device)
187.     one = torch.ones((number_elements+1,1)).to(device)
188.     q1_augmented = torch.kron(q1, one)
...
193.     pred_val_q  = model(xx, q1_augmented, ...).reshape(len(q1),-1).detach().cpu().numpy()
194.     val_error_q = np.mean((pred_val_q - val_trajectories[:,q_idx])**2)
195.     pred_val_qp = model.derivative(xx, q1_augmented, ...).reshape(len(q1),-1).detach().cpu().numpy()
196.     val_error_qp = np.mean((pred_val_qp - val_trajectories[:,qp_idx])**2)
...
200.     val_error   = np.mean((pred_val_all - val_trajectories)**2)
```

- **第 186–191 行**：用 `torch.kron`（Kronecker 乘积）将每条轨迹的边界条件向量复制 $N_{\text{elem}}+1$ 次，以匹配所有弧长采样点的批次维度。若 $\mathbf{q}_1 \in \mathbb{R}^{B \times 2}$，则：

$$\mathbf{q}_1^{\text{aug}} = \mathbf{q}_1 \otimes \mathbf{1}_{N+1} \in \mathbb{R}^{B(N+1) \times 2}$$

- **第 193–200 行**：在整条验证轨迹上（所有 $N_{\text{elem}}+1$ 个弧长点）进行预测，计算位置和切向量的 MSE，最终返回综合验证误差：

$$\mathcal{E}_{\text{val}} = \frac{1}{BM} \sum_{b=1}^{B}\sum_{k=0}^{M} \|\hat{\mathbf{q}}^{(b)}(s_k) - \mathbf{q}^{(b)}(s_k)\|^2$$

其中 $M = N_{\text{elem}}+1$ 为弧长采样点数，$B$ 为验证集样本数。

```python
225.     return val_error
```

- `objective` 函数返回验证误差，Optuna 将最小化此值来指导超参数搜索。

---

## 九、Optuna 超参数搜索（第 231–242 行）

```python
231. optuna_study = input("Do you want to do hyperparameter test? Type yes or no ")
232. params = {}
233. if optuna_study == "yes":
234.     optuna_study = True
235. else:
236.     optuna_study = False
237. if optuna_study:
238.     study = optuna.create_study(direction="minimize", study_name="Euler Elastica")
239.     study.optimize(objective, n_trials=300)
240.     print("Study statistics: ")
241.     print("  Number of finished trials: ", len(study.trials))
242.     params = study.best_params
```

- **第 231–236 行**：通过用户输入决定是否执行超参数搜索。
- **第 238 行**：创建 Optuna `Study` 对象，设置优化方向为最小化验证误差，并命名为 "Euler Elastica"。
- **第 239 行**：执行 300 次 trial 的超参数搜索。Optuna 默认使用 TPE（Tree-structured Parzen Estimator）算法，它对目标函数的分布进行建模：

$$\text{TPE 采样策略：} \quad x^* = \arg\max_{x} \frac{\ell(x)}{g(x)}$$

  其中 $\ell(x)$ 是使目标函数值低的超参数的核密度估计，$g(x)$ 是使目标函数值高的超参数的核密度估计。

- **第 242 行**：提取搜索到的最优超参数字典，用于后续模型构建。

---

## 十、超参数选取逻辑（第 248–266 行）

```python
248. torch.manual_seed(1)
249. np.random.seed(1)

255. manual_input = False
256. if params == {}:
257.     if manual_input:
258.         nlayers = int(input("How many layers do you want the network to have? "))
259.         hidden_nodes = int(input("How many hidden nodes do you want the network to have? "))
260.         params = {"n_layers": nlayers, "hidden_nodes": hidden_nodes}
261.     else:
262.         params = hyperparams(percentage_train)
```

- **第 248–249 行**：重置随机种子，确保最终模型训练的可重复性（与超参数搜索阶段相互独立）。
- **第 255–266 行**：超参数来源的三级优先级：
  1. 若已完成 Optuna 搜索（`params != {}`），直接使用搜索结果；
  2. 若 `manual_input=True`，通过用户输入手动指定；
  3. 否则（默认路径）调用 `hyperparams(percentage_train)` 加载预存的最优参数。

---

### 10.1 辅助函数 `hyperparams`（SavedParameters.py）

```python
# SavedParameters.py 第 1–15 行
def hyperparams(percentage_train):
    params = {}
    if percentage_train == 0.8:
        params = {'n_layers': 6, 'hidden_nodes': 106}
    elif percentage_train == 0.4:
        params = {'n_layers': 8, 'hidden_nodes': 181}
    elif percentage_train == 0.2:
        params = {'n_layers': 7, 'hidden_nodes': 185}
    else:
        params = {'n_layers': 6, 'hidden_nodes': 139}
    return params
```

根据训练集比例返回经过 Optuna 搜索预先确定的最优网络结构参数：

| 训练比例 $p_{\text{train}}$ | 隐藏层数 $L$ | 每层节点数 $H$ |
|---|---|---|
| 0.8 | 6 | 106 |
| 0.4 | 8 | 181 |
| 0.2 | 7 | 185 |
| 0.1 | 6 | 139 |

---

## 十一、最优模型构建函数 `define_best_model`（第 278–288 行）

```python
278. def define_best_model():
279.     normalize    = True
280.     act          = 'tanh'
281.     nlayers      = params["n_layers"]
282.     hidden_nodes = params["hidden_nodes"]
283.     is_mult      = True
284.     is_res       = False
285.     model = approximate_curve(normalize, act, nlayers, hidden_nodes,
286.                               correct_functional=False,
287.                               is_res=is_res, is_mult=is_mult, both=False)
288.     return model
```

- 使用 `params` 中存储的最优超参数，固定构建 MULT 网络（`is_mult=True`，`is_res=False`）。
- `correct_functional=False`：不使用硬性多项式边界修正，由模型自行学习满足边界条件。
- `both=False`：仅关注一端（或无多项式修正时无意义，此参数不影响结果）。

```python
294. model = define_best_model()
295. model.to(device)
```

- **第 294 行**：实例化最优模型。
- **第 295 行**：将模型迁移到目标设备（GPU 或 CPU）。

---

## 十二、训练配置与模型训练（第 294–341 行）

```python
301. TrainMode = input("Train Mode True or False? Type 0 for False and 1 for True: ")
302. TrainMode = int(TrainMode)
...
306. TrainMode = bool(TrainMode)
```

- **第 301–306 行**：由用户决定是训练新模型（`TrainMode=True`）还是加载预训练权重（`TrainMode=False`）。

```python
313. weight_decay = 0
314. lr = 1e-3
315. optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
316. criterion = nn.MSELoss()
```

- **第 313–316 行**：配置 Adam 优化器（学习率 $\eta = 10^{-3}$，无 L2 正则化）和 MSE 损失函数。

```python
322. if TrainMode:
323.     loss = trainModel(number_elements, device, model, criterion,
                          optimizer, epochs, trainloader, valloader,
                          train_with_tangents=False,
                          pde_regularisation=False,
                          soft_bcs_imposition=False)
324.     if percentage_train == 0.8:
325.         torch.save(model.state_dict(), 'TrainedModels/BothEnds0.8data.pt')
...
332. else:
333.     if percentage_train == 0.8:
334.         pretrained_dict = torch.load('TrainedModels/BothEnds0.8data.pt', map_location=device)
...
341.     model.load_state_dict(pretrained_dict)
```

- **第 322–331 行**：训练模式下，调用 `trainModel` 训练模型，并根据训练比例将权重保存到对应文件（`.pt` 格式为 PyTorch 模型文件）。
- **第 332–341 行**：加载模式下，从磁盘读取预训练权重字典，并加载到模型中（`load_state_dict`）。

---

### 12.1 训练函数 `trainModel`（Training.py）

```python
# Training.py 第 6–83 行
def trainModel(number_elements, device, model, criterion, optimizer,
               epochs, trainloader, valloader,
               train_with_tangents=False, pde_regularisation=True,
               soft_bcs_imposition=False):
    torch.manual_seed(1)
    np.random.seed(1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)
```

- **`scheduler`**：步进学习率调度器，每 45 个 epoch 将学习率乘以 $\gamma = 0.1$（即每 45 epoch 学习率下降一个数量级）：

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / 45 \rfloor}$$

**训练循环**（每个 epoch）：

```python
    for epoch in range(epochs):
        for i, inp in enumerate(trainloader):
            q1, q2, v1, v2, s, _, qs, vs = inp
            q1,q2,v1,v2,s,qs,vs = [t.to(device) for t in [q1,q2,v1,v2,s,qs,vs]]

            optimizer.zero_grad()
            res_q = model(s, q1, q2, v1, v2)
            loss = criterion(res_q, qs) + criterion(model.derivative(s,q1,q2,v1,v2), vs)
            loss += 1e-2 * torch.mean((torch.linalg.norm(
                                model.derivative(s,q1,q2,v1,v2), ord=2, dim=1)**2 - 1.)**2)
            loss.backward()
            optimizer.step()
```

完整损失函数由三项组成：

$$\mathcal{L} = \underbrace{\frac{1}{N}\sum_{i}\|\hat{\mathbf{q}}(s_i) - \mathbf{q}(s_i)\|^2}_{\text{位置 MSE}} + \underbrace{\frac{1}{N}\sum_{i}\|\hat{\mathbf{q}}'(s_i) - \mathbf{v}(s_i)\|^2}_{\text{切向量 MSE}} + \underbrace{\frac{\lambda}{N}\sum_{i}\left(\|\hat{\mathbf{q}}'(s_i)\|^2 - 1\right)^2}_{\text{弧长正则化}}$$

其中 $\lambda = 10^{-2}$。弧长正则化项要求曲线的切向量为单位向量（弧长参数化条件）：

$$\|\hat{\mathbf{q}}'(s)\| = 1 \iff \left\|\frac{\partial \hat{\mathbf{q}}}{\partial s}\right\|^2 = 1$$

**提前停止与早退机制**：

```python
            if epoch == 1:
                stored_res = loss.item()
            if epoch == 30:
                check = (loss.item() > stored_res * 1e-1)
                if check:
                    print("Early stop due to lack of progress")
                    loss = torch.tensor(torch.nan)
                    break
```

- 在第 1 epoch 记录基准损失 $\mathcal{L}_1$。
- 若第 30 epoch 时 $\mathcal{L}_{30} > 0.1 \times \mathcal{L}_1$（损失未下降足够多），则触发早退，将损失设为 NaN（告知 Optuna 此 trial 失败）。

**验证集评估**（每个 epoch 结束后）：

```python
        with torch.no_grad():
            for inp in valloader:
                ...
                res_q_val = model(s_val, ...)
                val_loss += criterion(res_q_val, qs_val).item()
            val_loss_avg = val_loss / len(valloader)
            val_losses.append(val_loss_avg)
```

在无梯度模式下评估验证集损失，追踪训练动态。此处验证集损失仅含位置 MSE（不含切向量项和正则化项），用于监控过拟合。

**学习率衰减**：

```python
        scheduler.step()
```

每个 epoch 结束后调用 `scheduler.step()`，按预设步进表更新学习率。

---

## 十三、结果评估与可视化（第 349–351 行）

```python
349. model.eval()
350. res, res_derivative = plotTestResults(model, device, number_elements,
                                           number_components,
                                           x_train, x_val, x_test,
                                           y_train, y_val, y_test,
                                           num_nodes, percentage_train)
```

- **第 349 行**：将模型切换到评估模式，禁用 Dropout 和 BatchNorm 的训练行为（本模型未使用这些层，但这是规范做法）。
- **第 350–351 行**：调用 `plotTestResults` 对训练/验证/测试集进行批量预测，打印各集合上的 MSE，并在 `percentage_train=0.8` 时生成可视化图表（详见[第 13.1 节](#131-辅助函数-plottestresultsplotresultspy)）。

---

### 13.1 辅助函数 `plotTestResults`（PlotResults.py）

```python
# PlotResults.py 第 40–229 行
def plotTestResults(model, device, number_elements, number_components,
                    x_train, x_val, x_test, y_train, y_val, y_test,
                    num_nodes, percentage_train):
```

**批量推理（以测试集为例）**：

```python
    xx = torch.linspace(0,1,number_elements+1).unsqueeze(1).repeat(len(q1),1).to(device)
    one = torch.ones((number_elements+1,1)).to(device)
    q1_augmented = torch.kron(q1, one)
    ...
    pred_test_q  = model(xx, q1_augmented, ...).reshape(len(q1),-1).detach().cpu().numpy()
    pred_test_qp = model.derivative(xx, q1_augmented, ...).reshape(len(q1),-1).detach().cpu().numpy()
```

- 通过 Kronecker 乘积将每条轨迹的边界条件向量沿弧长采样点方向广播：

$$\mathbf{q}_1^{\text{aug}} = \mathbf{q}_1 \otimes \mathbf{1}_{M \times 1} \in \mathbb{R}^{BM \times 2}, \quad M = N_{\text{elem}}+1$$

- 将模型预测结果重塑为 $(B, M \cdot 2)$ 的矩阵，每行为一条曲线所有节点的位置预测值。

**误差计算**：

```python
    pred_test_all = np.zeros_like(test_trajectories)
    pred_test_all[:, q_idx]  = pred_test_q
    pred_test_all[:, qp_idx] = pred_test_qp
    print(f"Error over test trajectories: {np.mean((pred_test_all-test_trajectories)**2)}.")
```

将预测的位置与切向量按索引填入完整轨迹矩阵，计算整体 MSE：

$$\mathcal{E}_{\text{test}} = \frac{1}{BD} \sum_{b=1}^{B} \sum_{d=1}^{D} (\hat{T}_{b,d} - T_{b,d})^2$$

其中 $D = 4N_{\text{nodes}}$ 为每条轨迹的总分量数。

**可视化（仅在 `percentage_train=0.8` 时生成）**：

- **图 1**（`fig1`）：在 $(q_x, q_y)$ 平面上对比预测曲线（红色虚线）与真实曲线（黑色实线）。
- **图 2**（`fig2`）：在单位圆上展示切向量的分布，对比预测值与真实值（单位圆上的点表示归一化的切向量）。
- **图 3**（`fig3`）：绘制各节点处位置误差和切向量误差的均值随节点编号的变化曲线：

$$\bar{e}^q_k = \frac{1}{B}\sum_{b=1}^{B} \|\hat{\mathbf{q}}^{(b)}(s_k) - \mathbf{q}^{(b)}(s_k)\|, \quad \bar{e}^{q'}_k = \frac{1}{B}\sum_{b=1}^{B} \|\hat{\mathbf{q}}'^{(b)}(s_k) - \mathbf{v}^{(b)}(s_k)\|$$

各图均保存为 PDF 文件（如 `continuous_80_10_10_qx_qy_test.pdf`）。

---

## 附录：网络结构总览

### 连续神经算子的输入/输出

模型将弹性曲线视为弧长参数 $s$ 的连续函数，将求解弹性曲线问题转化为一个以边界条件为参数的函数逼近问题：

$$\mathcal{N}_\theta : [0,1] \times \mathbb{R}^2 \times \mathbb{R}^2 \times \mathbb{R}^2 \times \mathbb{R}^2 \to \mathbb{R}^2$$

$$(s, \mathbf{q}_1, \mathbf{q}_2, \mathbf{v}_1, \mathbf{v}_2) \mapsto \hat{\mathbf{q}}(s)$$

### 数据流图（MULT 网络）

$$\mathbf{z} = [s, \tilde{\mathbf{q}}_1, \mathbf{q}_2, \mathbf{v}_1, \mathbf{v}_2] \in \mathbb{R}^9 \xrightarrow{\text{embed}+\sin(2\pi\cdot)} \mathbf{h}^{(0)} \in \mathbb{R}^H$$

$$\mathbf{h}^{(0)} \xrightarrow{\text{lift}_{U,V,H}} (U, V, H^{(0)}) \in \mathbb{R}^H \times \mathbb{R}^H \times \mathbb{R}^H$$

$$H^{(0)} \xrightarrow{\text{MULT} \times L} H^{(L)} \in \mathbb{R}^H \xrightarrow{\text{proj}} g(s) \in \mathbb{R}^2$$

### 各模块参数量估算（以 $L=6, H=106$ 为例）

| 模块 | 参数量 |
|---|---|
| `embed`（$9 \to H$）| $9H + H = 10H$ |
| `lift_U`, `lift_V`, `lift_H`（$H \to H$，共 3 个）| $3(H^2 + H)$ |
| `linears_Z`（$L$ 个 $H \to H$）| $L(H^2 + H)$ |
| `linearsO`（$L$ 个 $H \to H$）| $L(H^2 + H)$ |
| `proj`（$H \to 2$）| $2H + 2$ |
| **合计（$L=6, H=106$）** | $10H + 3(H^2+H) + 2L(H^2+H) + 2H + 2$，即 $10 \times 106 + 15 \times (106^2+106) + 2 \times 106 + 2 \approx \mathbf{172{,}000}$ |

> 注：合计行中"$15\times$"由 3 个门控提升层 + 2×6 = 12 个隐藏层（`linears_Z` 和 `linearsO` 各 6 层）共 15 组 $H \times H$ 矩阵组成。

### 训练目标总结

$$\min_\theta \mathbb{E}_{(s, \mathbf{q}_1, \mathbf{q}_2, \mathbf{v}_1, \mathbf{v}_2, \mathbf{q}, \mathbf{v})} \left[ \|\mathcal{N}_\theta(s, \cdot) - \mathbf{q}(s)\|^2 + \left\|\frac{\partial \mathcal{N}_\theta}{\partial s}(s, \cdot) - \mathbf{v}(s)\right\|^2 + \lambda \left(\left\|\frac{\partial \mathcal{N}_\theta}{\partial s}(s, \cdot)\right\|^2 - 1\right)^2 \right]$$

其中 $\lambda = 10^{-2}$，第三项为弧长参数化正则化项，鼓励网络学习到弧长参数化下的单位切向量 $\|\mathbf{q}'(s)\| = 1$。
