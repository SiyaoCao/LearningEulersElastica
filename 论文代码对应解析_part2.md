# 论文公式与代码对应解析 —— Part 2：第 4、5 节及附录 A

本文档续接 [论文代码对应解析_part1.md](./论文代码对应解析_part1.md)，以"代码在前、公式在后"的格式，解读论文第 4、5 节及附录 A 中所有公式与仓库底层计算代码之间的对应关系。所有代码均来自仓库 [SiyaoCao/LearningEulersElastica](https://github.com/SiyaoCao/LearningEulersElastica/tree/copilot/explain-continuousnetwork-main)，代码链接直达具体文件与行号。

---

## 第 4 节：离散网络（The Discrete Network）

### 4.1 数据集 \(\Omega\) 的定义（无编号）

**对应代码——数据加载（DiscreteNetwork/Scripts/GetData.py）：**

GitHub 链接：[DiscreteNetwork/Scripts/GetData.py，第 45–68 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DiscreteNetwork/Scripts/GetData.py#L45)

```python
# DiscreteNetwork/Scripts/GetData.py，第 45–68 行
def getDataLoaders(batch_size, datacase, percentage_train):

    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    _, data_train, data_test = loadData(datacase)

    # 构造输入 x^i：将左端边界条件（前4列）与右端边界条件（后4列）拼接
    # x^i = (q_0^i, (q_0^i)', q_N^i, (q_N^i)') in R^8
    x_full_train = np.concatenate((data_train[:, :4], data_train[:, -4:]), axis=1)

    # 构造输出 y^i：内部节点（去掉首尾边界节点）处的所有位置和切向量
    # y^i = (q_hat_1^i, (q_hat_1^i)', ..., q_hat_{N-1}^i, (q_hat_{N-1}^i)') in R^{4(N-1)}
    y_full_train = data_train[:, 4:-4]

    N = len(x_full_train)
    NTrain = int(percentage_train * N)

    idx_shuffle_train = np.arange(N)
    random.shuffle(idx_shuffle_train)

    x_full_train = x_full_train[idx_shuffle_train]
    y_full_train = y_full_train[idx_shuffle_train]

    x_full_test = np.concatenate((data_test[:, :4], data_test[:, -4:]), axis=1)
    y_full_test  = data_test[:, 4:-4]

    x_full_test = x_full_test[idx_shuffle_train]
    y_full_test  = y_full_test[idx_shuffle_train]

    fact = 0.1
    if percentage_train == 0.8:
        fact = 0.1
    elif percentage_train == 0.7:
        fact = 0.15
    else:
        fact = 0.2

    x_train, y_train = x_full_train[:NTrain], y_full_train[:NTrain]

    Number_Test_Points = int(fact * N)
    x_test, y_test = x_full_test[NTrain:NTrain + Number_Test_Points], \
                     y_full_test[NTrain:NTrain + Number_Test_Points]
    x_val,  y_val  = x_full_test[NTrain + Number_Test_Points:NTrain + 2*Number_Test_Points], \
                     y_full_test[NTrain + Number_Test_Points:NTrain + 2*Number_Test_Points]

    trainset = dataset(x_train, y_train)
    testset  = dataset(x_test,  y_test)
    valset   = dataset(x_val,   y_val)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader  = DataLoader(testset,  batch_size=len(x_test), shuffle=True)
    valloader   = DataLoader(valset,   batch_size=len(x_val),  shuffle=True)

    return x_train, y_train, x_test, y_test, x_val, y_val, trainloader, testloader, valloader
```

**对应公式（离散网络数据集 \(\Omega\)，无编号）：**

\[
\Omega = \left\{ \langle \mathbf{x}^i, \mathbf{y}^i \rangle \right\}_{i=1}^{M}
\]

该数据集由 \(M\) 条预先计算好的弹性曲线离散解组成，其中输入输出分别定义如下（见下文两个无编号公式）。

---

### 4.2 输入向量 \(\mathbf{x}^i\) 的定义（无编号）

**对应代码（DiscreteNetwork/Scripts/GetData.py）：**

GitHub 链接：[DiscreteNetwork/Scripts/GetData.py，第 52 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DiscreteNetwork/Scripts/GetData.py#L52)

```python
# DiscreteNetwork/Scripts/GetData.py，第 52 行
# x^i = (q_0^i, (q_0^i)', q_N^i, (q_N^i)') in R^8
# 拼接左端4个分量（位置+切向量）与右端4个分量（位置+切向量）
x_full_train = np.concatenate((data_train[:, :4], data_train[:, -4:]), axis=1)
```

**对应公式（输入向量，无编号）：**

\[
\mathbf{x}^i = \left(\mathbf{q}_0^i,\, (\mathbf{q}_0^i)^{\prime},\, \mathbf{q}_N^i,\, (\mathbf{q}_N^i)^{\prime}\right) \in \mathbb{R}^{8}
\]

输入为 8 维向量，包含曲线两端的位置向量与切向量（各 2 分量），作为神经网络的边界条件输入。

---

### 4.3 输出向量 \(\mathbf{y}^i\) 的定义（无编号）

**对应代码（DiscreteNetwork/Scripts/GetData.py）：**

GitHub 链接：[DiscreteNetwork/Scripts/GetData.py，第 53 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DiscreteNetwork/Scripts/GetData.py#L53)

```python
# DiscreteNetwork/Scripts/GetData.py，第 53 行
# y^i = (q_hat_1^i, (q_hat_1^i)', ..., q_hat_{N-1}^i, (q_hat_{N-1}^i)') in R^{4(N-1)}
# 去除首尾节点，只取内部节点（index 4 到 -4）
y_full_train = data_train[:, 4:-4]
```

**对应公式（输出向量，无编号）：**

\[
\mathbf{y}^i
= \left(\hat{\mathbf{q}}_1^i,\, (\hat{\mathbf{q}}_1^i)^{\prime},\, \ldots,\,
\hat{\mathbf{q}}_{N-1}^i,\, (\hat{\mathbf{q}}_{N-1}^i)^{\prime}\right)
\in \mathbb{R}^{4(N-1)}
\]

输出为 \(4(N-1)\) 维向量，包含 \(N-1\) 个内部节点处的位置 \(\hat{\mathbf{q}}_k^i\) 和切向量 \((\hat{\mathbf{q}}_k^i)'\)（各 2 分量），作为网络训练的目标数据。

---

### 4.4 离散网络的加权 MSE 损失（公式 11）

**对应代码（DiscreteNetwork/Scripts/Training.py）：**

GitHub 链接：[DiscreteNetwork/Scripts/Training.py，第 37–61 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DiscreteNetwork/Scripts/Training.py#L37)

```python
# DiscreteNetwork/Scripts/Training.py，第 37–61 行
for _, data in enumerate(trainloader):
    inputs, labels = data[0].to(device), data[1].to(device)
    optimizer.zero_grad()
    predicted = model(inputs)

    # 基础 MSE 损失：(1 / 4M(N-1)) * sum ||q_rho^d(x^i) - y^i||^2
    # （PyTorch 的 MSELoss 默认对所有分量取均值，等价于加权后的结果）
    loss = criterion(predicted, labels)

    # 拼接边界节点，构造完整轨迹向量（长度 4N）
    predicted = torch.cat((inputs[:, :4], predicted, inputs[:, 4:]), dim=1)
    labels    = torch.cat((inputs[:, :4], labels,    inputs[:, 4:]), dim=1)

    # 前向差分算子 G = S^4 - I（相邻节点 4 分量差）
    predicted_first   = predicted[:, :-4]      # x[0:4N-4]
    predicted_forward = predicted[:, 4:]       # x[4:4N]
    labels_first      = labels[:, :-4]
    labels_forward    = labels[:, 4:]

    diff_predicted = predicted_forward - predicted_first  # G * q_rho^d(x^i)
    diff_labels    = labels_forward    - labels_first     # G * y^i

    # 加权差分项：gamma * (1/4M(N-1)) * sum ||G*(q_rho^d(x^i) - y^i)||_W^2
    loss += gamma * criterion(diff_predicted, diff_labels)

    train_loss += loss.item()
    loss.backward()
    optimizer.step()
    counter += 1
```

**对应公式（公式 11）：**

\[
\mathrm{Loss}(\boldsymbol{\rho})
= \frac{1}{4M(N-1)} \sum_{i=1}^{M}
\left\| q_{\rho}^d(\mathbf{x}^i) - \mathbf{y}^i \right\|_{W}^{2}
\]

其中 \(q_{\rho}^d : \mathbb{R}^8 \to \mathbb{R}^{4(N-1)}\) 为离散网络（上标 d 代表 discrete），加权范数 \(\|\cdot\|_W^2 = \mathbf{x}^\top W \mathbf{x}\) 由以下权重矩阵定义：

\[
W = I + \gamma G^\top G
\]

其中 \(G = S^4 - I\)，\(S\) 为作用于 \(\mathbb{R}^{4(N-1)}\) 上向量的前向移位算子。代码中 `criterion(predicted, labels)` 对应基础 MSE（即 \(\|q_\rho^d - y^i\|^2\) 的均值项），而 `gamma * criterion(diff_predicted, diff_labels)` 则对应 \(\gamma \|G(q_\rho^d - y^i)\|^2\) 的差分惩罚项。

---

### 4.5 权重矩阵 \(W\)（无编号）

**对应代码（DiscreteNetwork_main.py）：**

GitHub 链接：[DiscreteNetwork_main.py，第 120–160 行（gamma 参数选取与训练调用）](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DiscreteNetwork_main.py#L120)

```python
# DiscreteNetwork_main.py（training 调用部分）
# gamma 来自 Optuna 超参数搜索，用于加权 W = I + gamma * G^T G
def objective(trial):
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    model = define_model(trial).to(device)
    gamma = trial.suggest_float("gamma", 0, 1-1e-2)
    
    # ...（训练调用）
    val_loss = train(model, gamma, criterion, scheduler, optimizer, epochs, trainloader, valloader, device)
    return val_loss
```

GitHub 链接（实际差分运算）：[DiscreteNetwork/Scripts/Training.py，第 44–56 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/DiscreteNetwork/Scripts/Training.py#L44)

```python
# DiscreteNetwork/Scripts/Training.py，第 44–56 行
# G = S^4 - I：对连接了边界条件的完整向量做前向 4 元素移位再做差
predicted_first   = predicted[:, :-4]   # 原始向量（去掉最后 4 个分量）
predicted_forward = predicted[:, 4:]    # 前向移位 4 个分量后的向量

labels_first   = labels[:, :-4]
labels_forward = labels[:, 4:]

diff_predicted = predicted_forward - predicted_first  # (G * predicted)
diff_labels    = labels_forward    - labels_first     # (G * labels)

# Loss += gamma * ||G*(predicted - labels)||^2 / (4M(N-1))
loss += gamma * criterion(diff_predicted, diff_labels)
```

**对应公式（权重矩阵）：**

\[
W = I + \gamma G^{\top} G, \qquad G = S^4 - I
\]

其中 \(S\) 为 \(\mathbb{R}^{4(N-1)}\) 上的前向移位算子（将向量整体向前平移 4 个元素，对应"下一个节点"的 4 个分量）。\(G\) 的作用等价于计算相邻节点（位置和切向量）之间的差分，使损失函数在鼓励逐点准确的同时也鼓励空间平滑性。参数 \(\gamma\) 由 Optuna 超参数搜索确定（典型值约为 \(3.853 \times 10^{-3}\)）。

---

## 第 5 节：连续网络（The Continuous Network）

### 5.1 连续网络数据集 \(\Omega\)（无编号）

**对应代码——数据加载（ContinuousNetwork/Scripts/GetData.py）：**

GitHub 链接：[ContinuousNetwork/Scripts/GetData.py，第 84–143 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetwork/Scripts/GetData.py#L84)

```python
# ContinuousNetwork/Scripts/GetData.py，第 84–143 行
def getDataLoaders(batch_size, datacase, percentage_train):

    num_elements = 50  # N = 50（节点数 N+1 = 51）
    L = 3.3
    x_range = np.linspace(0, L, num_elements + 1) / L  # s_k / L，归一化至 [0,1]

    _, trajectories_train, trajectories_test = loadData(datacase)
    Ntrain_total = len(trajectories_train)
    Ntrain = int(percentage_train * Ntrain_total)

    list_nodes = list(range(num_elements + 1))  # k = 0, 1, ..., N

    data_train = {"q1": [], "q2": [], "v1": [], "v2": [],
                  "s": [], "sample_number": [], "qs": [], "vs": []}
    data_test  = {"q1": [], "q2": [], "v1": [], "v2": [],
                  "s": [], "sample_number": [], "qs": [], "vs": []}

    number_samples = Ntrain_total

    for sample in range(number_samples):
        for node in list_nodes:
            if sample < Ntrain:
                # 弧长坐标 s_k = k * L / N
                data_train["s"].append(x_range[node])
                data_train["sample_number"].append(sample)

                # 边界条件 x^i = (q1, v1, q2, v2) in R^8
                data_train["q1"].append(trajectories_train[sample][0:2])     # q(0)
                data_train["v1"].append(trajectories_train[sample][2:4])     # q'(0)
                data_train["q2"].append(trajectories_train[sample][-4:-2])   # q(L)
                data_train["v2"].append(trajectories_train[sample][-2:])     # q'(L)

                # 节点 s_k 处目标值 y_k^i = (q_hat_k^i, (q_hat_k^i)')
                data_train["qs"].append(trajectories_train[sample][4*node:4*node+2])
                data_train["vs"].append(trajectories_train[sample][4*node+2:4*node+4])
            else:
                # 测试集（与训练集同格式）
                data_test["s"].append(x_range[node])
                data_test["sample_number"].append(sample)
                data_test["q1"].append(trajectories_test[sample][0:2])
                data_test["v1"].append(trajectories_test[sample][2:4])
                data_test["q2"].append(trajectories_test[sample][-4:-2])
                data_test["v2"].append(trajectories_test[sample][-2:])
                data_test["qs"].append(trajectories_test[sample][4*node:4*node+2])
                data_test["vs"].append(trajectories_test[sample][4*node+2:4*node+4])
    # ...（后续转为 numpy 数组并打包为 DataLoader）
```

**对应公式（连续网络数据集 \(\Omega\)，无编号）：**

\[
\Omega
= \left\{ (s_k, \mathbf{x}^i),\, \mathbf{y}_k^i \right\}_{k=0,\ldots,N}^{i=1,\ldots,M}
\]

其中 \(s_k = kL/N\)，\(\mathbf{x}^i \in \mathbb{R}^8\) 为边界条件，\(\mathbf{y}_k^i\) 为节点处的目标值。与离散网络数据集不同，连续网络同时使用所有 \(N+1\) 个节点（包括两端），并将弧长坐标 \(s_k\) 作为额外输入特征。

---

### 5.2 连续网络输入 \(\mathbf{x}^i\) 和目标 \(\mathbf{y}_k^i\)（无编号）

**对应代码（ContinuousNetwork/Scripts/GetData.py）：**

GitHub 链接：[ContinuousNetwork/Scripts/GetData.py，第 100–113 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetwork/Scripts/GetData.py#L100)

```python
# ContinuousNetwork/Scripts/GetData.py，第 100–113 行

# 边界条件输入 x^i = (q_0^i, (q_0^i)', q_N^i, (q_N^i)') in R^8
data_train["q1"].append(trajectories_train[sample][0:2])     # q_0^i
data_train["v1"].append(trajectories_train[sample][2:4])     # (q_0^i)'
data_train["q2"].append(trajectories_train[sample][-4:-2])   # q_N^i
data_train["v2"].append(trajectories_train[sample][-2:])     # (q_N^i)'

# 节点 s_k 处的目标值 y_k^i = (q_hat_k^i, (q_hat_k^i)') in R^4
data_train["qs"].append(trajectories_train[sample][4*node:4*node+2])   # q_hat_k^i
data_train["vs"].append(trajectories_train[sample][4*node+2:4*node+4]) # (q_hat_k^i)'
```

**对应公式（无编号）：**

\[
\mathbf{x}^i
= \left(\mathbf{q}_0^i,\, (\mathbf{q}_0^i)^{\prime},\, \mathbf{q}_N^i,\, (\mathbf{q}_N^i)^{\prime}\right)
\in \mathbb{R}^{8}
\]

\[
\mathbf{y}_k^i
= \left(\hat{\mathbf{q}}_k^i,\, (\hat{\mathbf{q}}_k^i)^{\prime}\right)
\]

\(\hat{\mathbf{q}}_k^i\) 为节点 \(s_k\) 处的数值解，满足第 \(i\) 组边界条件（公式 10）。代码中 `qs` 存储位置，`vs` 存储切向量，分别作为连续网络的训练目标。

---

### 5.3 微分算子 \(D\)（无编号）

**对应代码（ContinuousNetwork/Scripts/Network.py）：**

GitHub 链接：[ContinuousNetwork/Scripts/Network.py，第 53–63 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetwork/Scripts/Network.py#L53)

```python
# ContinuousNetwork/Scripts/Network.py，第 53–63 行
# 微分算子 D：计算网络输出 q_rho^c 关于弧长 s 的导数
def derivative(self, s, q1, q2, v1, v2):
    B = len(q1)

    # 将 forward 包装为以 s 为自变量的函数，以便对 s 求雅可比
    q = lambda s, a, b, c, d: self.forward(
        s.reshape(-1, 1),
        a.reshape(-1, 2),
        b.reshape(-1, 2),
        c.reshape(-1, 2),
        d.reshape(-1, 2)
    )

    if self.is_norm:
        q1n, q2n, v1n, v2n = self.normalize(q1, q2, v1, v2)
        # vmap + jacfwd：对批次中每个样本并行计算 d/ds q_rho^c(x^i)(s)
        return vmap(jacfwd(q, argnums=0))(s, q1n, q2n, v1n, v2n)[:, 0, :, 0]
    else:
        return vmap(jacfwd(q, argnums=0))(s, q1, q2, v1, v2)[:, 0, :, 0]
```

**对应公式（微分算子，无编号）：**

\[
D : C^{\infty}\!\left([0,L],\mathbb{R}^2\right)
\to C^{\infty}\!\left([0,L],\mathbb{R}^2\right),
\qquad
D\!\left(q_{\rho}^c\!\left(\mathbf{x}^i\right)\right)\!(s_k)
= \frac{d}{ds}\!\left(q_{\rho}^c\!\left(\mathbf{x}^i\right)\right)\!(s)\Big|_{s=s_k}
\]

该算子计算连续网络输出 \(q_{\rho}^c(\mathbf{x}^i)\) 关于弧长坐标 \(s\) 的空间导数，在每个节点 \(s_k\) 处求值。代码中通过 `torch.func.jacfwd` 对 `forward` 方法（网络前向传播）做前向自动微分，结合 `vmap` 实现批量化求导，对应公式中的 \(D(q_\rho^c(\mathbf{x}^i))(s_k)\)。

---

### 5.4 连续网络的组合输出 \(y_{\rho}(\mathbf{x}^i)(s_k)\)（无编号）

**对应代码（ContinuousNetwork/Scripts/Training.py）：**

GitHub 链接：[ContinuousNetwork/Scripts/Training.py，第 35–37 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetwork/Scripts/Training.py#L35)

```python
# ContinuousNetwork/Scripts/Training.py，第 35–37 行

# q_rho^c(x^i)(s_k)：网络对位置 q 的预测
res_q = model(s, q1, q2, v1, v2)

# D(q_rho^c(x^i))(s_k)：网络对切向量 q' 的预测（对 s 求导）
# y_rho(x^i)(s_k) = (q_rho^c(x^i)(s_k), D(q_rho^c(x^i))(s_k))
loss = criterion(res_q, qs) \
     + criterion(model.derivative(s, q1, q2, v1, v2), vs)
```

**对应公式（组合输出，无编号）：**

\[
y_{\rho}\!\left(\mathbf{x}^i\right)\!(s_k)
\coloneqq
\left(
q_{\rho}^c\!\left(\mathbf{x}^i\right)\!(s_k),\;
D\!\left(q_{\rho}^c\!\left(\mathbf{x}^i\right)\right)\!(s_k)
\right)
\]

该向量将网络对曲线位置 \(\mathbf{q}(s_k)\) 和切向量 \(\mathbf{q}'(s_k)\) 的同时预测打包为一个整体输出，供损失函数与目标值 \(\mathbf{y}_k^i\) 进行比较。

---

### 5.5 连续网络的损失函数（公式 12）

**对应代码（ContinuousNetwork/Scripts/Training.py）：**

GitHub 链接：[ContinuousNetwork/Scripts/Training.py，第 35–41 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetwork/Scripts/Training.py#L35)

```python
# ContinuousNetwork/Scripts/Training.py，第 35–41 行

optimizer.zero_grad()
res_q = model(s, q1, q2, v1, v2)

# 公式 (12) 三项：
# 1. MSE on q：||q_rho^c(x^i)(s_k) - q_hat_k^i||^2
# 2. MSE on q'：||D(q_rho^c(x^i))(s_k) - (q_hat_k^i)'||^2
# 3. 约束惩罚：gamma * (||D(q_rho^c(x^i))(s_k)||^2 - 1)^2（公式 12 第二项）
loss = criterion(res_q, qs) \
     + criterion(model.derivative(s, q1, q2, v1, v2), vs) \
     + 1e-2 * torch.mean(
           (torch.linalg.norm(model.derivative(s, q1, q2, v1, v2), ord=2, dim=1)**2 - 1.)**2
       )
loss.backward()
optimizer.step()
running_loss += loss.item()
```

其中 `gamma = 1e-2`（即 \(\gamma = 10^{-2}\)），`criterion` 为 PyTorch 的 `nn.MSELoss()`，`model.derivative` 对应微分算子 \(D\) 的数值实现，`torch.linalg.norm(...)**2 - 1.` 即 \(\|\pi_D(y_\rho(\mathbf{x}^i)(s_k))\|_2^2 - 1\)。

**对应公式（公式 12）：**

\[
\mathrm{Loss}(\boldsymbol{\rho})
= \frac{1}{4M(N+1)} \sum_{i=1}^{M} \sum_{k=0}^{N}
\left(
  \left\| y_{\rho}\!\left(\mathbf{x}^i\right)\!(s_k) - \mathbf{y}_k^i \right\|_2^2
  + \gamma \left( \left\| \pi_D\!\left(y_{\rho}\!\left(\mathbf{x}^i\right)\!(s_k)\right) \right\|_2^2 - 1 \right)^2
\right)
\]

其中 \(\pi_D : \mathbb{R}^4 \to \mathbb{R}^2\) 为投影到第二分量 \(D(q_\rho^c(\mathbf{x}^i))(s_k)\)（即切向量预测）的算子，\(\gamma \geq 0\) 为切向量单位化约束的惩罚权重（数值实验中设为 \(10^{-2}\)）。

---

### 5.6 角度神经网络 \(\hat{\theta}_\rho^c\) 的定义（公式 13）

**对应代码（ContinuousNetworkTheta/Scripts/network.py）：**

GitHub 链接：[ContinuousNetworkTheta/Scripts/network.py，第 15–51 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetworkTheta/Scripts/network.py#L15)

```python
# ContinuousNetworkTheta/Scripts/network.py，第 15–51 行
class theta_net(nn.Module):
    """
    theta_hat_rho^c: R^2 -> C^\infty([0,L], R)
    输入为 (theta_0, theta_N) 和弧长 s（合并后 3 维输入），输出为标量角度 theta(s)
    """
    def __init__(self, act_name, nlayers, hidden_nodes, is_res, is_deeponet):
        super().__init__()

        torch.manual_seed(1)
        np.random.seed(1)

        if act_name == 'tanh':
            self.act = lambda x: torch.tanh(x)
        elif act_name == "sin":
            self.act = lambda x: torch.sin(x)
        elif act_name == "swish":
            self.act = lambda x: x * torch.sigmoid(x)
        else:
            self.act = lambda x: torch.sigmoid(x)

        self.hidden_nodes = hidden_nodes
        self.nlayers = nlayers
        self.is_res = is_res
        self.is_deeponet = is_deeponet

        # 嵌入层：(theta_0, theta_N, s) -> R^hidden_nodes
        self.embed = nn.Linear(3, self.hidden_nodes)

        # MULT 架构（DeepONet 乘法门控）
        self.lift_U = nn.Linear(self.hidden_nodes, self.hidden_nodes)
        self.lift_V = nn.Linear(self.hidden_nodes, self.hidden_nodes)
        self.lift_H = nn.Linear(self.hidden_nodes, self.hidden_nodes)
        self.linears_Z = nn.ModuleList([
            nn.Linear(self.hidden_nodes, self.hidden_nodes)
            for i in range(self.nlayers)
        ])

        self.lift = nn.Linear(self.hidden_nodes, self.hidden_nodes)
        self.linears  = nn.ModuleList([nn.Linear(self.hidden_nodes, self.hidden_nodes) for _ in range(self.nlayers)])
        self.linearsO = nn.ModuleList([nn.Linear(self.hidden_nodes, self.hidden_nodes) for _ in range(self.nlayers)])

        # 输出投影：R^hidden_nodes -> R（标量角度）
        self.proj = nn.Linear(self.hidden_nodes, 1)

    def find_theta(self, v):
        """从切向量 v = (vx, vy) 提取角度 theta = atan2(vy, vx)"""
        return torch.atan2(v[:, 1:2], v[:, 0:1])

    def forward(self, s, q1, q2, v1, v2):
        """
        前向传播：输入 (s, theta_0, theta_N)，输出 theta_hat_rho^c(s)
        其中 theta_0 = atan2(v1_y, v1_x)，theta_N = atan2(v2_y, v2_x)
        """
        s     = s.reshape(-1, 1)
        v1    = v1.reshape(-1, 2)
        v2    = v2.reshape(-1, 2)

        # 提取边界切向角并归一化到 [-1, 1]（除以 pi）
        theta1 = self.find_theta(v1) / torch.pi   # theta_0 / pi
        theta2 = self.find_theta(v2) / torch.pi   # theta_N / pi

        # 输入：pi([x^i]) = (theta_0^i, theta_N^i, s)
        input = torch.cat((theta1, theta2, s), dim=1)
        # 正弦嵌入：sin(2 pi * W * input + b)
        input = torch.sin(2 * torch.pi * self.embed(input))
        # ...（后续 MULT 或 MLP 传播，见附录 A）
        output = self.proj(input)
        return output * torch.pi  # 乘以 pi 还原到角度空间
```

**对应公式（公式 13）：**

\[
\hat{\theta}_{\rho}^c : \mathbb{R}^2 \to C^{\infty}\!\left([0,L],\mathbb{R}\right)
\]

\(\hat{\theta}_\rho^c\) 以边界切向角 \((\theta_0, \theta_N)\) 和弧长 \(s\) 为输入，输出沿梁长度方向的角度函数 \(\theta(s)\)。完整的角度网络为 \(\theta_\rho^c = \hat{\theta}_\rho^c \circ \pi\)，其中 \(\pi : \mathbb{R}^8 \to \mathbb{R}^2\) 从边界条件 \(\mathbf{x}^i\) 中提取切向角 \((\theta_0^i, \theta_N^i)\)。代码中 `find_theta` 函数实现 \(\pi\)，`theta_net.forward` 实现 \(\hat{\theta}_\rho^c\)。

---

### 5.7 切向量近似公式（公式 14）

**对应代码（ContinuousNetworkTheta/Scripts/network.py）：**

GitHub 链接：[ContinuousNetworkTheta/Scripts/network.py，第 166–168 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetworkTheta/Scripts/network.py#L166)

```python
# ContinuousNetworkTheta/Scripts/network.py，第 166–168 行
def forward(self, s, q1, q2, v1, v2):
    """
    tau_rho^c(x^i)(s) = (cos(theta_rho^c(x^i)(s)), sin(theta_rho^c(x^i)(s)))
    直接由角度网络 theta 的输出计算切向量，满足 ||tau_rho^c|| = 1
    """
    theta = self.theta(s, q1, q2, v1, v2)
    return torch.cat((torch.cos(theta), torch.sin(theta)), dim=1)
```

其中 `self.theta` 方法对应 \(\theta_\rho^c\)（含或不含边界条件修正，见下文公式 17/18），`torch.cos` 和 `torch.sin` 则对应 \(\cos(\cdot)\) 和 \(\sin(\cdot)\)。

**对应公式（公式 14）：**

\[
\tau_{\rho}^c\!\left(\mathbf{x}^i\right)\!(s)
\coloneqq
\left(
  \cos\!\left(\theta_{\rho}^c\!\left(\mathbf{x}^i\right)\!(s)\right),\;
  \sin\!\left(\theta_{\rho}^c\!\left(\mathbf{x}^i\right)\!(s)\right)
\right)
\in \mathbb{R}^2
\]

由于 \(\cos^2 + \sin^2 = 1\)，切向量 \(\tau_\rho^c\) 的单位范数约束 \(\|\tau_\rho^c(\mathbf{x}^i)(s)\|_2 = 1\) 由网络结构自动满足，无需额外惩罚。

---

### 5.8 曲线位置的重构公式（公式 15）

**对应代码（ContinuousNetworkTheta/Scripts/utils.py）：**

GitHub 链接：[ContinuousNetworkTheta/Scripts/utils.py，第 47–59 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetworkTheta/Scripts/utils.py#L47)

```python
# ContinuousNetworkTheta/Scripts/utils.py，第 47–59 行
def reconstruct_q(q1, q2, v1, v2, model, device):
    """
    q_rho^c(x^i)(s) = q_0 + I(tau_rho^c(x^i))(s)
    对每个节点 s_k，用 3 点 Gauss 积分近似：
    I(tau_rho^c)(s_k) ≈ ∫_0^{s_k} tau_rho^c(x^i)(s_bar) ds_bar
    """
    L = 3.3
    beam_nodes = np.linspace(0, L, num_elements + 1)  # s_k，k = 0,...,N

    q1t = torch.from_numpy(q1.astype(np.float32)).to(device)
    q2t = torch.from_numpy(q2.astype(np.float32)).to(device)
    v1t = torch.from_numpy(v1.astype(np.float32)).to(device)
    v2t = torch.from_numpy(v2.astype(np.float32)).to(device)

    q = np.zeros((len(q1), 2, num_elements + 1))

    for count, upper in enumerate(beam_nodes):
        # x 分量：q_x(s_k) = q1_x + ∫_0^{s_k} tau_x ds_bar
        q[:, 0, count] = reconstruct_q_comp(
            q1t, q2t, v1t, v2t, model, upper, comp=0, k=10
        ).detach().cpu().numpy()
        # y 分量：q_y(s_k) = q1_y + ∫_0^{s_k} tau_y ds_bar
        q[:, 1, count] = reconstruct_q_comp(
            q1t, q2t, v1t, v2t, model, upper, comp=1, k=10
        ).detach().cpu().numpy()

    return q
```

**对应公式（公式 15）：**

\[
q_{\rho}^c\!\left(\mathbf{x}^i\right)\!(s)
= \mathbf{q}_0 + \mathcal{I}\!\left(\tau_{\rho}^c\!\left(\mathbf{x}^i\right)\right)\!(s)
\]

其中积分算子 \(\mathcal{I}\) 定义为 \(\mathcal{I}(\tau_\rho^c(\mathbf{x}^i))(s) \approx \int_0^s \tau_\rho^c(\mathbf{x}^i)(\bar{s})\,d\bar{s}\)（见下文）。代码中 `q1t` 对应 \(\mathbf{q}_0\)（曲线左端点），`reconstruct_q_comp` 的返回值 `q1[:,comp] + integral` 即 \(\mathbf{q}_0 + \mathcal{I}(\tau_\rho^c)\)(s_k)。

---

### 5.9 积分算子 \(\mathcal{I}\)（3 点 Gauss 积分，无编号）

**对应代码（ContinuousNetworkTheta/Scripts/utils.py）：**

GitHub 链接：[ContinuousNetworkTheta/Scripts/utils.py，第 14–43 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetworkTheta/Scripts/utils.py#L14)

```python
# ContinuousNetworkTheta/Scripts/utils.py，第 14–43 行
def reconstruct_q_comp(q1, q2, v1, v2, model, upper, comp=0, k=10):
    """
    对 [0, upper] 分成 k 个子区间，在每个子区间上用 3 点 Gauss 积分
    近似 ∫_0^{upper} tau_rho^c(s) ds（仅计算 comp 分量）
    """
    integrand = lambda s, q1, q2, v1, v2: model(s, q1, q2, v1, v2)[:, comp:comp+1]

    bs = len(v1)
    # k+1 个等间距端点（划分 k 个子区间）
    tt = torch.linspace(0, upper, k + 1).unsqueeze(1).to(q1.device)
    ba = tt[1] - tt[0]  # 子区间宽度

    # 3 点 Gauss 积分权重：w = [8/9, 5/9, 5/9] / 2 * ba（每个子区间）
    w1 = ba * 8/9 / 2 * torch.ones_like(tt[:-1]).to(q1.device)
    w2 = ba * 5/9 / 2 * torch.ones_like(tt[:-1]).to(q1.device)
    w3 = ba * 5/9 / 2 * torch.ones_like(tt[:-1]).to(q1.device)

    # 3 点 Gauss 积分节点（映射到每个子区间）
    x1 = (tt[:-1] + tt[1:]) / 2                                  # 节点 0：子区间中心
    x2 = (tt[:-1] + tt[1:]) / 2 + 0.5 * (-np.sqrt(3/5)) * ba    # 节点 -sqrt(3/5)
    x3 = (tt[:-1] + tt[1:]) / 2 + 0.5 * ( np.sqrt(3/5)) * ba    # 节点 +sqrt(3/5)

    quad_nodes   = torch.cat((x1, x2, x3), dim=0).to(q1.device)
    quad_weights = torch.cat((w1, w2, w3), dim=0).to(q1.device)

    ones_like_nodes = torch.ones_like(quad_nodes).to(q1.device)
    ones_like_bcs   = torch.ones((len(v1), 1)).to(q1.device)

    # 广播：构造每个样本与每个积分节点的组合
    s     = torch.kron(ones_like_bcs, quad_nodes)
    w     = torch.kron(ones_like_bcs, quad_weights)
    q1_kron = torch.kron(q1, ones_like_nodes)
    q2_kron = torch.kron(q2, ones_like_nodes)
    v1_kron = torch.kron(v1, ones_like_nodes)
    v2_kron = torch.kron(v2, ones_like_nodes)

    # q_upper = q1 + sum(integrand(s) * w)
    # 等价于 q(upper) = q(0) + ∫_0^{upper} tau_rho^c(s) ds
    q_upper_comp_left = q1[:, comp] + torch.sum(
        (integrand(s, q1_kron, q2_kron, v1_kron, v2_kron) * w).reshape(bs, 3*k),
        dim=1
    )
    return q_upper_comp_left
```

**对应公式（积分算子 \(\mathcal{I}\)，无编号）：**

\[
\mathcal{I}\!\left(\tau_{\rho}^c\!\left(\mathbf{x}^i\right)\right)\!(s)
\approx \int_0^s \tau_{\rho}^c\!\left(\mathbf{x}^i\right)\!(\bar{s})\,d\bar{s}
\]

积分算子 \(\mathcal{I}\) 通过将区间 \([0, s]\) 分成 \(k = 10\) 个子区间并在每个子区间上应用 3 点 Gauss 积分（节点 \(\{0, \pm\sqrt{3/5}\}\)，权重 \(\{8/9, 5/9, 5/9\}\)）来近似该积分。代码中 `w1`、`w2`、`w3` 为 Gauss 积分权重，`x1`、`x2`、`x3` 为积分节点，`torch.sum(integrand(s,...)*w, dim=1)` 即对应 \(\mathcal{I}(\tau_\rho^c)(s_k)\) 的数值近似。

---

### 5.10 \(\theta\) 网络的组合输出 \(y_\rho(\mathbf{x}^i)(s_k)\)（公式 16）

**对应代码（ContinuousNetworkTheta/Scripts/training.py）：**

GitHub 链接：[ContinuousNetworkTheta/Scripts/training.py，第 36–44 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetworkTheta/Scripts/training.py#L36)

```python
# ContinuousNetworkTheta/Scripts/training.py，第 36–44 行
optimizer.zero_grad()
# model(s, q1, q2, v1, v2) 输出 tau_rho^c(x^i)(s_k)（即公式 14）
loss = criterion(model(s, q1, q2, v1, v2), vs)   # ||tau_rho^c(x^i)(s_k) - (q_hat_k^i)'||^2

# q 比较：q_pred = q_0 + I(tau_rho^c)(s_k)（即公式 15）
q_pred = torch.zeros_like(qs)
q_pred[:, 0] = reconstruct_q_torch(model, s, q1, q2, v1, v2, comp=0, k=10)
q_pred[:, 1] = reconstruct_q_torch(model, s, q1, q2, v1, v2, comp=1, k=10)
loss_q = criterion(q_pred, qs)                    # ||q_rho^c(x^i)(s_k) - q_hat_k^i||^2
loss += loss_q

# 总损失：||y_rho(x^i)(s_k) - y_k^i||^2 = loss_q + loss_tangent
# y_rho(x^i)(s_k) := (q_rho^c(x^i)(s_k), tau_rho^c(x^i)(s_k))
loss.backward()
optimizer.step()
```

**对应公式（公式 16）：**

\[
y_{\rho}\!\left(\mathbf{x}^i\right)\!(s_k)
\coloneqq
\left(
  q_{\rho}^c\!\left(\mathbf{x}^i\right)\!(s_k),\;
  \tau_{\rho}^c\!\left(\mathbf{x}^i\right)\!(s_k)
\right)
\]

其中 \(q_\rho^c\) 的各分量由公式 (15) 的积分重构（代码中 `reconstruct_q_torch`），\(\tau_\rho^c\) 由公式 (14)（代码中 `model(s,...)`）给出。此向量仍用公式 (12) 中的同一损失函数训练，但此时 \(\|\pi_D(y_\rho^c(\mathbf{x}^i)(s))\|_2 = \|\tau_\rho^c(\mathbf{x}^i)(s)\|_2 \equiv 1\)，故约束惩罚系数设为 \(\gamma = 0\)。

---

### 5.11 不强制边界条件的角度参数化（公式 17）

**对应代码（ContinuousNetworkTheta/Scripts/network.py）：**

GitHub 链接：[ContinuousNetworkTheta/Scripts/network.py，第 115–138 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetworkTheta/Scripts/network.py#L115)

```python
# ContinuousNetworkTheta/Scripts/network.py，第 115–138 行
def theta(self, s, q1, q2, v1, v2):
    """
    公式 (17)（不施加边界条件）：
    theta_rho^c(x^i)(s) = f_rho(s, theta_0^i, theta_N^i)
    
    公式 (18)（施加边界条件）：
    theta_rho^c(x^i)(s) = f_rho(s, theta_0^i, theta_N^i)
                         + (theta_0^i - f_rho(0, theta_0^i, theta_N^i)) * exp(-100 s^2)
                         + (theta_N^i - f_rho(L, theta_0^i, theta_N^i)) * exp(-100 (s-L)^2)
    """
    L = 3.3
    s             = s.reshape(-1, 1)
    v1            = v1.reshape(-1, 2)
    v2            = v2.reshape(-1, 2)

    # f_rho(s, theta_0^i, theta_N^i)（参数化网络部分）
    parametric_part = self.parametric_part(s, q1, q2, v1, v2)

    if self.impose_bcs:
        # 公式 (18)：添加高斯型修正项以精确满足边界条件
        left_node  = torch.zeros_like(s).to(s.device)
        right_node = (torch.ones_like(s) * self.L).to(s.device)

        theta_1 = self.find_theta(v1)  # theta_0^i
        theta_2 = self.find_theta(v2)  # theta_N^i

        # 在左右端点处评估参数化网络
        g_left  = self.parametric_part(left_node,  q1, q2, v1, v2).to(s.device)
        g_right = self.parametric_part(right_node, q1, q2, v1, v2).to(s.device)

        # 修正项：强制 theta(0) = theta_0，theta(L) = theta_N
        return (
            parametric_part
            + (theta_1 - g_left)  * torch.exp(-100 * s**2)
            + (theta_2 - g_right) * torch.exp(-100 * (s - self.L)**2)
        )
    else:
        # 公式 (17)：不施加边界条件
        return parametric_part
```

**对应公式（公式 17，不强制边界条件）：**

\[
\hat{\theta}_{\rho}^c\!\left(\mathbf{x}^i\right)\!(s)
= f_{\rho}(s,\,\theta_0^i,\,\theta_N^i)
\]

此时 \(\theta_\rho^c\) 直接由参数化网络 \(f_\rho\)（即 `theta_net.forward`）给出，边界条件仅通过损失函数（弱）施加。代码中当 `self.impose_bcs = False` 时，`theta` 方法直接返回 `parametric_part`，即对应公式 (17)。

---

### 5.12 强制边界条件的角度参数化（公式 18）

**对应代码（ContinuousNetworkTheta/Scripts/network.py）：**

GitHub 链接：[ContinuousNetworkTheta/Scripts/network.py，第 125–136 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetworkTheta/Scripts/network.py#L125)

```python
# ContinuousNetworkTheta/Scripts/network.py，第 125–136 行（impose_bcs=True 分支）
if self.impose_bcs:
    left_node  = torch.zeros_like(s).to(s.device)
    right_node = (torch.ones_like(s) * self.L).to(s.device)

    theta_1 = self.find_theta(v1)  # theta_0^i = atan2(v1_y, v1_x)
    theta_2 = self.find_theta(v2)  # theta_N^i = atan2(v2_y, v2_x)

    # 网络在左/右端点处的预测值
    g_left  = self.parametric_part(left_node,  q1, q2, v1, v2).to(s.device)
    g_right = self.parametric_part(right_node, q1, q2, v1, v2).to(s.device)

    return (
        parametric_part
        + (theta_1 - g_left)  * torch.exp(-100 * s**2)            # 左端修正：(theta_0 - f_rho(0,...)) * e^{-100s^2}
        + (theta_2 - g_right) * torch.exp(-100 * (s - self.L)**2) # 右端修正：(theta_N - f_rho(L,...)) * e^{-100(s-L)^2}
    )
```

**对应公式（公式 18，强制边界条件）：**

\[
\hat{\theta}_{\rho}^c\!\left(\mathbf{x}^i\right)\!(s)
= f_{\rho}(s,\,\theta_0^i,\,\theta_N^i)
+ \left(\theta_0^i - f_{\rho}(0,\,\theta_0^i,\,\theta_N^i)\right) e^{-100s^2}
+ \left(\theta_N^i - f_{\rho}(L,\,\theta_0^i,\,\theta_N^i)\right) e^{-100(s-L)^2}
\]

两个高斯衰减项在端点处"修正"网络输出，精确满足 \(\theta_\rho^c(\mathbf{x}^i)(0) = \theta_0^i\) 和 \(\theta_\rho^c(\mathbf{x}^i)(L) = \theta_N^i\)（精度达到机器精度），原因在于高斯函数的快速衰减。代码中 `torch.exp(-100 * s**2)` 和 `torch.exp(-100 * (s - self.L)**2)` 即为两个 Gauss 修正项。

---

## 附录 A：连续网络的乘法型（MULT）架构（公式 A.1–A.6）

### A.1 MULT 架构的前向传播（公式 A.1–A.6）

**对应代码——连续网络（ContinuousNetwork/Scripts/Network.py）：**

GitHub 链接：[ContinuousNetwork/Scripts/Network.py，第 77–104 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetwork/Scripts/Network.py#L77)

```python
# ContinuousNetwork/Scripts/Network.py，第 77–104 行
def parametric_part(self, s, q1, q2, v1, v2):
    """
    MULT 架构（is_mult=True）与 MLP/ResNet（is_mult=False）的统一实现。
    """
    # 输入特征：(s, q1, q2, v1, v2) 拼接为 9 维向量
    input = torch.cat((s, q1, q2, v1, v2), dim=1)

    # 正弦嵌入（Fourier feature）
    input = torch.sin(2 * torch.pi * self.embed(input))  # dim: hidden_nodes

    if self.is_mult:  # ===== MULT 架构 =====

        # 公式 (A.1): U = sigma(W_1 * x + b_1)
        U = self.act(self.lift_U(input))

        # 公式 (A.2): V = sigma(W_2 * x + b_2)
        V = self.act(self.lift_V(input))

        # 初始隐藏层
        # 公式 (A.3): H_1 = sigma(W_3 * x + b_3)
        H = self.act(self.lift_H(input))

        for i in range(self.nlayers):
            # 公式 (A.4): Z_j = sigma(W_j^T * H_j + b_j^T)
            Z = self.linearsO[i](self.act(self.linears_Z[i](H)))

            # 公式 (A.5): H_{j+1} = (1 - Z_j) ⊙ U + Z_j ⊙ V
            H = U * (1 - Z) + V * Z

        input = H

    else:  # ===== MLP / ResNet 架构 =====

        input = self.act(self.lift(input))

        for i in range(self.nlayers):
            if self.is_res:
                input = input + self.linearsO[i](self.act(self.linears[i](input)))
            else:
                input = self.act(self.linears[i](input))

    # 公式 (A.6): f_rho^MULT(x) = W * H_{l+1} + b
    output = self.proj(input)
    return output
```

**对应代码——角度网络（ContinuousNetworkTheta/Scripts/network.py）：**

GitHub 链接：[ContinuousNetworkTheta/Scripts/network.py，第 71–81 行](https://github.com/SiyaoCao/LearningEulersElastica/blob/copilot/explain-continuousnetwork-main/ContinuousNetworkTheta/Scripts/network.py#L71)

```python
# ContinuousNetworkTheta/Scripts/network.py，第 71–81 行（DeepONet/MULT 分支）
if self.is_deeponet:
    # 公式 (A.1): U = sigma(W_1 * x + b_1)
    U = self.act(self.lift_U(input))

    # 公式 (A.2): V = sigma(W_2 * x + b_2)
    V = self.act(self.lift_V(input))

    # 公式 (A.3): H_1 = sigma(W_3 * x + b_3)
    H = self.act(self.lift_H(input))

    for i in range(self.nlayers):
        # 公式 (A.4): Z_j = sigma(W_j^T * H_j + b_j^T)
        Z = self.linearsO[i](self.act(self.linears_Z[i](H)))

        # 公式 (A.5): H_{j+1} = (1 - Z_j) ⊙ U + Z_j ⊙ V
        H = U * (1 - Z) + V * Z

    input = H
```

**对应公式（公式 A.1–A.6）：**

\[
\mathbf{U} = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \tag{A.1}
\]
\[
\mathbf{V} = \sigma(\mathbf{W}_2 \mathbf{x} + \mathbf{b}_2), \tag{A.2}
\]
\[
\mathbf{H}_1 = \sigma(\mathbf{W}_3 \mathbf{x} + \mathbf{b}_3), \tag{A.3}
\]
\[
\mathbf{Z}_j = \sigma\!\left(\mathbf{W}_j^T \mathbf{H}_j + \mathbf{b}_j^T\right),\quad j = 1,\ldots,\ell, \tag{A.4}
\]
\[
\mathbf{H}_{j+1} = (1 - \mathbf{Z}_j) \odot \mathbf{U} + \mathbf{Z}_j \odot \mathbf{V},\quad j = 1,\ldots,\ell, \tag{A.5}
\]
\[
\mathbf{f}_{\rho}^{\mathrm{MULT}}(\mathbf{x}) = \mathbf{W} \mathbf{H}_{\ell+1} + \mathbf{b}, \tag{A.6}
\]

其中 \(\odot\) 为逐分量乘积（Hadamard 积）。该 MULT（乘法型）架构受神经注意力机制启发，通过两个"门控"向量 \(\mathbf{U}\) 和 \(\mathbf{V}\)（分别来自 `lift_U`、`lift_V`）的插值 \((1-\mathbf{Z}_j) \odot \mathbf{U} + \mathbf{Z}_j \odot \mathbf{V}\) 替代标准的仿射变换加激活，使得网络能够捕获变量间的乘法交互，有效改善梯度行为。参数集合为 \(\rho = \{\mathbf{W}_1, \mathbf{b}_1, \mathbf{W}_2, \mathbf{b}_2, \mathbf{W}_3, \mathbf{b}_3, (\mathbf{W}_j^T, \mathbf{b}_j^T)_{j=1}^\ell, \mathbf{W}, \mathbf{b}\}\)，各权重矩阵和偏置向量的形状确保上述各式在维度上相容。

---

*以上为论文 *Neural networks for the approximation of Euler's elastica* 所有章节公式与仓库底层计算代码的完整对应解析。Part 1（第 2、3 节）请见 [论文代码对应解析_part1.md](./论文代码对应解析_part1.md)。*
