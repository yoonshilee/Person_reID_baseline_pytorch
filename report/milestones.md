# Person ReID 实验里程碑计划（可交互）

## 总览

- [x] M1 环境与依赖准备
- [x] M2 数据集准备（Market-1501）
- [x] M3 Baseline 训练（ResNet50）
- [x] M4 特征提取与测试集前向
- [x] M5 性能评估（CMC 与 mAP）
- [x] M6 检索可视化验证
- [ ] M7 扩展实验（可选）
- [ ] M8 实验记录与报告提交

## 里程碑详情

### M1 环境与依赖准备

- 目标：完成代码与运行环境准备，确保可进入训练流程。
- 步骤：
  - [x] 克隆项目并进入目录。
  - [x] 安装 PyTorch 与 `requirements.txt` 依赖。
  - [x] 确认 GPU 可用（如 `nvidia-smi`）。
- 验收标准：
  - [x] 可以正常执行 Python 脚本且无依赖报错。
  - [x] 训练命令可启动到数据读取阶段。

### M2 数据集准备（Market-1501）

- 目标：按教程完成数据下载与目录重组。
- 步骤：
  - [x] 下载 Market-1501 数据集。
  - [x] 运行 `python prepare.py`（按本机路径修改脚本中的数据路径）。
  - [x] 检查 `pytorch/train`、`pytorch/val`、`pytorch/train_all`、`pytorch/query`、`pytorch/gallery` 是否生成。
- 验收标准：
  - [x] `./data/Market-1501-v15.09.15/pytorch` 目录结构完整。
  - [x] 训练脚本可正确识别类别数与样本。

### M3 Baseline 训练（ResNet50）

- 目标：完成基线模型训练并保存权重。
- 步骤：
  - [x] 执行命令：`python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32 --data_dir ./data/Market-1501-v15.09.15/pytorch`
  - [x] 观察训练日志中的 train/val Loss 与 Acc 收敛情况。
  - [x] 检查 `./model/ft_ResNet50` 下权重与配置文件。
- 验收标准：
  - [x] 训练跑满设定 epoch。
  - [x] 生成 `net_010.pth ... net_060.pth` 与 `net_last.pth`。
  - [x] 生成 `opts.yaml` 与训练曲线图 `train.jpg`。

- 训练记录摘要（来自 `report/record.md`）：
  - [x] 训练区间：`Epoch 0/59` 到 `Epoch 59/59`，总耗时 `33m 9s`。
  - [x] 最终指标（Epoch 59）：`train_loss=0.0209`，`train_acc=0.9994`，`val_loss=0.0062`，`val_acc=0.9800`。
  - [x] 最佳验证准确率：`0.9800`（首次出现在 Epoch 12，后续基本保持）。
  - [x] 最佳验证损失：`0.0053`（Epoch 41）。
  - [x] 关键收敛节点：
    - Epoch 0：`val_acc=0.5246`
    - Epoch 2：`val_acc=0.8535`
    - Epoch 7：`val_acc=0.9734`
    - Epoch 12：`val_acc=0.9800`
  - [x] 产物确认：`./model/ft_ResNet50/` 下存在 `net_010.pth ... net_060.pth`、`net_last.pth`、`opts.yaml`、`train.jpg`。
  - [x] 运行备注：日志中出现 `torchvision pretrained` 弃用警告；并出现一处重复的 `Epoch 58/59` 输出行，但不影响训练完成与结果一致性。

### M4 特征提取与测试集前向

- 目标：使用训练好的模型提取 query/gallery 特征。
- 步骤：
  - [x] 执行：`python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir ./data/Market-1501-v15.09.15/pytorch --batchsize 32 --which_epoch 060`
  - [x] 确认测试过程无路径或权重加载错误。
- 验收标准：
  - [x] 测试脚本运行完成。
  - [x] 成功产出评估所需特征结果文件。

- 执行命令（复制即用）：

```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir ./data/Market-1501-v15.09.15/pytorch --batchsize 32 --which_epoch 060
```

- 备选命令（直接加载最后权重）：

```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir ./data/Market-1501-v15.09.15/pytorch --batchsize 32 --which_epoch last
```

- 记录模板（执行后填写）：
  - [x] 执行时间：`2026-02-28`
  - [x] 使用权重：`which_epoch=060`
  - [x] 运行状态：`成功`
  - [x] 特征文件路径：`pytorch_result.mat`
  - [x] 备注（报错或警告）：`torchvision pretrained 参数弃用警告`

- 终端关键输出粘贴区：

```text
19732it [00:29, 670.56it/s]
3368it [00:05, 577.51it/s]
Training complete in 0m 35.26s
ft_ResNet50
torch.Size([3368, 512])
Rank@1:0.877375 Rank@5:0.956057 Rank@10:0.972387 mAP:0.721856
```

### M5 性能评估（CMC 与 mAP）

- 目标：得到可汇报的检索性能指标。
- 步骤：
  - [x] 执行：`python evaluate_gpu.py`（由 `test.py` 自动调用完成）
  - [x] 记录 Rank-1、Rank-5、Rank-10 与 mAP 指标。
- 验收标准：
  - [x] 成功输出 CMC/mAP。
  - [x] 指标可与仓库 baseline 结果进行对比。

- 执行命令（复制即用）：

```bash
python evaluate_gpu.py
```

- 指标填写区（执行后填写）：

| 指标 | 本次结果 | Baseline/参考 | 结论 |
| --- | ---: | ---: | --- |
| Rank-1 | 0.877375 | 待填写 | 达到可用水平 |
| Rank-5 | 0.956057 | 待填写 | 达到可用水平 |
| Rank-10 | 0.972387 | 待填写 | 达到可用水平 |
| mAP | 0.721856 | 待填写 | 达到可用水平 |

- 运行记录：
  - [x] 执行时间：`2026-02-28`
  - [x] 运行状态：`成功`
  - [x] 结果文件路径（如有）：`./model/ft_ResNet50/result.txt`
  - [x] 异常与修复：`已修复 subprocess 解释器不一致问题，评估可正常执行`

- 终端关键输出粘贴区：

```text
torch.Size([3368, 512])
Rank@1:0.877375 Rank@5:0.956057 Rank@10:0.972387 mAP:0.721856
```

### M6 检索可视化验证

- 目标：通过可视化确认模型检索效果。
- 步骤：
  - [x] 执行：`python demo.py --query_index 777`（可替换不同 query）。
  - [x] 观察 Top-K 返回结果中正负样本分布。
- 验收标准：
  - [x] 可正常显示或打印 Top-10 检索结果。
  - [x] 能解释至少 2 个成功匹配与 2 个失败匹配案例。

- Duke 成功/失败案例（基于最近一次 Duke 测试生成的 `./pytorch_result.mat`）：
  - Success-1（Top-10 全为正样本）：
    - Query：`query_index=0`，`label=5`，`cam=2`
    - Query path：`./data/DukeMTMC-reID/pytorch/query/0005/0005_c2_f0046985.jpg`
    - Top-1：`./data/DukeMTMC-reID/pytorch/gallery/0005/0005_c5_f0052021.jpg`
    - Top-10 正样本数：`10/10`
    - 可视化：`./report/pics/show_success1.png`
  - Success-2（Top-10 全为正样本）：
    - Query：`query_index=1`，`label=5`，`cam=5`
    - Query path：`./data/DukeMTMC-reID/pytorch/query/0005/0005_c5_f0051781.jpg`
    - Top-1：`./data/DukeMTMC-reID/pytorch/gallery/0005/0005_c2_f0047345.jpg`
    - Top-10 正样本数：`10/10`
    - 可视化：`./report/pics/show_success2.png`
  - Failure-1：见 `M7.7 Case-1`（可视化 `./report/pics/show_case1.png`）
  - Failure-2：见 `M7.7 Case-2`（可视化 `./report/pics/show_case2.png`）

### M7 扩展实验（可选）

- 目标：基于 DukeMTMC-reID 完成 README Part4 提到的可选方向，并形成可对比的实验矩阵。

- 统一记录规范（本文件内记录）：
  - [x] 每个实验记录：命令、环境、权重路径、Rank@1/Rank@5/Rank@10/mAP、失败案例、结论。
  - [x] 总表位置：本文件 `M7.8 Duke 实验总表`。

#### M7.1 DukeMTMC 数据准备

- 执行命令（逐步）：

```bash
# 1) 确认压缩包已解压（你已执行）
tar -xf .\DukeMTMC-reID.zip

# 2) 运行数据整理脚本（默认会按 ./data/DukeMTMC-reID 处理）
python prepare_Duke.py

# 3) 目录验收（PowerShell）
Get-ChildItem .\data\DukeMTMC-reID\pytorch
```

- 步骤：
  - [x] 下载 DukeMTMC-reID 数据集。
  - [x] 运行 `python prepare_Duke.py`（或按 README 修改 `prepare.py`）。
  - [x] 确认目录包含 `train`、`val`、`train_all`、`query`、`gallery`。
- 记录：
  - 日期：`2026-02-28`
  - 结论：`Duke 数据准备完成并可被 train.py/test.py 正常读取`
- 验收标准：
  - [x] Duke 的 `pytorch` 结构完整。
  - [x] 可被 `train.py/test.py` 正常读取。

#### M7.2 跨域泛化测试（Market 训练模型直接测 Duke）

- 目的：回答“Market 训练模型能否直接泛化到 Duke”。
- 执行命令（逐步）：

```bash
# 1) 在 Duke 上用 Market 权重提特征并自动评估
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir ./data/DukeMTMC-reID/pytorch --batchsize 32 --which_epoch 060

# 2) 如需单独复跑评估
python evaluate_gpu.py
```

- 步骤：
  - [x] 用当前 `ft_ResNet50` 权重在 Duke 上提特征。
  - [x] 在 Duke 上评估 CMC/mAP。
  - [x] 与 Market 上结果对比，分析域偏移影响。
- 记录区：
  - Duke Rank@1：`0.329892`
  - Duke Rank@5：`0.485189`
  - Duke Rank@10：`0.548474`
  - Duke mAP：`0.170039`
  - 与 Market 差异结论：`相较 Market（Rank@1=0.877375, mAP=0.721856）明显下降，说明直接跨域泛化能力较弱，存在显著域偏移。`
  - 本次运行信息：`python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir ./data/DukeMTMC-reID/pytorch --batchsize 32 --which_epoch 060`，`Exit Code: 0`
- 验收标准：
  - [x] 得到 Duke 上的 Rank@1 与 mAP。
  - [x] 给出“是否可直接泛化”的结论和证据。

#### M7.3 Duke Baseline 复现（ResNet50）

- 目的：建立 Duke 自身 baseline，作为后续变体参照。
- 执行命令（逐步）：

```bash
# 1) 训练 Duke baseline
python train.py --gpu_ids 0 --name duke_ft_ResNet50 --train_all --batchsize 32 --data_dir ./data/DukeMTMC-reID/pytorch

# 2) 测试并自动评估
python test.py --gpu_ids 0 --name duke_ft_ResNet50 --test_dir ./data/DukeMTMC-reID/pytorch --batchsize 32 --which_epoch 060

# 3) （可选）单独评估
python evaluate_gpu.py
```

- 步骤：
  - [x] 在 Duke 上训练 `ft_ResNet50` baseline。
  - [x] 在 Duke 上完成 test + evaluate。
  - [x] 保存权重、曲线、指标与关键日志。
- 记录区：
  - 训练日期：`2026-02-28`
  - 训练命令：`python train.py --gpu_ids 0 --name duke_ft_ResNet50 --train_all --batchsize 32 --data_dir ./data/DukeMTMC-reID/pytorch`
  - 训练配置摘要：`nclasses=702`，`batchsize=32`，`lr=0.05`，`total_epoch=60`，`droprate=0.5`
  - 训练耗时（从 train.py 日志填写）：`113m 35s`
  - 产物确认：`net_010.pth ... net_060.pth`，`net_last.pth`，`opts.yaml`，`train.jpg`
  - 测试日期：`2026-02-28`
  - 测试命令：`python test.py --gpu_ids 0 --name duke_ft_ResNet50 --test_dir ./data/DukeMTMC-reID/pytorch --batchsize 32 --which_epoch 060`
  - Rank@1：`0.793537`
  - Rank@5：`0.889587`
  - Rank@10：`0.920108`
  - mAP：`0.617418`
  - 权重目录：`./model/duke_ft_ResNet50/`
  - 备注：`Exit Code: 0；torchvision pretrained 参数弃用警告`

- 终端关键输出粘贴区：

```text
duke_ft_ResNet50
torch.Size([2228, 512])
Rank@1:0.793537 Rank@5:0.889587 Rank@10:0.920108 mAP:0.617418
```

- 验收标准：
  - [x] 得到 Duke baseline 的 Rank@1/5/10 和 mAP。
  - [x] 有可复现实验命令与权重路径。

#### M7.4 Backbone 变体（README Part4）

- 覆盖项：`--use_dense`、`--use_hr`（至少各 1 次完整训练-评估）。
- 执行命令（DenseNet）：

```bash
python train.py --gpu_ids 0 --name duke_ft_Dense --train_all --batchsize 32 --data_dir ./data/DukeMTMC-reID/pytorch --use_dense
python test.py --gpu_ids 0 --name duke_ft_Dense --test_dir ./data/DukeMTMC-reID/pytorch --batchsize 32 --which_epoch 060 --use_dense
```

- 执行命令（HRNet）：

```bash
python train.py --gpu_ids 0 --name duke_ft_HR --train_all --batchsize 32 --data_dir ./data/DukeMTMC-reID/pytorch --use_hr
python test.py --gpu_ids 0 --name duke_ft_HR --test_dir ./data/DukeMTMC-reID/pytorch --batchsize 32 --which_epoch 060 --use_hr
```


- 步骤：
  - [x] DenseNet 方案：训练、测试、评估。
  - [x] HRNet 方案：训练、测试、评估。
  - [x] 与 Duke baseline 对比性能与速度。
- 记录区：
  - Dense 训练日期：`2026-02-28`
  - Dense 训练耗时（从 train.py 日志填写）：`138m 6s`
  - Dense 测试日期：`2026-02-28`
  - Dense 训练命令：`python train.py --gpu_ids 0 --name duke_ft_Dense --train_all --batchsize 32 --data_dir ./data/DukeMTMC-reID/pytorch --use_dense`
  - Dense 测试命令：`python test.py --gpu_ids 0 --name duke_ft_Dense --test_dir ./data/DukeMTMC-reID/pytorch --batchsize 32 --which_epoch 060 --use_dense`
  - Dense Rank@1/mAP：`Rank@1=0.815081`，`mAP=0.648376`
  - Dense Rank@5/10：`Rank@5=0.912926`，`Rank@10=0.936715`
  - HR 训练日期：`2026-02-28`
  - HR 训练耗时（从 train.py 日志填写）：
  - HR 测试日期：`2026-02-28`
  - HR 训练命令：`python train.py --gpu_ids 0 --name duke_ft_HR --train_all --batchsize 32 --data_dir ./data/DukeMTMC-reID/pytorch --use_hr`
  - HR 测试命令：`python test.py --gpu_ids 0 --name duke_ft_HR --test_dir ./data/DukeMTMC-reID/pytorch --batchsize 32 --which_epoch 060 --use_hr`
  - HR Rank@1/mAP：`Rank@1=0.838869`，`mAP=0.694562`
  - HR Rank@5/10：`Rank@5=0.924147`，`Rank@10=0.942101`
  - 结论：`HRNet 在 Duke 上当前最好（Rank@1 与 mAP 均高于 ResNet50 baseline 与 DenseNet）。`
  - 开销对比（训练时长/显存）：
    - ResNet50 训练耗时：113m 35s
    - DenseNet 训练耗时：138m 6s
    - HRNet 训练耗时：
    - 显存峰值（可选，MB）：

- Dense 终端关键输出粘贴区：

```text
duke_ft_Dense
torch.Size([2228, 512])
Rank@1:0.815081 Rank@5:0.912926 Rank@10:0.936715 mAP:0.648376
```

- HR 终端关键输出粘贴区：

```text
duke_ft_HR
torch.Size([2228, 512])
Rank@1:0.838869 Rank@5:0.924147 Rank@10:0.942101 mAP:0.694562
```

- 验收标准：
  - [x] 两个 backbone 均有完整指标。
  - [x] 有“性能-开销”对比结论。

#### M7.5 Loss 变体（README Part4）

- 覆盖项：`--circle`、`--instance`、`triplet`（可用本仓库 triplet 开关，或按 README 外部 triplet 项目复现）。
- 执行命令（Circle）：

```bash
python train.py --gpu_ids 0 --name duke_ft_circle --train_all --batchsize 32 --data_dir ./data/DukeMTMC-reID/pytorch --circle
python test.py --gpu_ids 0 --name duke_ft_circle --test_dir ./data/DukeMTMC-reID/pytorch --batchsize 32 --which_epoch 060 --circle
```

- 执行命令（Instance）：

```bash
python train.py --gpu_ids 0 --name duke_ft_instance --train_all --batchsize 32 --data_dir ./data/DukeMTMC-reID/pytorch --instance
python test.py --gpu_ids 0 --name duke_ft_instance --test_dir ./data/DukeMTMC-reID/pytorch --batchsize 32 --which_epoch 060 --instance
```

- 执行命令（Triplet）：

```bash
python train.py --gpu_ids 0 --name duke_ft_triplet --train_all --batchsize 32 --data_dir ./data/DukeMTMC-reID/pytorch --triplet
python test.py --gpu_ids 0 --name duke_ft_triplet --test_dir ./data/DukeMTMC-reID/pytorch --batchsize 32 --which_epoch 060 --triplet
```

- 步骤：
  - [x] Circle Loss 实验。
  - [x] Instance Loss 实验。
  - [x] Triplet Loss 实验（本仓库开关）。
  - [x] 与 Duke baseline 做公平对比（尽量保持同 batchsize/lr/epoch）。
- 记录区：
  - Circle 训练日期：`2026-02-28`
  - Circle 训练命令：`python train.py --gpu_ids 0 --name duke_ft_circle --train_all --batchsize 32 --data_dir ./data/DukeMTMC-reID/pytorch --circle`
  - Circle 训练耗时（从 train.py 日志填写）：`123m 42s`
  - Circle 权重目录：`./model/duke_ft_circle/`
  - Circle 备注：`torchvision pretrained 参数弃用警告；GradScaler FutureWarning`
  - Circle 测试日期：`2026-02-28`
  - Circle 测试命令：`python test.py --gpu_ids 0 --name duke_ft_circle --test_dir ./data/DukeMTMC-reID/pytorch --batchsize 32 --which_epoch 060 --circle`
  - Circle which_epoch：`060`
  - Instance 训练日期：`2026-02-28`
  - Instance 训练命令：`python train.py --gpu_ids 0 --name duke_ft_instance --train_all --batchsize 32 --data_dir ./data/DukeMTMC-reID/pytorch --instance`
  - Instance 训练耗时（从 train.py 日志填写）：`116m 27s`
  - Instance 权重目录：`./model/duke_ft_instance/`
  - Instance 备注：`torchvision pretrained 参数弃用警告；GradScaler FutureWarning`
  - Instance 测试日期：`2026-02-28`
  - Instance 测试命令：`python test.py --gpu_ids 0 --name duke_ft_instance --test_dir ./data/DukeMTMC-reID/pytorch --batchsize 32 --which_epoch 060 --instance`
  - Instance which_epoch：`060`
  - Triplet 训练日期：`2026-02-28`
  - Triplet 训练命令：`python train.py --gpu_ids 0 --name duke_ft_triplet --train_all --batchsize 32 --data_dir ./data/DukeMTMC-reID/pytorch --triplet`
  - Triplet 训练耗时（从 train.py 日志填写）：`122m 12s`
  - Triplet 权重目录：`./model/duke_ft_triplet/`
  - Triplet 备注：`torchvision pretrained 参数弃用警告；GradScaler FutureWarning`
  - Triplet 测试日期：`2026-02-28`
  - Triplet 测试命令：`python test.py --gpu_ids 0 --name duke_ft_triplet --test_dir ./data/DukeMTMC-reID/pytorch --batchsize 32 --which_epoch 060 --triplet`
  - Triplet which_epoch：`060`
  - 显存峰值（可选，MB）：
  - Circle Rank@1/mAP：`Rank@1=0.782765`，`mAP=0.612843`
  - Circle Rank@5/10：`Rank@5=0.884201`，`Rank@10=0.912478`
  - Instance Rank@1/mAP：`Rank@1=0.800718`，`mAP=0.623174`
  - Instance Rank@5/10：`Rank@5=0.889587`，`Rank@10=0.918312`
  - Triplet Rank@1/mAP：`Rank@1=0.793088`，`mAP=0.623227`
  - Triplet Rank@5/10：`Rank@5=0.887792`，`Rank@10=0.921454`
  - 公平对比说明：`Circle/Instance/Triplet 均使用同数据（Duke train_all）、batchsize=32、lr=0.05、total_epoch=60、which_epoch=060，backbone=ResNet50（未启用 use_dense/use_hr）。`
  - 对比结论：`Circle 在 Duke 上略低于 baseline（mAP 0.612843 < 0.617418）；Instance/Triplet 在 mAP 上均有小幅提升（约 +0.006），其中 Triplet mAP 略高但差距很小；若更看重 Rank@1，则 Instance 更高（0.800718）。`
  - 最优 loss：`按 mAP 选择 Triplet（0.623227），按 Rank@1 选择 Instance（0.800718）；两者差距很小，建议结合多次重复或更强评估（rerank）再定最终结论。`
- 验收标准：
  - [x] 三种 loss 至少完成 2 种（目标全做）。
  - [x] 明确指出哪种 loss 对 Duke 提升最明显。

#### M7.6 Verification + Identification（README Part4）

- 目标：尝试 README 指向的 verification+identification 方向并形成结论。
- 执行命令（按外部仓库说明）：

```bash
# 1) 克隆项目
git clone https://github.com/layumi/Person-reID-verification.git

# 2) 进入项目并按其 README 配置后运行训练/测试
# （在该仓库中执行对应命令）
```

- 步骤：
  - [ ] 阅读并复现 `Person-reID-verification` 的最小流程。
  - [ ] 对齐可比较指标（至少 Rank@1 和 mAP 或等效指标）。
  - [ ] 对比 baseline：是否提升、代价是什么。
- 参考仓库：`https://github.com/layumi/Person-reID-verification`
- 记录区：
  - 复现是否成功：
  - 关键指标：
  - 与 baseline 差异：
- 验收标准：
  - [ ] 有可复现过程记录。
  - [ ] 有与 baseline 的定量/定性对比。

#### M7.7 失败案例分析（报告硬性要求）

- 目标：至少提取 2 个失败案例，并解释错误原因。

- 判定说明（如何区分“异人误匹配”与“同人误检/同人未检出”）：
  - 异人误匹配（False Positive in Top-K）：
    - 定义：在 Top-K（如 Top-1/Top-10）返回结果中，高排名样本属于错误身份。
    - 可操作判据：`Rank-1` 的 gallery 身份 `label != query_label`；或 Top-K 中错误身份占据多数。
    - 直观现象：视觉上 query 与 Top-1/Top-K 目标外观相似（衣着、姿态、背景），但身份不同。
  - 同人未检出/同人误检（漏检，False Negative in Top-K）：
    - 定义：正确身份（同一 label）的 gallery 样本存在，但没有出现在 Top-K 内，或排名非常靠后。
    - 可操作判据：Top-K 内正样本数为 0（`topK_positives=0`），或“首个正确匹配出现的 rank”远大于 K（如 `first_positive_rank >> 10`）。
    - 直观现象：模型没有把同一人（跨相机、遮挡、模糊、光照变化）排到前列。
  - 备注：同一个 query 往往会同时满足两类（Top-1 错 + Top-10 无正样本），写报告时建议按“主要现象”归类：
    - 若强调“Top-1/Top-K 被错误身份占据”，归为“异人误匹配”。
    - 若强调“正确身份存在但排得很靠后/Top-K 完全没有正样本”，归为“同人未检出（漏检）”。
- 执行命令（建议）：

```bash
# 选取多个 query 观察 Top-10
python demo.py --query_index 42
python demo.py --query_index 777
python demo.py --query_index 1200
```

- 步骤：
  - [x] 基于 `demo.py` 或排名结果，收集 hard cases。
  - [x] 至少记录 2 类失败：
    - 同人误检（姿态/遮挡/跨相机）
    - 异人误匹配（衣着相似/背景干扰）
  - [x] 对每个失败案例记录 query ID、误检 gallery ID、原因分析。
- 记录区：
  - Case-1（同人误检，正样本极靠后）：
    - 使用结果文件：`./pytorch_result.mat`（最近一次 Duke 测试生成）
    - Query：`query_index=1883`，`label=4315`，`cam=6`
    - Query path：`./data/DukeMTMC-reID/pytorch/query/4315/4315_c6_f0076814.jpg`
    - Top-1 误检：`./data/DukeMTMC-reID/pytorch/gallery/6367/6367_c8_f0073570.jpg`（label=6367）
    - 首个正确匹配：`Rank=4274`，`./data/DukeMTMC-reID/pytorch/gallery/4315/4315_c7_f0080881.jpg`
    - Top-10 正样本数：`0/10`
    - 现象与可能原因：`Top-10 全为负样本且首个正样本非常靠后，属于典型 hard case，常见原因包括跨相机光照差异、遮挡/模糊、以及衣着/背景相似导致的误匹配。建议结合 ./report/pics/show_case1.png 做可视化说明。`
  - Case-2（异人误匹配，Top-10 全负样本）：
    - 使用结果文件：`./pytorch_result.mat`（最近一次 Duke 测试生成）
    - Query：`query_index=67`，`label=51`，`cam=1`
    - Query path：`./data/DukeMTMC-reID/pytorch/query/0051/0051_c1_f0060060.jpg`
    - Top-1 误检：`./data/DukeMTMC-reID/pytorch/gallery/6794/6794_c8_f0178831.jpg`（label=6794）
    - 首个正确匹配：`Rank=3277`，`./data/DukeMTMC-reID/pytorch/gallery/0051/0051_c2_f0060553.jpg`
    - Top-10 正样本数：`0/10`
    - 现象与可能原因：`Query 来自 cam1，正确匹配在 cam2，但 Top-10 被其它身份占据，说明跨相机外观变化下特征区分度不足。建议结合 ./report/pics/show_case2.png 做可视化说明，并尝试 re-ranking 或更强 backbone。`
- 验收标准：
  - [x] 至少 2 个具体失败案例（含图像 ID 或路径）。
  - [x] 每个案例有可解释原因。

- M7 总验收：
  - [ ] Duke 相关实验覆盖 README Part4 可选项。
  - [ ] 形成一份总对比表（见下方 M7.8）。

#### M7.8 Duke 实验总表（汇总区）

- [x] E0 Duke Baseline ResNet50（`duke_ft_ResNet50`）：Rank-1=0.793537，Rank-5=0.889587，Rank-10=0.920108，mAP=0.617418，备注=Duke baseline（`which_epoch=060`）
- [x] E1 跨域 Market->Duke（`ft_ResNet50 on Duke`）：Rank-1=0.329892，Rank-5=0.485189，Rank-10=0.548474，mAP=0.170039，备注=跨域泛化显著下降
- [x] E2 Backbone DenseNet（`--use_dense`）：Rank-1=0.815081，Rank-5=0.912926，Rank-10=0.936715，mAP=0.648376，备注=DenseNet121（`which_epoch=060`）
- [x] E3 Backbone HRNet（`--use_hr`）：Rank-1=0.838869，Rank-5=0.924147，Rank-10=0.942101，mAP=0.694562，备注=HRNet（`which_epoch=060`）
- [x] E4 Loss Circle（`--circle`）：Rank-1=0.782765，Rank-5=0.884201，Rank-10=0.912478，mAP=0.612843，备注=Circle loss（`which_epoch=060`）
- [x] E5 Loss Instance（`--instance`）：Rank-1=0.800718，Rank-5=0.889587，Rank-10=0.918312，mAP=0.623174，备注=Instance loss（`which_epoch=060`；训练耗时=116m 27s）
- [x] E6 Loss Triplet（`--triplet`）：Rank-1=0.793088，Rank-5=0.887792，Rank-10=0.921454，mAP=0.623227，备注=Triplet loss（`which_epoch=060`；训练耗时=122m 12s）
- [ ] E7 Verification+ID（external repo）：Rank-1= ，Rank-5= ，Rank-10= ，mAP= ，备注=

### M8 实验记录与报告提交

- 目标：满足 `requirement.md` 的报告要求并按时提交。

- 步骤（本文件内直接填写）：
  - [x] 基线与变体结果汇总（Market + Duke）已完成。
  - [x] 至少 3 个 Quick Questions 已回答。
  - [x] 至少 2 个失败案例已写入。
  - [x] 至少 2 条改进建议已给出。
  - [x] AI Safety 反思已完成。
  - [ ] 最终提交版（LaTeX，≥2页）已生成。

#### M8.1 Quick Questions（至少 3 题）

- Q1: Why use AdaptiveAvgPool2d? What is the difference between AvgPool2d and AdaptiveAvgPool2d?
  - 回答：`AdaptiveAvgPool2d((1,1))` can output a fixed spatial size regardless of input resolution, which makes the classifier head shape-stable. `AvgPool2d` uses a fixed kernel/stride, so the output size depends on the input size. In ReID, this helps accept different input shapes while still producing a global descriptor.
- Q2: Why `optimizer.zero_grad()`? What happens if we remove it?
  - 回答：PyTorch accumulates gradients by default. `optimizer.zero_grad()` clears old gradients; otherwise gradients from multiple iterations add up, effectively changing the update rule (larger/unstable steps) and usually harming convergence unless you intentionally do gradient accumulation.
- Q3: Why is output dimension `batchsize x 751`?
  - 回答：The classifier outputs logits over the number of training identities (classes). For Market-1501 the class count is 751, so the logits shape is `N x 751`. For DukeMTMC-reID it is 702, so you see `N x 702` during Duke training.
- Q4: Why flip image in test?
  - 回答：Horizontal flip is a simple test-time augmentation. Extracting features from original+flipped images and averaging them reduces sensitivity to left-right pose bias, often improving retrieval robustness with minimal cost.
- Q5: Why L2-normalize feature?
  - 回答：L2-normalization makes features comparable by direction (unit length), so cosine similarity and dot product become consistent. It stabilizes distance computation and is standard for retrieval embeddings.
- Q6: Can Market-trained model generalize to Duke?
  - 回答：Based on M7.2, performance drops sharply on Duke (Rank@1=0.329892, mAP=0.170039) compared with Market (Rank@1=0.877375, mAP=0.721856). This indicates strong domain shift; direct zero-shot generalization is weak.

#### M8.2 改进建议（至少 2 条）

- Proposal 1：Re-ranking（k-reciprocal / GNN re-ranking）
  - 动机：ReID 的检索排序对 hard negatives 敏感，后处理通常能显著提升 mAP。
  - 预计收益：mAP 常见提升明显，尤其在 Duke 等难数据上对长尾更友好。
  - 成本与风险：推理耗时与内存增加；参数（k1/k2/lambda）需调优。
- Proposal 2：更强的数据增强与采样策略（Random Erasing + Color Jitter + 身份均衡采样）
  - 动机：提升对遮挡/光照/背景变化的鲁棒性，减少过拟合。
  - 预计收益：通常提升泛化与稳定性，Rank@1 和 mAP 可能同步小幅提升。
  - 成本与风险：训练更慢；增强过强可能导致收敛变慢或欠拟合。

#### M8.3 AI Safety Reflection

- 潜在正向价值：用于公共安全与寻人（在合法授权前提下）、人员走失救援、智能安防告警辅助、零售/交通场景的人员轨迹分析（匿名化聚合）。
- 潜在风险/滥用场景：未经同意的跨摄像头跟踪、对特定群体的歧视性监控、与人脸/身份信息联动后形成更强的个人画像与隐私泄露风险。
- 技术防护建议：最小化数据留存（只保留必要 embedding 且加密）、访问控制与审计、对外提供检索时进行阈值与速率限制、尽量采用去标识化与分级权限。
- 产品与治理建议：明确使用边界与告知机制、合规审批流程（用途、场所、时长）、事后可追溯（日志）、对高风险部署进行第三方评估。
- 结论：ReID 具备明显的双重用途属性，应以“合法授权 + 最小必要 + 可审计”为基线，避免将研究代码直接用于现实监控场景。

#### M8.4 提交前检查清单

- 内容完整性：
  - [ ] 报告不少于 2 页（LaTeX 模板）
  - [ ] 包含 baseline 与变体结果
  - [x] 至少 3 个 Quick Questions 解答
  - [ ] 至少 2 个失败案例分析
  - [x] 至少 2 条改进建议
  - [x] AI Safety 反思
- 可复现性：
  - [x] 关键命令已列出
  - [x] 关键超参数已列出（详见 `./model/*/opts.yaml`）
  - [ ] 结果指标与对应实验一致
  - [ ] 权重与日志路径可追溯
- 提交事项：
  - [ ] 文件命名符合课程要求
  - [ ] 截止时间确认（Report-2: 3月8日）
  - [ ] Moodle 上传成功

- 交付物检查（对应 requirement.md）：
  - [ ] 基线 + 变体性能总结已包含。
  - [x] 至少 3 个 Quick Questions 已回答。
  - [x] 至少 2 个失败案例已分析。
  - [x] 至少 2 条改进建议已给出。
  - [x] AI 安全反思已完成。

- 截止提醒：
  - [ ] 按课程最新更新确认 DDL（`requirement.md` 更新写明 Report-2 延至 3 月 8 日）。

- 验收标准：
  - [ ] 他人可按记录复现实验。
  - [ ] 报告内容完整覆盖课程要求。
