# Person ReID 实验里程碑计划（可交互）

## 总览

- [x] M1 环境与依赖准备
- [x] M2 数据集准备（Market-1501）
- [x] M3 Baseline 训练（ResNet50）
- [x] M4 特征提取与测试集前向
- [x] M5 性能评估（CMC 与 mAP）
- [ ] M6 检索可视化验证
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
  - [ ] 观察 Top-K 返回结果中正负样本分布。
- 验收标准：
  - [ ] 可正常显示或打印 Top-10 检索结果。
  - [ ] 能解释至少 2 个成功匹配与 2 个失败匹配案例。

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
  - [ ] 用当前 `ft_ResNet50` 权重在 Duke 上提特征。
  - [ ] 在 Duke 上评估 CMC/mAP。
  - [ ] 与 Market 上结果对比，分析域偏移影响。
- 记录区：
  - Duke Rank@1：
  - Duke mAP：
  - 与 Market 差异结论：
- 验收标准：
  - [ ] 得到 Duke 上的 Rank@1 与 mAP。
  - [ ] 给出“是否可直接泛化”的结论和证据。

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
  - [ ] 在 Duke 上训练 `ft_ResNet50` baseline。
  - [ ] 在 Duke 上完成 test + evaluate。
  - [ ] 保存权重、曲线、指标与关键日志。
- 记录区：
  - Rank@1：
  - Rank@5：
  - Rank@10：
  - mAP：
  - 权重目录：`./model/duke_ft_ResNet50/`
- 验收标准：
  - [ ] 得到 Duke baseline 的 Rank@1/5/10 和 mAP。
  - [ ] 有可复现实验命令与权重路径。

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
  - [ ] DenseNet 方案：训练、测试、评估。
  - [ ] HRNet 方案：训练、测试、评估。
  - [ ] 与 Duke baseline 对比性能与速度。
- 记录区：
  - Dense Rank@1/mAP：
  - HR Rank@1/mAP：
  - 开销对比（训练时长/显存）：
- 验收标准：
  - [ ] 两个 backbone 均有完整指标。
  - [ ] 有“性能-开销”对比结论。

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
  - [ ] Circle Loss 实验。
  - [ ] Instance Loss 实验。
  - [ ] Triplet Loss 实验（本仓库或 README 提供的 triplet repo）。
  - [ ] 与 Duke baseline 做公平对比（尽量保持同 batchsize/lr/epoch）。
- 记录区：
  - Circle Rank@1/mAP：
  - Instance Rank@1/mAP：
  - Triplet Rank@1/mAP：
  - 最优 loss：
- 验收标准：
  - [ ] 三种 loss 至少完成 2 种（目标全做）。
  - [ ] 明确指出哪种 loss 对 Duke 提升最明显。

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
- 执行命令（建议）：

```bash
# 选取多个 query 观察 Top-10
python demo.py --query_index 42
python demo.py --query_index 777
python demo.py --query_index 1200
```

- 步骤：
  - [ ] 基于 `demo.py` 或排名结果，收集 hard cases。
  - [ ] 至少记录 2 类失败：
    - 同人误检（姿态/遮挡/跨相机）
    - 异人误匹配（衣着相似/背景干扰）
  - [ ] 对每个失败案例记录 query ID、误检 gallery ID、原因分析。
- 记录区：
  - Case-1（同人误检）：
  - Case-2（异人误匹配）：
- 验收标准：
  - [ ] 至少 2 个具体失败案例（含图像 ID 或路径）。
  - [ ] 每个案例有可解释原因。

- M7 总验收：
  - [ ] Duke 相关实验覆盖 README Part4 可选项。
  - [ ] 形成一份总对比表（见下方 M7.8）。

#### M7.8 Duke 实验总表（汇总区）

- [ ] E0 Duke Baseline ResNet50（`duke_ft_ResNet50`）：Rank-1= ，Rank-5= ，Rank-10= ，mAP= ，备注=
- [ ] E1 跨域 Market->Duke（`ft_ResNet50 on Duke`）：Rank-1= ，Rank-5= ，Rank-10= ，mAP= ，备注=
- [ ] E2 Backbone DenseNet（`--use_dense`）：Rank-1= ，Rank-5= ，Rank-10= ，mAP= ，备注=
- [ ] E3 Backbone HRNet（`--use_hr`）：Rank-1= ，Rank-5= ，Rank-10= ，mAP= ，备注=
- [ ] E4 Loss Circle（`--circle`）：Rank-1= ，Rank-5= ，Rank-10= ，mAP= ，备注=
- [ ] E5 Loss Instance（`--instance`）：Rank-1= ，Rank-5= ，Rank-10= ，mAP= ，备注=
- [ ] E6 Loss Triplet（`--triplet`）：Rank-1= ，Rank-5= ，Rank-10= ，mAP= ，备注=
- [ ] E7 Verification+ID（external repo）：Rank-1= ，Rank-5= ，Rank-10= ，mAP= ，备注=

### M8 实验记录与报告提交

- 目标：满足 `requirement.md` 的报告要求并按时提交。

- 步骤（本文件内直接填写）：
  - [ ] 基线与变体结果汇总（Market + Duke）已完成。
  - [ ] 至少 3 个 Quick Questions 已回答。
  - [ ] 至少 2 个失败案例已写入。
  - [ ] 至少 2 条改进建议已给出。
  - [ ] AI Safety 反思已完成。
  - [ ] 最终提交版（LaTeX，≥2页）已生成。

#### M8.1 Quick Questions（至少 3 题）

- Q1: Why use AdaptiveAvgPool2d? What is the difference between AvgPool2d and AdaptiveAvgPool2d?
  - 回答：
- Q2: Why `optimizer.zero_grad()`? What happens if we remove it?
  - 回答：
- Q3: Why is output dimension `batchsize x 751`?
  - 回答：
- Q4: Why flip image in test?
  - 回答：
- Q5: Why L2-normalize feature?
  - 回答：
- Q6: Can Market-trained model generalize to Duke?
  - 回答：

#### M8.2 改进建议（至少 2 条）

- Proposal 1：
  - 动机：
  - 预计收益：
  - 成本与风险：
- Proposal 2：
  - 动机：
  - 预计收益：
  - 成本与风险：

#### M8.3 AI Safety Reflection

- 潜在正向价值：
- 潜在风险/滥用场景：
- 技术防护建议：
- 产品与治理建议：
- 结论：

#### M8.4 提交前检查清单

- 内容完整性：
  - [ ] 报告不少于 2 页（LaTeX 模板）
  - [ ] 包含 baseline 与变体结果
  - [ ] 至少 3 个 Quick Questions 解答
  - [ ] 至少 2 个失败案例分析
  - [ ] 至少 2 条改进建议
  - [ ] AI Safety 反思
- 可复现性：
  - [ ] 关键命令已列出
  - [ ] 关键超参数已列出
  - [ ] 结果指标与对应实验一致
  - [ ] 权重与日志路径可追溯
- 提交事项：
  - [ ] 文件命名符合课程要求
  - [ ] 截止时间确认（Report-2: 3月8日）
  - [ ] Moodle 上传成功

- 交付物检查（对应 requirement.md）：
  - [ ] 基线 + 变体性能总结已包含。
  - [ ] 至少 3 个 Quick Questions 已回答。
  - [ ] 至少 2 个失败案例已分析。
  - [ ] 至少 2 条改进建议已给出。
  - [ ] AI 安全反思已完成。

- 截止提醒：
  - [ ] 按课程最新更新确认 DDL（`requirement.md` 更新写明 Report-2 延至 3 月 8 日）。

- 验收标准：
  - [ ] 他人可按记录复现实验。
  - [ ] 报告内容完整覆盖课程要求。
