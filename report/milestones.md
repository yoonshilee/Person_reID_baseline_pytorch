# Person ReID 实验里程碑计划（可交互）

## 总览

- [x] M1 环境与依赖准备
- [x] M2 数据集准备（Market-1501）
- [x] M3 Baseline 训练（ResNet50）
- [ ] M4 特征提取与测试集前向
- [ ] M5 性能评估（CMC 与 mAP）
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
  - [x] `./Market-1501-v15.09.15/pytorch` 目录结构完整。
  - [x] 训练脚本可正确识别类别数与样本。

### M3 Baseline 训练（ResNet50）

- 目标：完成基线模型训练并保存权重。
- 步骤：
  - [x] 执行命令：`python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32 --data_dir ./Market-1501-v15.09.15/pytorch`
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
  - [ ] 执行：`python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir ./Market-1501-v15.09.15/pytorch --batchsize 32 --which_epoch 60`
  - [ ] 确认测试过程无路径或权重加载错误。
- 验收标准：
  - [ ] 测试脚本运行完成。
  - [ ] 成功产出评估所需特征结果文件。

- 执行命令（复制即用）：

```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir ./Market-1501-v15.09.15/pytorch --batchsize 32 --which_epoch 60
```

- 记录模板（执行后填写）：
  - [ ] 执行时间：`YYYY-MM-DD HH:mm`
  - [ ] 使用权重：`which_epoch=60`
  - [ ] 运行状态：`成功/失败`
  - [ ] 特征文件路径：`待填写`
  - [ ] 备注（报错或警告）：`待填写`

- 终端关键输出粘贴区：

```text
[在此粘贴 test.py 末尾关键输出]
```

### M5 性能评估（CMC 与 mAP）

- 目标：得到可汇报的检索性能指标。
- 步骤：
  - [ ] 执行：`python evaluate_gpu.py`
  - [ ] 记录 Rank-1、Rank-5、Rank-10 与 mAP 指标。
- 验收标准：
  - [ ] 成功输出 CMC/mAP。
  - [ ] 指标可与仓库 baseline 结果进行对比。

- 执行命令（复制即用）：

```bash
python evaluate_gpu.py
```

- 指标填写区（执行后填写）：

| 指标 | 本次结果 | Baseline/参考 | 结论 |
| --- | ---: | ---: | --- |
| Rank-1 | 待填写 | 待填写 | 待填写 |
| Rank-5 | 待填写 | 待填写 | 待填写 |
| Rank-10 | 待填写 | 待填写 | 待填写 |
| mAP | 待填写 | 待填写 | 待填写 |

- 运行记录：
  - [ ] 执行时间：`YYYY-MM-DD HH:mm`
  - [ ] 运行状态：`成功/失败`
  - [ ] 结果文件路径（如有）：`待填写`
  - [ ] 异常与修复：`待填写`

- 终端关键输出粘贴区：

```text
[在此粘贴 evaluate_gpu.py 输出]
```

### M6 检索可视化验证

- 目标：通过可视化确认模型检索效果。
- 步骤：
  - [ ] 执行：`python demo.py --query_index 777`（可替换不同 query）。
  - [ ] 观察 Top-K 返回结果中正负样本分布。
- 验收标准：
  - [ ] 可正常显示或打印 Top-10 检索结果。
  - [ ] 能解释至少 2 个成功匹配与 2 个失败匹配案例。

### M7 扩展实验（可选）

- 目标：在 baseline 上做变量对比实验。
- 步骤（建议三选一起步）：
  - [ ] 更换 backbone（如 DenseNet / Swin / ConvNeXt）。
  - [ ] 更换或组合损失（如 Triplet、Circle、Instance）。
  - [ ] 更换数据集（如 DukeMTMC-reID）并复现实验流程。
- 验收标准：
  - [ ] 每组实验至少有完整命令、配置、最终指标。
  - [ ] 与 baseline 对比形成结论（提升/退化及可能原因）。

### M8 实验记录与报告提交

- 目标：形成可复现实验文档。
- 步骤：
  - [ ] 统一整理训练命令、关键日志、权重版本与评估指标。
  - [ ] 在 `report` 目录保留实验记录文件。
  - [ ] 输出最终总结：方法、结果、问题与下一步。
- 验收标准：
  - [ ] 他人可按文档复现你的主要结果。
  - [ ] 报告包含“配置-过程-结果-分析”完整闭环。
