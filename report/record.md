# Training Record

## 2026-02-28 - Market1501 - ft_ResNet50

### Command

```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32 --data_dir ./Market-1501-v15.09.15/pytorch
```

### Config (from ./model/ft_ResNet50/opts.yaml)

- Dataset: `./Market-1501-v15.09.15/pytorch`
- Train split: `train_all=true`
- GPU: `gpu_ids=[0]`
- Batch size: `32`
- Epochs: `60`
- LR: `0.05`
- Weight decay: `5e-4`
- Dropout: `0.5`
- Stride: `2`
- Feature dim: `512`
- Num classes: `751`
- Mixed precision: `fp16=false`, `bf16=false`
- Backbone flags: `use_dense=false`, `use_swin=false`, `use_swinv2=false`, `use_convnext=false`, `use_hr=false`, `use_efficient=false`, `use_NAS=false`
- Loss flags: `arcface=false`, `circle=false`, `cosface=false`, `contrast=false`, `instance=false`, `triplet=false`, `lifted=false`, `sphere=false`

### Artifacts

- Output dir: `./model/ft_ResNet50/`
- Checkpoints: `net_010.pth`, `net_020.pth`, `net_030.pth`, `net_040.pth`, `net_050.pth`, `net_060.pth`, `net_last.pth`
- Snapshot files: `opts.yaml`, `train.jpg`, `train.py`, `model.py`

### Notes

- This record is organized from full terminal stdout provided by user.
- torchvision showed deprecation warnings for `pretrained`; behavior mapped to `weights=ResNet50_Weights.IMAGENET1K_V1`.
- Pretrained backbone weight was downloaded: `resnet50-0676ba61.pth` (97.8MB).
- A duplicated `Epoch 58/59` line appeared in stdout; metrics remained consistent with normal progression.

### Training Progress (key points)

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 3.8583 | 0.2818 | 1.8749 | 0.5246 | 0m 40s |
| 1 | 1.2178 | 0.6873 | 0.8082 | 0.7550 | 1m 10s |
| 2 | 0.6512 | 0.8214 | 0.4951 | 0.8535 | 1m 41s |
| 5 | 0.2223 | 0.9406 | 0.1612 | 0.9374 | 3m 12s |
| 7 | 0.1098 | 0.9775 | 0.0362 | 0.9734 | 4m 15s |
| 10 | 0.0499 | 0.9948 | 0.0235 | 0.9787 | 5m 51s |
| 12 | 0.0372 | 0.9975 | 0.0124 | 0.9800 | 6m 57s |
| 30 | 0.0276 | 0.9992 | 0.0082 | 0.9800 | 17m 0s |
| 41 | 0.0181 | 0.9994 | 0.0053 | 0.9800 | 23m 8s |
| 59 | 0.0209 | 0.9994 | 0.0062 | 0.9800 | 33m 9s |

### Final Summary

- Run completed all `60` epochs (`0` to `59`).
- Final metrics (Epoch 59): `train_loss=0.0209`, `train_acc=0.9994`, `val_loss=0.0062`, `val_acc=0.9800`.
- Best validation accuracy observed: `0.9800` (first reached at Epoch 12, maintained afterwards).
- Best validation loss observed in shown log: `0.0053` (Epoch 41).

