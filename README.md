1、创建环境

```
conda create -n pytorch_env python=3.8
```

2、安装依赖

```
conda install --file requirements.txt
```

3、训练

```
python train.py
```

4、目录结构

```
├── output                   输出内容
├── model                    模型
│   ├── train_seg2_512.pkl
│   └── val_seg2_512.pkl
├── inference.py            For generating music. (Detailed usage are written in the file)
├── logger.py               日志
├── mymodel.py              The overal Theme Transformer Architecture
```

