# Handwriting_Detector

基于卷积神经网络（CNN），使用pytorch自带数据库MNIST的手写数字识别，包含模型训练、模型加载和手写面板UI，可识别画板上用鼠标等设备书写的数字，并展示处理后的图像、识别结果与置信度

# 依赖清单

- torch
- torchvision
- numpy
- matplotlib
- pillow
```bash
pip install torch torchvision numpy matplotlib pillow
```
Linux下可能需要装Tkinter
```bash
sudo apt install python3-tk
```

# 使用方法

运行程序：

```bash
python main.py
```

# 加载模型

加载路径默认在main.py同级下的model.pth

# 模型结构

CNN 网络结构：

Conv1 卷积层1 
- Conv2d(1 → 10, kernel=5)  
- ReLU  
- MaxPool2d  

Conv2 卷积层2
- Conv2d(10 → 20, kernel=5)  
- ReLU  
- MaxPool2d  

Fully Connected 全连接层
- Linear(320 → 50)  
- ReLU  
- Linear(50 → 10)

# 项目结构

HandWriting_Detect
│
├─ main.py           # 主程序
├─ model.pth         # 训练好的模型（可选）
├─ README.md
└─ demo.png          # 演示图片