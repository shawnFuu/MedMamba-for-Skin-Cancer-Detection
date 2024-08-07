
##Overview

本工作主要应用了MedMamba[[YubiaoYue/MedMamba: This is the official code repository for "MedMamba: Vision Mamba for Medical Image Classification" (github.com)](https://github.com/YubiaoYue/MedMamba)] 提出的模型，基于VisionMamba解决医学图像分类问题，即皮肤癌检测。

本项目以kaggle竞赛ISIC 2024 - Skin Cancer Detection with 3D-TBP[[ISIC 2024 - Skin Cancer Detection with 3D-TBP | Kaggle](https://www.kaggle.com/competitions/isic-2024-challenge/overview)] 为背景，旨在提高皮肤癌检测的效率及正确率。我们通过该工作进一步验证了MedMamba在解决医学检测任务的有效性。

##Method
![1](https://github.com/user-attachments/assets/0ce35509-9f7d-49e6-944e-fe52d34420de)

我们基本遵循MedMamba提出的pipeline，该方法以SS-Conv-SSM为主要组成部分，融合了CNN和Mamba状态模型，保证在捕捉局部细节信息的同时获得全局关系，并使用VisionMamba提出的SS2D模块，通过多方向扫描使得Mamba能够合理地建模patch序列信息。

本工作做出的主要改动如下：

1. 将分类头`head`由单线形层`Linear`扩展为小型`MLP`，维度为 \[768, 384, 192, 96, 16, 2] ，以适应本工作为二分类任务的需求；
2. 为缓解原数据集严重的类别失衡问题，我们在原数据集的基础上，添加了过去三年比赛官方提供的训练集中的正例，并把新合成的数据集按照10:1划分出训练集和测试集；
3. 我们设计了一个全新的损失函数，MedMamba中简单使用交叉熵作为损失函数，在类别不平衡的数据集中不容易学习到数量过少的类别，因此我们设计了自适应的交叉熵损失函数，将正例和负例的损失系数作为神经网络的参数参与学习，从而根据学习情况灵活地调整系数；
4. 我们使用了以80%为基准的pAUC作为评价指标，从而更精准可靠地评估模型性能


##Result

在我们划分得到的测试集中，accuracy达到0.999，pAUC达到了0.122，达到了较好的效果。

##Limitation

尽管我们做出一些努力以缓解数据的类别失衡问题，但是其仍然没有得到合理地解决，借用GAN或者diffusion model进行data augmentation是可行的方向。

其次，对于比赛官方提供的数据集，我们仅使用了图片和其是否患病作为输入，对于其他丰富的信息均没有加入模型，这显然对模型性能有明显影响，比如患病类型、病人编号等等，这也可能是单纯的神经网络方法无法与基于lightLGBM的传统机器学习方法相抗衡的原因。

最后，基于MedMamba的方法对于该任务来说是否过于复杂也是值得探索的问题，在训练过程中我们发现了过拟合的现象，因此可以尝试直接使用VisionMamba观察是否能提高模型性能。

