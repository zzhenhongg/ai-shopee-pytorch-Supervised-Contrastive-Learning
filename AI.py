import os
import cv2
import copy
import time
import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn #除了大量的损失函数与激活函数，里面还包含了大量用于构建网络的函数
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.cuda import amp

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, f1_score

from tqdm.notebook import tqdm
from collections import defaultdict
import albumentations as A
from albumentations.pytorch import ToTensorV2

import timm
from pytorch_metric_learning import losses

#基础配置
np.set_printoptions(threshold=np.inf) # threshold 指定超过多少使用省略号，np.inf代表无限大
pd.set_option('display.max_columns', 1000000)   # 可以在大数据量下，没有省略号
pd.set_option('display.max_rows', 1000000)
pd.set_option('display.max_colwidth', 1000000)
pd.set_option('display.width', 1000000)

#训练配置
class CFG:
    seed = 42
    model_name = 'tf_efficientnet_b4_ns' #模型名称
    img_size = 64 #图像缩放，一个图片的维数
    scheduler = 'CosineAnnealingLR' #设置学习率
    T_max = 10 
    lr = 1e-5
    min_lr = 1e-6
    batch_size = 16 #一次迭代用的样本量
    weight_decay = 1e-6 #权重衰减
    num_epochs = 2 #表示过几次所有样本数据，1epoch表示过一次
    num_classes = 100 #分类个数
    embedding_size = 128 #embedding的本质是用一个较低维度的向量来代替较高维度的原始特征 这个是对应每个image的vector的dimension数，定义了输出张量
    n_fold = 5 #分为多少份
    n_accumulate = 4
    temperature = 0.1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #cuda或者cpu
#训练集和测试集
TRAIN_DIR = 'shopee-product-matching/train_images/'
TEST_DIR = 'shopee-product-matching/test_images/'

#设置可重复的种子
def set_seed(seed=42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    '''设置整个笔记本的种子，因此每次运行时结果都是相同的。这是出于可复制性。'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # When running on the CuDNN backend, two further options must be set 在CuDNN后端上运行时，必须设置另外两个选项
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Set a fixed value for the hash seed 为哈希种子设置一个固定值
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(CFG.seed)

#将文本与图像对应，形成一个df_train
#df_train = pd.read_csv('shopee-product-matching/train.csv')
df_train = pd.read_csv('folds.csv',nrows=512)
df_train['file_path'] = df_train.image.apply(lambda x: os.path.join(TRAIN_DIR, x))
#print(df_train.head(5)) #输出前5个数据
print(len(df_train))


le = LabelEncoder() #sklearn，用于训练数据，将label标准化，将原数据的label_group转成一个编号
df_train.label_group = le.fit_transform(df_train.label_group)
#print(df_train.head(5)) #输出前5个数据

#数据集类
class ShopeeDataset(Dataset):
    def __init__(self, root_dir, df, transforms=None):
        self.root_dir = root_dir
        self.df = df
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    #获得某一项
    def __getitem__(self, index):
        img_path = self.df.iloc[index, -1]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.df.iloc[index, -3]

        if self.transforms:
            img = self.transforms(image=img)["image"]

        return img, label

#增强与变换
data_transforms = {
    "train": A.Compose([
        A.Resize(CFG.img_size, CFG.img_size), #图像缩放
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast( #随机亮度压缩
            brightness_limit=(-0.1, 0.1),
            contrast_limit=(-0.1, 0.1),
            p=0.5
        ),
        A.Normalize( #正则化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.),
    "valid": A.Compose([
        A.Resize(CFG.img_size, CFG.img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)
}


#训练函数 使用自动混合精度来加快训练过程，并使用梯度累积来增加批次大小
#混合精度训练 mixed precision training
#梯度累积 gradient accumulation
def train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, fold):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict()) #最好的模型参数 state_dict变量存放训练过程中需要学习的权重和偏执系数，state_dict作为python的字典对象将每一层的参数映射成tensor张量
    best_loss = np.inf #最好的损失情况
    history = defaultdict(list)
    scaler = amp.GradScaler() #自动混合精度 该方法在训练网络时将单精度（FP32）与半精度(FP16)结合在一起，并使用相同的超参数实现了与FP32几乎相同的精度

    for step, epoch in enumerate(range(1, num_epochs + 1)):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase 每个时期都有训练和验证阶段
        for phase in ['train', 'valid']:
            if (phase == 'train'):
                model.train()  # Set model to training mode 训练模式 dropout层会按照设置好的失活概率进行失活，batchnorm会继续计算数据的均值和方差等参数并在每个batch size之间不断更新
            else:
                model.eval()  # Set model to evaluation mode eval主要是用来影响网络中的dropout层和batchnorm层的行为

            running_loss = 0.0

            # Iterate over data 遍历数据
            for inputs, labels in tqdm(dataloaders[phase],disable=True):#加了个disable=True tqdm进度条
                print(len(labels),"labels len is")
                if len(labels)<=1:
                    continue
                inputs = inputs.to(CFG.device)
                labels = labels.to(CFG.device)

                # forward 前向传播
                # track history if only in train 仅在训练中跟踪历史
                with torch.set_grad_enabled(phase == 'train'): #set_grad_enabled 会影响网络的自动求导机制
                    with amp.autocast(enabled=True):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        loss = loss / CFG.n_accumulate

                    # backward only if in training phase 仅在训练阶段时才反向传播
                    if phase == 'train':
                        scaler.scale(loss).backward()

                    # optimize only if in training phase 仅在训练阶段才进行优化
                    if phase == 'train' and (step + 1) % CFG.n_accumulate == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()

                        # zero the parameter gradients 参数梯度设为0
                        optimizer.zero_grad()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            history[phase + ' loss'].append(epoch_loss)

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'valid' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                PATH = f"Fold{fold}_{best_loss}_epoch_{epoch}.bin"
                torch.save(model.state_dict(), PATH)

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss ", best_loss)

    # load best model weights 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model, history

#按照每一份来运行调用模型
def run_fold(model, criterion, optimizer, scheduler, device, fold, num_epochs=10):
    valid_df = df_train[df_train.fold == fold]
    train_df = df_train[df_train.fold != fold]

    train_data = ShopeeDataset(TRAIN_DIR, train_df, transforms=data_transforms["train"])
    valid_data = ShopeeDataset(TRAIN_DIR, valid_df, transforms=data_transforms["valid"])

    dataset_sizes = {
        'train': len(train_data),
        'valid': len(valid_data)
    }

    #num_workers 不可以为多线程，改为0 dataloader就是一个迭代器，利用多线程加速处理
    #shuffle 每个epoch开始的时候，对数据进行重新排序
    #由于batchnorm层需要大于一个样本去计算其中的参数，解决方法是将dataloader的一个丢弃参数设置为true
    train_loader = DataLoader(dataset=train_data, batch_size=CFG.batch_size, num_workers=0, pin_memory=False,shuffle=True,drop_last=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=CFG.batch_size, num_workers=0, pin_memory=False,shuffle=False)

    dataloaders = {
        'train': train_loader,
        'valid': valid_loader
    }

    model, history = train_model(model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device,fold)

    return model, history

#加载模型
model = timm.create_model(CFG.model_name, pretrained=True) #这是一个与训练embedding model，下载了一个eff模型，
# 加载预训练神经网络模型，并将其中的分类器修改为自己的分类器，注意特征检测器和分类器不同，分类器需要我们重写
in_features = model.classifier.in_features #得到分类输入层，二维张量的大小
model.classifier = nn.Linear(in_features, CFG.embedding_size) #输入张量、输出张量，输出一个全连接层，得到一个embedding
out = model(torch.randn(1, 3, CFG.img_size, CFG.img_size)) #标准正态分布
print(f'Embedding shape: {out.shape}')
model.to(CFG.device) #代表将模型加载到指定设备上


#执行 使用pytorch 有监督的对比损失模型
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, feature_vectors, labels):
        # Normalize feature vectors
        feature_vectors_normalized = F.normalize(feature_vectors, p=2, dim=1)
        # Compute logits
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))
criterion = SupervisedContrastiveLoss(temperature=CFG.temperature).to(CFG.device) # Custom Implementation
# criterion = losses.SupConLoss(temperature=CFG.temperature).to(CFG.device)
optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr)


#运行fold 0 看起来是选取一个部分来执行
model, history = run_fold(model, criterion, optimizer, scheduler, device=CFG.device, fold=0, num_epochs=CFG.num_epochs)


#可视化训练和验证指标
plt.style.use('fivethirtyeight')
plt.rcParams["font.size"] = "20"
fig = plt.figure(figsize=(22,8))
epochs = list(range(CFG.num_epochs))
plt.plot(epochs, history['train loss'], label='train loss')
plt.plot(epochs, history['valid loss'], label='valid loss')
plt.ylabel('Loss', fontsize=20)
plt.xlabel('Epoch', fontsize=20)
plt.legend()
plt.title('Loss Curve');
plt.show()
print("do it")




