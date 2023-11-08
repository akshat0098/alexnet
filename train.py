from models import AlexNet
import torch
import os
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import torch.nn as nn

##Contants

GPUS = [0]
EPOCHS = 90
NO_CLASSES = 1000

TRAIN_DIR = "imagenet-mini/train"
VAL_DIR = "imagenet-mini/val"
IMG_DIM = 227
BATCH_SIZE = 128
L_RATE = 0.01
W_DECAY = 0.0005
MOMENTUM = 0.9
CHECKPOINT_DIR = 'checkpoints/'

device = 'cuda' if torch.cudo.is_available() else 'cpu'

seed = torch.initial_seed()

##create model
model = AlexNet(NO_CLASSES).to(device)

##train with multi GPU

model = torch.nn.parallel.DataParallel(model,device_ids=GPUS)
print(model)

#Image augmentationa and transformation

data_transform = transforms.Compose([
    transforms.CenterCrop(IMG_DIM),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalizer()
])

##prepare the dataset
train_dataset = datasets.ImageFolder(TRAIN_DIR,data_transform)
val_dataset = datasets.ImageFolder(VAL_DIR)

train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE
)

val_loader = DataLoader(
    val_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE
)

##optimizer
optim = torch.optim.SGD(
    model.parameters(),
    lr=L_RATE,
    momemtum = MOMENTUM,
    weight_decay=W_DECAY
)

##loss function
criterion = nn.CrossEntropyLoss()

##decay the learning rate
lr_scheduler = torch.optim.lr_scheduler.StepLR(optim,step_size=50,gamma=0.1)

total_steps = 1

##training 

for epoch in range(EPOCHS):
    for step ,(X,y) in enumerate(train_loader):
        X,y = X.to(device),y.to(device)
        ##refresh
        optim.zero_grad()
        ##forward pass
        pred = model(X)
        ##taking loss
        loss = criterion(pred,y).to(device)
        ##backward loss
        loss.backward()
        ##taking step
        optim.step()

        if total_steps % 10 ==0:
            print(f'step: {total_steps} | Loss: {loss} ')
        total_steps +=1

    ##saving checkponits

    checkpoint_path = os.path.join(CHECKPOINT_DIR,f'model_checkpoint{epoch+1}.pkl')

    state = {
        'epoch':epoch,
        'total_steps':total_steps,
        'optimizer': optim.state_dicts(),
        'model': model.state_dict(),
        'seed': seed
    }

    torch.save(state,checkpoint_path)