import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
matplotlib.use('Agg')

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
torch.set_printoptions(sci_mode=False)

from tqdm.notebook import tqdm

from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from types import SimpleNamespace

import LoadModel

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATASET_PATH = "../data"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from torchvision.datasets import CIFAR10
from torchvision import transforms

test_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

test_set = CIFAR10(root=DATASET_PATH,train=False,transform=test_transform,download=True)

test_loader = data.DataLoader(test_set,batch_size=64,shuffle=False,num_workers=2)

label_name = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

def show_image(org,fake,noise,label,fake_label):
    org = torch.clamp(org,0,1)
    a, b = noise.min(), noise.max()
    noise = (noise - a) / (b-a)
    noise = torch.clamp(noise,0,1)
    org = org.cpu().permute(1, 2, 0).numpy()
    noise = noise.cpu().permute(1, 2, 0).numpy()
    fake = fake.cpu().permute(1, 2, 0).numpy()
    _ , ax = plt.subplots(1,3)
    ax[0].imshow(org)
    ax[0].set_title(label_name[label])
    ax[1].imshow(noise)
    ax[1].set_title("Noise")
    ax[2].imshow(fake)
    ax[2].set_title(label_name[fake_label])
    plt.savefig("img.png")
    plt.close()

def PGDL2(model,imgs,labels,eps,alpha,steps,random_start=True,**kwargs):
    imgs = imgs.to(device)
    fake_imgs = imgs.clone()

    if random_start:
        fake_imgs = fake_imgs + torch.empty_like(fake_imgs).uniform_(
            -eps,eps
        )
        fake_imgs = torch.clamp(fake_imgs, min=0, max=1).detach()

    for stp in range(steps):

        # pass through model
        inp_imgs = fake_imgs.clone().requires_grad_()
        
        preds = model(inp_imgs.to(device))

        # Calculate loss by NLL
        preds = F.log_softmax(preds, dim=-1)
        loss = -torch.gather(preds, 1, labels.to(device).unsqueeze(dim=-1))
        loss.sum().backward()

        # scale grad L2 norm to 1
        grads = inp_imgs.grad.to(device)
        grad_norms = 1e-8 + torch.linalg.vector_norm(grads.view(grads.shape[0],-1),ord=2,dim=1)
        grads = grads / grad_norms.view(-1,1,1,1)

        # add alpha * grad
        fake_imgs = fake_imgs + alpha * grads

        # scale noise into L2 norm ball
        noise = fake_imgs - imgs
        noise_norms = torch.linalg.vector_norm(noise.view(noise.shape[0],-1),ord=2,dim=1)
        factor = eps / noise_norms
        factor = torch.min(factor, torch.ones_like(noise_norms))
        noise = noise * factor.view(-1,1,1,1)
        fake_imgs = torch.clamp(imgs + noise,min=0,max=1)

    
    return fake_imgs.detach_(), fake_imgs - imgs


def test_on_fakes(**kwargs):

    corr, total = 0,0
    batch_cnt = len(kwargs["data_loader"])
    for batch_idx,item in enumerate(kwargs["data_loader"]):

        imgs, labels = item
        kwargs["imgs"] = imgs
        kwargs["labels"] = labels

        fakes, noise = PGDL2(**kwargs)

        with torch.no_grad():
            preds = kwargs["model"](fakes.to(device))
        
        labels = labels.to(device)
        corr += preds.argmax(dim=-1).eq(labels).sum().item()
        total += labels.shape[0]
        if batch_idx==batch_cnt-1 or batch_idx%20==19:
            print(f"{batch_idx+1}/{batch_cnt}")
            show_image(imgs[0],fakes[0],noise[0],labels[0],preds.argmax(dim=-1)[0])
        if batch_idx==batch_cnt-1:
            print(torch.linalg.vector_norm(noise.view(noise.shape[0],-1),ord=2,dim=1))

    return corr/total


resnet_model = LoadModel.load_model(model_name="AA Task1-NO NORM")
args = {
    "model" : resnet_model,
    "data_loader" : test_loader,
}

for args["eps"] in [0.25,1,1.5]:
    for args["alpha"] in [32/255]:
        for args["steps"] in [25]:
            resnet_result = test_on_fakes(**args)

            print(f"ResNet Accuracy on eps = {args["eps"]}, alpha = {args["alpha"]}, steps = {args["steps"]} : {resnet_result:.3f}\n")