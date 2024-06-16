from torchvision.models import resnet18
net = resnet18(num_classes=10).cuda()

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

import pytorch_lightning as pl

from tqdm.notebook import tqdm

from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from types import SimpleNamespace

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CHECKPOINT_PATH = "../model"
DATASET_PATH = "../data"

os.makedirs(CHECKPOINT_PATH, exist_ok=True)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

from torchvision.datasets import CIFAR10
from torchvision import transforms

test_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

test_set = CIFAR10(root=DATASET_PATH,train=False,transform=test_transform,download=True)

test_loader = data.DataLoader(test_set,batch_size=64,shuffle=False,num_workers=2)


model_dict = {}

def create_model(model_name):
  if model_name in model_dict:
    return model_dict[model_name]
  else:
    assert False, f"Unknown model \"{model_name}\""

class PLWrapper(pl.LightningModule):

  def __init__(self,model_name,optimizer_name,optimizer_hparams):

    super().__init__()

    self.save_hyperparameters()

    self.model = create_model(model_name)

    self.loss_module = nn.CrossEntropyLoss()

    self.example_input_array = torch.zeros((1,3,32,32),dtype=torch.float32)

  def forward(self,img):
    return self.model(img)


def load_model(model_name, **kwarg):
    model_folder = os.path.join(CHECKPOINT_PATH, model_name)
    pretrained_filename = os.path.join(model_folder, model_name + ".ckpt")
    print(f"Expected : {pretrained_filename}")
    if os.path.isfile(pretrained_filename):
        print("Found Trained model, loading ...")
        model = PLWrapper.load_from_checkpoint(pretrained_filename)
        print("Finished.")
    else:
        assert False,"Expected Model"

    return model.model

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
#    print(plt.get_backend())
    org = torch.clamp(org,0,1)
    # print(org.shape)
    noise = noise * 0.5 + 0.5
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

def fast_gradient_sign_method(model,imgs,labels,eps,**kwargs):
    inp_imgs = imgs.clone().requires_grad_()
    preds = model(inp_imgs.to(device))
    preds = F.log_softmax(preds, dim=-1)
    # Calculate loss by NLL
    loss = -torch.gather(preds, 1, labels.to(device).unsqueeze(dim=-1))
    loss.sum().backward()
    # Update image to adversarial example as written above
    noise_grad = torch.sign(inp_imgs.grad.to(device))
    fake_imgs = imgs + eps * noise_grad
    fake_imgs.detach_()
    return fake_imgs, noise_grad

def fast_gradient_sign_method2(model,imgs,labels,eps,**kwargs):
    inp_imgs = imgs.clone().requires_grad_()
    preds = model(inp_imgs.to(device))
    preds = F.log_softmax(preds, dim=-1)
    # Calculate loss by NLL
    fake_labels = torch.full(labels.shape,9)
    fake_labels = fake_labels - labels
    loss = -torch.gather(preds, 1, fake_labels.to(device).unsqueeze(dim=-1))
    loss.sum().backward()
    # Update image to adversarial example as written above
    noise_grad = torch.sign(inp_imgs.grad.to(imgs.device))
    fake_imgs = imgs - eps * noise_grad
    fake_imgs.detach_()
    return fake_imgs, noise_grad


model_dict["AA Task1-NO NORM"] = net
fn_dict = {
    "FGSM" : fast_gradient_sign_method,
    "FGSM2" : fast_gradient_sign_method2,
}

def clamp_inf(x,org,eps):
    x = torch.clamp(x,0,1)
    x = torch.clamp(x,org-eps,org+eps)
    return x

def clamp_2(x,org,eps):
   pass

def test_on_fakes(**kwargs):

    corr, total = 0,0
    batch_cnt = len(kwargs["data_loader"])
    for batch_idx,item in enumerate(kwargs["data_loader"]):
        imgs, labels = item
        # print(imgs.shape)
        # print(torch.min(imgs))
        # print(torch.max(imgs))
        kwargs["imgs"] = imgs
        kwargs["labels"] = labels
        fakes, noise = fn_dict[kwargs["fn_name"]](**kwargs)
        # print(noise[0])
        print(math.sqrt(torch.square(noise).sum().item()))
        with torch.no_grad():
            preds = kwargs["model"](fakes.to(device))
        labels = labels.to(device)
        corr += preds.argmax(dim=-1).eq(labels).sum().item()
        total += labels.shape[0]
        if (batch_idx==batch_cnt-1 or batch_idx%10==9):
            print(f"{batch_idx+1}/{batch_cnt}")
            show_image(imgs[0],fakes[0],noise[0],labels[0],preds.argmax(dim=-1)[0])

    return corr/total

resnet_model = load_model(model_name="AA Task1-NO NORM")
args = {
    "model" : resnet_model,
    "data_loader" : test_loader,
    "alpha" : 0.5/255,
    "clamp_fn" : clamp_2
}


for args["fn_name"] in ["FGSM"]:
    for args["eps"] in [0.02]:
        resnet_result = test_on_fakes(**args)

        print(f"ResNet Accuracy on eps = {args["eps"]} , Function = {args["fn_name"]}: {resnet_result:.3f}\n")