from torchvision.models import resnet18
net = resnet18(num_classes=10).cuda()

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

import pytorch_lightning as pl

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

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(32,scale=(0.8,1.0),ratio=(0.9,1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_dataset = CIFAR10(root=DATASET_PATH,train=True,transform=train_transform,download=True)
val_dataset = CIFAR10(root=DATASET_PATH,train=True,transform=test_transform,download=True)
pl.seed_everything(177)
train_set, _ = torch.utils.data.random_split(train_dataset,[45000,5000])
pl.seed_everything(177)
_, val_set = torch.utils.data.random_split(val_dataset,[45000,5000])

test_set = CIFAR10(root=DATASET_PATH,train=False,transform=test_transform,download=True)

train_loader = data.DataLoader(train_set,batch_size=64,shuffle=True,drop_last=True,pin_memory=True,num_workers=2)
val_loader = data.DataLoader(val_set,batch_size=64,shuffle=False,num_workers=2)
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

  def configure_optimizers(self):
    if self.hparams.optimizer_name == "Adam":
      optimizer = optim.AdamW(self.parameters(),**self.hparams.optimizer_hparams)
    elif self.hparams.optimizer_name == "SGD":
      optimizer = optim.SGD(self.parameters(),**self.hparams.optimizer_hparams)
    else:
      assert False, f"Unknown optimizer \"{self.hparams.optimizer_name}\""

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    return [optimizer], [scheduler]

  def training_step(self,batch,batch_idx):
    imgs, labels = batch
    preds = self.model(imgs)
    loss = self.loss_module(preds,labels)
    acc = preds.argmax(dim=-1).eq(labels).float().mean()

    self.log("train_acc",acc,on_step=True,on_epoch=True)
    self.log("train_loss",loss)
    return loss

  def validation_step(self,batch,batch_idx):
    imgs, labels = batch
    preds = self.model(imgs)
    acc = preds.argmax(dim=-1).eq(labels).float().mean()

    self.log("val_acc",acc)

  def test_step(self,batch,batch_idx):
    imgs, labels = batch
    preds = self.model(imgs)
    acc = preds.argmax(dim=-1).eq(labels).float().mean()

    self.log("test_acc",acc)

def train_model(model_name, **kwarg):
  model_folder = os.path.join(CHECKPOINT_PATH,model_name)
  trainer = pl.Trainer(default_root_dir=model_folder,
              accelerator="gpu" if str(device)=="cuda" else "cpu",
              devices=1,
              max_epochs=200,
              callbacks=[ModelCheckpoint(save_weights_only=True,mode="max",monitor="val_acc",
                                        dirpath=model_folder,filename=model_name),
                    LearningRateMonitor("epoch")],
              enable_progress_bar=True)

  trainer.logger._log_graph = True
  trainer.logger._default_hp_metric = True

  pretrained_filename = os.path.join(model_folder,model_name+".ckpt")
  print(f"Expected : {pretrained_filename}")
  if os.path.isfile(pretrained_filename):
    print("Found Trained model, loading ...")
    model = PLWrapper.load_from_checkpoint(pretrained_filename)
    print("Finished.")
  else:
    pl.seed_everything(177)
    model = PLWrapper(model_name=model_name,**kwarg)
    trainer.fit(model,train_loader,val_loader)
    model = PLWrapper.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

  test_result = trainer.test(model,test_loader,verbose=False)

  return model,test_result

model_dict["AA Task1-2"] = net

resnet_model, resnet_results = train_model(
    model_name="AA Task1-2",
    optimizer_name = "SGD",
    optimizer_hparams = {
        "lr":0.01,
        "momentum":0.9,
        "weight_decay":5e-4
    }
)
print(f"ResNet Test Accuracy: {resnet_results[0]['test_acc']:.3f}")
