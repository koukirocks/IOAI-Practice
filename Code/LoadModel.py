from torchvision.models import resnet18
net = resnet18(num_classes=10).cuda()

import os
import torch
import torch.nn as nn
import pytorch_lightning as pl

CHECKPOINT_PATH = "../model"

model_dict = {}
model_dict["AA Task1-NO NORM"] = net

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