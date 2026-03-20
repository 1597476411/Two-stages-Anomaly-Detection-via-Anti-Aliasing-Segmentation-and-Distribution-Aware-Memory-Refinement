import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob

from data_loader import Rescale, RescaleT, RandomCrop, ToTensor, ToTensorLab, SalObjDataset
from model.u2net import U2NET,U2NETP,U2NETP_LLPF

# ------- 1. define loss function --------
bce_loss = nn.BCELoss(reduction='mean')

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    """
    Calculate BCE loss for multi-scale outputs
    """
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
    loss0.data.item(), loss1.data.item(), loss2.data.item(), loss3.data.item(), loss4.data.item(), loss5.data.item(),
    loss6.data.item()))

    return loss0, loss

# ------- 2. set the directory of training dataset --------
model_name = 'u2netp_llpf'

# Training data paths
tra_image_dir = 'train_image'
tra_label_dir = 'train_mask'
image_ext = '.bmp'
label_ext = '.png'

# Load training file lists
tra_img_name_list = glob.glob(tra_image_dir + '*' + image_ext)
tra_lbl_name_list = glob.glob(tra_label_dir + '*' + label_ext)

# Use only first 10 samples for testing
tra_img_name_list = tra_img_name_list[:10]
tra_lbl_name_list = tra_lbl_name_list[:10]

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

# Model save directory
model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)
epoch_num = 1000
batch_size_train = 4
batch_size_val = 1
train_num = len(tra_img_name_list)
val_num = 0

# ====================== Core addition: Set Patience for Early Stopping ======================
patience = 10  # Stop training if loss doesn't decrease for consecutive epochs
min_delta = 1e-3  # Threshold to prevent jitter (loss decrease < this value = no improvement)
best_train_loss = float('inf')  # Record the best loss value
stop_counter = 0  # Counter for non-decreasing loss
early_stop_flag = False  # Early stop switch
# ===========================================================================================

# Modification 1: Set fixed model filename (overwrite save)
best_model_path = os.path.join(model_dir, f"{model_name}_best_model.pth")

# Build dataset and dataloader
salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)

# ------- 3. define model --------
if (model_name == 'u2net'):
    print("---loading U2NET (Big)...")
    net = U2NET(3, 1)
elif (model_name == 'u2netp'):
    print("---loading U2NETP (Small)...")
    net = U2NETP(3, 1)
elif (model_name == 'u2netp_llpf'):
    print("---loading U2NETP with Learnable Low-Pass Filter...")
    net = U2NETP_LLPF(3, 1) # Load LLPF version
else:
    raise ValueError("Unknown model_name: Choose from 'u2net', 'u2netp', 'u2netp_llpf'")

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0

# Outer epoch loop
for epoch in range(0, epoch_num):
    if early_stop_flag:
        print(f"\n======= Early stopping triggered! Training loss did not decrease for {patience} consecutive Epochs. Terminating training early. =======")
        break

    net.train()
    epoch_total_loss = 0.0

    # Inner batch loop
    for i, data in enumerate(salobj_dataloader):
        ite_num = ite_num + 1
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_v, labels_v = inputs.cuda(), labels.cuda()
        else:
            inputs_v, labels_v = inputs, labels

        optimizer.zero_grad()
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()
        running_tar_loss += loss2.data.item()
        epoch_total_loss += loss.data.item()

        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val,
            running_tar_loss / ite_num4val))

    # ====================== Core logic for saving best model ======================
    epoch_avg_loss = epoch_total_loss / len(salobj_dataloader)
    print(f"Epoch {epoch + 1} average training loss: {epoch_avg_loss:.6f} | Best historical loss: {best_train_loss:.6f}")

    # Check if current loss is better than best loss
    if epoch_avg_loss < best_train_loss - min_delta:
        best_train_loss = epoch_avg_loss
        stop_counter = 0
        # Create directory if not exists
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(net.state_dict(), best_model_path)
        print(f"Loss decreased effectively! Overwriting best model to: {best_model_path}\n")
    else:
        stop_counter += 1
        print(f"⚠️  Loss did not decrease. Early stop counter: {stop_counter}/{patience}\n")
        # Trigger early stop if counter reaches patience
        if stop_counter >= patience:
            early_stop_flag = True
    # ==============================================================================

print("\n======= Training completed (finished all Epochs / terminated early by early stopping) =======")
print(f"Final best training loss: {best_train_loss:.6f}, best model saved at: {best_model_path}")