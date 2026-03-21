"""
Author: 雷尚谕
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from MLP import MLP
import time
import os
import json
from torch.cuda.amp import autocast, GradScaler
# import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tool import plot_loss
from tool import plot_loss_dual
from tool import signed_log10_torch


class MyData(Dataset):
    def __init__(self, inputs, outpus, dtype):
        self.Inputs = torch.tensor(inputs, dtype=dtype)
        self.Outputs = torch.tensor(outpus, dtype=dtype)

    def __getitem__(self, index):
        return self.Inputs[index], self.Outputs[index]

    def __len__(self):
        return len(self.Inputs)


def loss_fn(preds, labels):

    # preds_ij = preds[::2]  # 从索引 0 开始，每隔两个取一个，即 0, 2, 4, ...
    # preds_ji = preds[1::2] # 从索引 1 开始，每隔两个取一个，即 1, 3, 5, ...

    # lambda_sym=0.05
    mse_loss = nn.MSELoss()
    loss_pre = mse_loss(preds, labels)
    # loss_sym=mse_loss(preds_ij,preds_ji)

    return loss_pre


def loss_fn_rel(preds, labels):
    print("use rel loss")
    preds_ij = preds[::2]  # 0, 2, 4, ...
    preds_ji = preds[1::2]  # 1, 3, 5, ...

    lambda_sym = 0.05
    eps = 1e-10

    # 相对误差：|pred - label| / (|label| + eps)
    rel_error = torch.abs(preds - labels) / (torch.abs(labels) + eps)
    loss_pre = torch.mean(rel_error)

    rel_sym_error = torch.abs(preds_ij - preds_ji) / (torch.abs(preds_ij) + eps)
    loss_sym = torch.mean(rel_sym_error)

    return loss_pre + lambda_sym * loss_sym


def loss_hybrid(preds, labels):
    mse_loss = nn.MSELoss()
    loss_pre_mse = mse_loss(preds, labels)

    preds_log = signed_log10_torch(preds)
    labels_log = signed_log10_torch(labels)

    loss_pre_log = mse_loss(preds_log, labels_log)
    # lambda_sym = 0.1
    alpha = 0.1
    print("log", loss_pre_log.item(), "mse", loss_pre_mse.item())
    return alpha * loss_pre_mse + (1 - alpha) * loss_pre_log


def loss_hybrid_rel(preds, labels):
    mse_loss = nn.MSELoss()
    loss_pre_mse = mse_loss(preds, labels)

    preds_log = signed_log10_torch(preds) * 1e2
    labels_log = signed_log10_torch(labels) * 1e2

    rel_error = torch.abs(preds_log - labels_log) / (torch.abs(labels_log) + 1e-6)
    loss_pre_log = torch.sum(rel_error)

    # lambda_sym = 0.1
    alpha = 0.3
    print("log", loss_pre_log.item(), "mse", loss_pre_mse.item())
    return alpha * loss_pre_mse + (1 - alpha) * loss_pre_log


def train(use_dropout, dropout_p,
          use_L2, L2weight,
          hidden_dim, block_num,
          use_cosine, lr_min, max_lr, epoch_num,
          batchsize, numworks,
          folder_name, outputdim, Stype, basePath
):
    torch.cuda.manual_seed(1)
    torch.mps.manual_seed(1)
    torch.manual_seed(1)

    # device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    dtype = torch.float16

    # basePath = f"dataset/{name_psd}"
    # labelPath = f"dataset/{name_psd}/{Stype}"
    train_inputs_scaled = np.load(f"{basePath}/train/train_inputs_scaled.npy")
    train_labels_scaled = np.load(f"{basePath}/{Stype}/train/train_labels_scaled{outputdim}.npy")
    test_inputs_scaled = np.load(f"{basePath}/test/test_inputs_scaled.npy")
    test_labels_scaled = np.load(f"{basePath}/{Stype}/test/test_labels_scaled{outputdim}.npy")
    print("train_inputs_scaled", train_inputs_scaled.shape)
    print("train_labels_scaled", train_labels_scaled.shape)

    dataset = MyData(train_inputs_scaled, train_labels_scaled, dtype=dtype)
    dataloader = DataLoader(
        dataset=dataset, batch_size=batchsize, shuffle=True, num_workers=numworks
    )
    test_set = MyData(test_inputs_scaled, test_labels_scaled, dtype=dtype)
    test_loader = DataLoader(
        dataset=test_set, batch_size=batchsize, shuffle=False, num_workers=numworks
    )

    model = MLP(
        use_dropout=use_dropout,
        dropout_p=dropout_p,
        hidden_dim=hidden_dim,
        block_num=block_num,
    ).to(device=device, dtype=torch.float32)

    # scaler = GradScaler(enabled=False)

    loss_list = []
    loss_list_batch = []
    test_loss = []
    test_loss_batch = []

    patience = 500  # 容忍epoch数
    no_improve = 0
    min_delta = 1e-6

    base_lr = max_lr/5
    warmup_epochs = 5

    kernel_params = [
        model.w1,
        model.sigma1,
        model.shift1_x,
        model.shift1_y,
        model.shift1_z,
    ]
    other_params = [
        p
        for name, p in model.named_parameters()
        if name not in {"w1", "sigma1", "shift1_x", "shift1_y", "shift1_z"}
    ]

    if use_L2:
        optimizer = optim.AdamW(
            [
                {"params": other_params, "lr": max_lr, "weight_decay": L2weight},
                {"params": kernel_params, "lr": max_lr},
            ]
        )
        # optimizer=optim.AdamW(model.parameters(),lr=max_lr,weight_decay=L2weight)
    else:
        optimizer = optim.AdamW(
            [
                {"params": other_params, "lr": max_lr},
                {"params": kernel_params, "lr": max_lr},
            ]
        )
        # optimizer=optim.AdamW(model.parameters(),lr=max_lr)
    # scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=lr_min)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=lr_min)
    if use_cosine:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=epoch_num, T_mult=2, eta_min=lr_min)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epoch_num, eta_min=lr_min)
        
    start = time.time()
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        epoch_start = time.time()

        if use_cosine:
            if epoch < warmup_epochs + 1:
                for i, param_group in enumerate(optimizer.param_groups):
                    group_max_lr = max_lr if i == 0 else max_lr
                    lr = base_lr + (group_max_lr - base_lr) * epoch / warmup_epochs
                    param_group["lr"] = lr
                # lr = base_lr + (max_lr - base_lr) * epoch / warmup_epochs
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr
            else:
                scheduler.step()
        else:
                scheduler.step()

        for inputs, labels in dataloader:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            optimizer.zero_grad()

            with torch.amp.autocast(device.type, dtype=dtype):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

            # outputs = model(inputs)
            # loss = loss_fn(outputs, labels=labels)
            # loss=loss_hybrid(outputs,labels=labels)
            # loss = loss_hybrid_rel(outputs, labels)

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            total_loss += loss.item()

            # print(f"Epoch {epoch}: shift1_x end = {model.shift1_x.item()}")

        for i, param_group in enumerate(optimizer.param_groups):
            print(f"Epoch {epoch}, group {i}, lr: {param_group['lr']}")

        epoch_end = time.time()
        print(f"epoch{epoch} 用时{epoch_end-epoch_start}")

        total_loss = total_loss / len(dataloader)
        print(total_loss)
        if len(loss_list_batch) > 0:
            improve = abs(total_loss - loss_list_batch[-1]) / (
                loss_list_batch[-1] + 1e-10
            )
            if improve < min_delta:
                no_improve += 1
        loss_list_batch.append(total_loss)

        #model.eval()
        # Loss_fn=nn.MSELoss()
        #with torch.no_grad():
            #avg_loss = 0
            #for inputs, labels in test_loader:
               # inputs = inputs.to(device, dtype=torch.float32)
                #labels = labels.to(device, dtype=torch.float32)

                #with torch.amp.autocast(device.type, dtype=dtype):
                    #outputs = model(inputs)
                    #loss = loss_fn(outputs, labels)

                ## outputs = model(inputs)
                ## loss = loss_fn(outputs, labels=labels)
                ## loss=loss_hybrid(outputs,labels=labels)
                ## loss = loss_hybrid_rel(outputs, labels)

                #test_loss.append(loss.item())
                #avg_loss += loss.item()
            #avg_loss = avg_loss / len(test_loader)
            #test_loss_batch.append(avg_loss)
            #print("test_loss_batch", avg_loss)

        if (epoch + 2) % 50 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(folder_name, f"mlp_model_epoch{epoch+1}.pth"),
            )
            print(f"epoch:{epoch} loss:{total_loss}")

        if no_improve > patience:
            print(f"early stop at{epoch+1}")
            torch.save(
                model.state_dict(), os.path.join(folder_name, f"mlp_model_best.pth")
            )
            print(f"epoch:{epoch} loss:{total_loss}")
            break

        if (epoch + 1) == epoch_num:
            print(f"final stop at{epoch+1}")
            torch.save(
                model.state_dict(), os.path.join(folder_name, f"mlp_model_best.pth")
            )
            print(f"epoch:{epoch} loss:{total_loss}")
            break

    end = time.time()
    # torch.save(model.state_dict(), "mlp_model.pth")
    print(f"训练总耗时{end-start}")

    plot_loss(loss_list=loss_list, plot_name="loss_list", folder_name=folder_name)
    plot_loss(loss_list=test_loss, plot_name="test_loss", folder_name=folder_name)
    plot_loss_dual(
        loss_list1=loss_list_batch,
        loss_list2=test_loss_batch,
        plot_name="train+test batch loss",
        folder_name=folder_name,
    )

    param_json = os.path.join(folder_name, "params.json")
    params = {
        "use_dropout": use_dropout,
        "dropout_p": dropout_p,
        "use_L2": use_L2,
        "L2weight": L2weight,
        "learning_rate": max_lr,
        "batchsize": batchsize,
        "numworks": numworks,
        "hidden_dim": hidden_dim,
        "block_num": block_num,
        "time": end-start
    }

    with open(param_json, "w") as f:
        json.dump(params, f, indent=4)
