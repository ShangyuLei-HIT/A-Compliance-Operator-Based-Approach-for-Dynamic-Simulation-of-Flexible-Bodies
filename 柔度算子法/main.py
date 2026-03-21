"""
Author: 雷尚谕
"""
import os
from sympy import false
import train_model
from evaluate_model import evaluate
import re
from generate import gen
from export_pt import export_onnx
import argparse

def train(with_train, output_dim, Stype, basePath, name_psd):
    # output_dim = 0
    use_dropout = False
    dropout_p = 0.1
    use_L2 = False
    L2weight = 1e-3
    hidden_dim = 648
    block_num = 3
    use_cosine = True
    lr = 1 * 1e-3
    lr_min = 1 * 1e-4
    batchsize = 2048
    numworks = 4
    epoch_num = 25
    # scale_node=1e2
    # scale_S=1e9
    use_standerscale = True

    folder_name = (
        f"models/{name_psd}/"
        f"{Stype}/"
        f"output_dim{output_dim}_"
        f"hdim{hidden_dim}_"
        f"block{block_num}"
        f"epoch{epoch_num}_"
        f"lr{lr}_lrmin{lr_min}_"
        f"cosine{use_cosine}_"
        f"batchsize{batchsize}_"
        # f'scale_node{scale_node}'
        # f'scale_S{scale_S}'
        f"dropout{use_dropout}{dropout_p}_"
        f"L2{use_L2}{L2weight}_"
        f"use_stsc{use_standerscale}"
    )

    os.makedirs(folder_name, exist_ok=True)

    if with_train:
        train_model.train(
            use_dropout=use_dropout, dropout_p=dropout_p,
            use_L2=use_L2, L2weight=L2weight,
            hidden_dim=hidden_dim, block_num=block_num,
            use_cosine=use_cosine, lr_min=lr_min, max_lr=lr, epoch_num=epoch_num,
            batchsize=batchsize, numworks=numworks,
            folder_name=folder_name, outputdim=output_dim, Stype=Stype, basePath=basePath
        )

    export_onnx(
        use_dropout = use_dropout,dropout_p = dropout_p,
        hidden_dim = hidden_dim,block_num = block_num,
        folder_name = folder_name, outputdim=output_dim, Stype=Stype,
        name_psd=name_psd)

    return folder_name, use_standerscale, output_dim


def eval(folder_name, model_name, use_standerscale,
         outputdim, Stype, basePath):
    pattern = re.compile(r"mlp_model_epoch(\d+)\.pth")
    epoch_files = []
    for filename in os.listdir(folder_name):
        match = pattern.match(filename)
        if match:
            epoch = int(match.group(1))
            epoch_files.append((epoch, filename))
    epoch_files.sort()

    # ????????????
    # for epoch, filename in epoch_files:
    #     print(f"Evaluating {filename}")
    #     evaluate(folder_name, filename,use_standerscale=use_standerscale)

    # ??????????????????
    print(f"Evaluating final model: {model_name}")
    evaluate(
        folder_name,
        model_name=model_name,
        use_standerscale=use_standerscale,
        outputdim=outputdim,
        Stype=Stype,
        basePath=basePath
    )

def main():
    # for i in range(1, 9):
    #     print(f'dim{i}')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--i",
        type=int,
        default=0,
        help="output dimension index"
)
    parser.add_argument(
    "--stype",
    type=str,
    default="Dis",
    choices=["Stress", "Strain", "Dis"],
    help="data type category"
)
    parser.add_argument(
    "--with_train",
    action="store_true",
    help="enable training"
)
    args = parser.parse_args()

    i = args.i
    Stype = args.stype
    with_train = args.with_train

    print(f"i{i} stype{Stype} with_train{with_train}")

    # i = 0
    # with_train = True
    # Stype = "Dis"
    name_psd = "gear19_2gearContact_geneData_Sp"
    # name_psd = "D_ModalBodySp_gearHelical_Modal"

    basePath = f"dataset/{name_psd}"

    foldername, use_standerscale, output_dim = train(
        with_train, output_dim=i, Stype=Stype,
        basePath=basePath, name_psd=name_psd
    )
    model_name = "mlp_model_best.pth"
    # model_name="mlp_model_epoch240.pth"
    eval(
        folder_name=foldername,
        model_name=model_name,
        use_standerscale=use_standerscale,
        outputdim=output_dim,
        Stype=Stype,
        basePath=basePath
    )
    gen(folder_name=foldername,model_name=model_name,
        use_standerscale=use_standerscale,outputdim=output_dim,
        Stype=Stype,basePath=basePath)


if __name__ == "__main__":
    main()

