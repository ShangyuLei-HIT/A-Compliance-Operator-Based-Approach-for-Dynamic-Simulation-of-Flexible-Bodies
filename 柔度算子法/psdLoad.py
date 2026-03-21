"""
Author: 雷尚谕
"""
import numpy as np

def read_dataset_in_chunks(dset, block_rows=1000_000):
    """
    以分块的方式读取任意一维/多维数组，并最终 reshape 成 (-1,1)
    不爆内存
    """
    total = dset.shape[0]
    chunks = []

    for start in range(0, total, block_rows):
        end = min(start + block_rows, total)
        block = dset[start:end]           # 一次只读几万到几十万行，安全
        block = block.reshape(-1, 1)      # 强制列向量
        chunks.append(block)

    return np.vstack(chunks)              # 拼接成完整列向量
