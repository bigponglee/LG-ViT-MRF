import logging
from configs import lgvit_cfg as cfg
from torch.nn.functional import pad as F_pad
import torch
import os


def input_padding(x):
        '''
        Args:
            x: [batch, channel, height_in, width_in]
        Returns:
            out: [batch, channel, height_out, width_out]
        '''
        if len(x.shape) == 3:
            batch = 1
            channel, height_in, width_in = x.shape
        else:
            batch, channel, height_in, width_in = x.shape
        height_out = cfg.block_num * cfg.block_size
        width_out = cfg.block_num * cfg.block_size
        if height_out != height_in or width_out != width_in:
            x = F_pad(x, (0, width_out-width_in, 0, height_out-height_in))
        return x


def out_unpadding(x):
        '''
        Args:
            x: [batch, channel, height_in, width_in]
        Returns:
            out: [batch, channel, height_out, width_out]
        '''
        height_out = 220
        width_out = 220
        x = x[:,:, :height_out, :width_out]
        return x


def get_batch_index(LEN, batch_size):
    '''
    Args:
        LEN: 数据集长度
        batch_size: batch大小
    Returns:
        batch_index: batch索引
    '''
    batch_size = int(batch_size)
    if LEN <= batch_size:
        raise ValueError(
            'The LEN should be longer than batch_size.')
    num = LEN//batch_size
    yu = LEN % batch_size
    out = []
    for i in range(num):
        out.append([i*batch_size, (i+1)*batch_size])
    if yu != 0:
        out.append([num*batch_size, LEN])
    return out


def def_logger(path):
    """

    Args:
        path: logger存储路径

    Returns:

    """
    LOGGING_LEVEL = logging.DEBUG
    logger = logging.getLogger(__name__)
    logger.setLevel(level=LOGGING_LEVEL)

    handler = logging.FileHandler(path)
    handler.setLevel(LOGGING_LEVEL)
    formatter = logging.Formatter(
        '%(asctime)s--->>>%(message)s', "%Y/%m/%d--%H:%M:%S")
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(LOGGING_LEVEL)
    console.setFormatter(formatter)

    logger.addHandler(handler)
    logger.addHandler(console)

    return logger


def get_block_index(data_shape, block_size, overlap_size):
    '''
    Args:
        data_shape: 数据形状
        block_size: block大小
        overlap_size: overlap大小
    Returns:
        block_index: block索引 [x- , x+ , y- , y+]
        overlap_index: overlap索引
    '''
    if data_shape[0] % block_size != 0 or data_shape[1] % block_size != 0:
        raise ValueError(
            'The data_shape should be a multiple of block_size.')
    block_index = []
    overlap_index = []
    for i in range(data_shape[0]//block_size):
        for j in range(data_shape[1]//block_size):
            block_index.append([i*block_size, (i+1)*block_size,
                                j*block_size, (j+1)*block_size])
            overlap_index.append([i*block_size, (i+1)*block_size+2*overlap_size,
                                  j*block_size, (j+1)*block_size+2*overlap_size])
    return block_index, overlap_index


def get_block_index_v2(data_shape, block_size):
    '''
    Args:
        data_shape: 数据形状
        block_size: block大小
    Returns:
        block_index: block索引 [x- , x+ , y- , y+]
    '''
    if data_shape[0] % block_size != 0 or data_shape[1] % block_size != 0:
        raise ValueError(
            'The data_shape should be a multiple of block_size.')
    block_index = []
    for i in range(data_shape[0]//block_size):
        for j in range(data_shape[1]//block_size):
            block_index.append([i*block_size, (i+2)*block_size,
                                j*block_size, (j+2)*block_size])
    return block_index


def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    if block_mem >0:
        device = torch.device("cuda:{}".format(cuda_device)
                      if torch.cuda.is_available() else "cpu")
        x = torch.ones((256, 1024, block_mem), device=device)
        del x
