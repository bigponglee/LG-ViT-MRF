"""
常量声明，设置需要的参数
"""
import torch
import time
now_time = time.strftime("%Y-%m-%d-%H", time.localtime(time.time()))

def get_block_num(block_size):
    """
    计算block数量
    """
    block_num = 220//block_size
    if 220 % block_size != 0:
        block_num += 1
    return block_num

data_length = 500
## block  [2*T, block_size, block_size]
# (220)%(block_size) == 0
# 4 5 10  11 20 22 44 55
block_size = 6
block_num = get_block_num(block_size)
# (output_dim*block_size*block_size) % global_num_heads ==0
global_num_heads = 2*block_size
batch_size = 1
# 模型加载
train_X = False  # 是否训练X
load_model = False  # 是否加载已存储的模型
model_saved_path = 'model_saved_path'  # 模型存储路径

# 存储设置
data_path = '/media/deep/E/MRF_dataset/' # 数据集路径
X_recon_path = 'recon_{}/'.format(data_length)
if train_X == True:
    data_name = 'X'
else:
    data_name = 'X_recon'
OUT_dir = '../output_lnlvit/'+'/nofe_vit_X_recon_{}_{}_b_{}/'.format(
    data_length, now_time, block_size)  # 输出路径
# 序列参数范围，归一化以方便网路训练
FA_min = 10.0
FA_max = 70.0
TR_min = 12.0
TR_max = 15.0
TE = 3.0

# 参数范围
T1_max = 5000.0
T2_max = 2500.0

# 默认数据类型，统一网络及代码中的数据类型，避免类型不匹配
dtype_float = torch.float32
dtype_complex = torch.complex64
cuda_device = '0'
device = torch.device("cuda:{}".format(cuda_device)
                      if torch.cuda.is_available() else "cpu")

# data
para_maps = 2  # 参数图数量 2=T1, T2; 3=T1, T2, PD
data_shape = (220, 220, data_length)  # 数据形状[Kx, Ky, T]
dataset_len = 380  # 数据集长度
k_sample_points = 2880  # k采样点数

# 网络参数
input_dim = 2*data_shape[2]  # input channel num complex
output_dim = para_maps
csm_path = data_path + 'csm_maps/csm_maps.mat' # coil sensitivity map path
ktraj_path = data_path + 'ktraj_nufft.mat' # k-space trajectory path
net_depth= 8  # depth of network

local_net_dim= 512
local_num_heads=16 #16


# 训练参数
LearningRate = 1e-3  # decoder学习率
scheduler_update = 500
use_scheduler = True  # 是否使用学习率调整器
iter_epoch = 50
print_every = 10    # 每隔多少个batch打印一次loss
save_every = 5     # 每隔多少个epoch保存一次模型
dc_loss_weight = 0.01  # dc loss weight
