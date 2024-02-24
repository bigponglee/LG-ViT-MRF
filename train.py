'''
train
'''
import torch
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from utils.func import def_logger, input_padding, out_unpadding
from model.lgvit import Vit_global
# from model.dc import Data_Consistency, gen_atb
from dataset.datasets import train_dataloader
from configs import lgvit_cfg as cfg
from einops.layers.torch import Rearrange


def train(data_index_list):
    #####################文件路径##########################################
    OUT_dir = cfg.OUT_dir
    Save_NET_root = OUT_dir+'/Model_save'
    try:
        os.mkdir(OUT_dir)
        logger = def_logger(OUT_dir+'/log.txt')
    except:
        logger = def_logger(OUT_dir+'/log.txt')
        logger.info('文件夹已经存在！！！')
    try:
        os.mkdir(Save_NET_root)
    except:
        logger.info('文件夹已经存在！！！')
    shutil.copy('./configs/lnlvit_configs.py', OUT_dir)  # 复制配置文件
    ####################可视化###########################################
    writer_train = SummaryWriter(log_dir=OUT_dir+'/runs')  # tensorboard
    logger.info('tensorboard saved at: {}'.format(OUT_dir+'/runs'))
    ####################构建网络#########################################
    Net = Vit_global( input_dim = cfg.input_dim, block_size=cfg.block_size, 
                block_num = cfg.block_num, net_dim=cfg.local_net_dim, 
                net_depth=cfg.net_depth, out_dim=cfg.para_maps).to(cfg.device)
    if cfg.load_model:
        Net.load_state_dict(torch.load(cfg.model_saved_path))
        logger.info('load decoder from: {}'.format(cfg.model_saved_path))
    logger.info(
        "===================================Net Build======================================")
    ####################优化器#########################################
    Optimizer = torch.optim.Adam([
        {'params': Net.parameters(), 'lr': cfg.LearningRate},
    ])
    if cfg.use_scheduler:  # 学习率衰减
        Scheduler_decoder = torch.optim.lr_scheduler.StepLR(
            Optimizer, step_size=1, gamma=0.95)
    logger.info('Optimizer: Adam, scheduler: {}'.format(False))
    ####################损失函数#########################################
    Loss_para_maps = torch.nn.MSELoss().to(cfg.device)
    Loss_para_local = torch.nn.MSELoss().to(cfg.device)
    logger.info('using Loss Function: MSE Loss')
    ####################dataset#########################################
    data_path = cfg.data_path
    data_name = cfg.data_name
    X_recon_path = cfg.X_recon_path
    logger.info(
        "===================================Training START=====================================")
    loss_all_avg = 0.0
    loss_local_avg = 0.0
    loss_global_avg = 0.0
    iter_i = 0
    Net.train()
    for e in range(cfg.iter_epoch):
        data_loader = train_dataloader(
            data_index_list, data_path, X_recon_path, data_name, cfg.batch_size)
        for X, para_maps, max_X in data_loader:
            with torch.no_grad():
                para_maps = input_padding(para_maps)
                para_local = Rearrange('b c (h b1) (w b2) -> (b h w) c b1 b2', b1=cfg.block_size,
                         b2=cfg.block_size)(para_maps)
                para_maps = out_unpadding(para_maps)
            Optimizer.zero_grad()
            para_local_est, para_est = Net(X)
            loss_local = Loss_para_local(para_local_est, para_local)
            loss_fianl = Loss_para_maps(para_est, para_maps)
            loss = loss_local + loss_fianl
            loss.backward()
            Optimizer.step()

            writer_train.add_scalar('loss', loss.item(), iter_i)
            writer_train.add_scalar('loss_local', loss_local.item(), iter_i)
            writer_train.add_scalar('loss_fianl', loss_fianl.item(), iter_i)
            loss_all_avg += loss.item()
            loss_local_avg += loss_local.item()
            loss_global_avg += loss_fianl.item()

            if cfg.use_scheduler and iter_i % cfg.scheduler_update == 0:
                Scheduler_decoder.step()

            if iter_i % cfg.print_every == cfg.print_every-1:
                logger.info('Epoch: {}, iter: {}, Avg: loss: {:.6f}; Local: {:.6f}; Global: {:.6f}'.format(
                    e, iter_i, loss_all_avg / cfg.print_every, loss_local_avg / cfg.print_every, loss_global_avg / cfg.print_every))
                writer_train.add_scalar(
                    'Avg-loss', loss_all_avg / cfg.print_every, global_step=iter_i)
                writer_train.add_scalar(
                    'Avg-loss_local', loss_local_avg / cfg.print_every, global_step=iter_i)
                writer_train.add_scalar(
                    'Avg-loss_global', loss_global_avg / cfg.print_every, global_step=iter_i)

                # reset to zero
                loss_all_avg = 0.0
                loss_local_avg = 0.0
                loss_global_avg = 0.0

            writer_train.add_scalar('learning_rate',
                                    Optimizer.param_groups[-1]['lr'],
                                    global_step=iter_i)
            iter_i += 1
        if e % cfg.save_every == cfg.save_every-1:
            torch.save(Net.state_dict(), Save_NET_root+'/Net_{}.pth'.format(e))
            logger.info('save model at: {}'.format(
                Save_NET_root+'/Net_{}.pth'.format(e)))
    torch.save(Net.state_dict(), Save_NET_root+'/Net_final.pth')
    logger.info('save model at: {}'.format(Save_NET_root+'/Net_final.pth'))
    logger.info(
        "===================================END======================================")


if __name__ == "__main__":
    data_index_list = list(range(1, 380)) #数据集的 index
    test_data_index_list = [1,2,3] # 更换为随机生成的测试集 index
    data_index_list = [
        i for i in data_index_list if i not in test_data_index_list]
    train(data_index_list)
