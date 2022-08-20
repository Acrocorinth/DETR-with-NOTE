# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

# 提取训练参数


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

# 执行训练脚本,分布式8卡训练


def main(args):
    utils.init_distributed_mode(args)  # 根据参数初始化分布式进程
    print("git:\n  {}\n".format(utils.get_sha()))  # 验证git版本是否满足
    # 冻结权重
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)  # 打印所有参数

    # 实例化device对象
    device = torch.device(args.device)

    # fix the seed for reproducibility,当seed固定则每张卡的种子取决于rank
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # model,criterion(损失)&postprocessors都通过build_model生成,
    model, criterion, postprocessors = build_model(args)
    # ?模型在torch里的数据类型是什么
    model.to(device)
    '''
    DDP将input module复制到device ID对应的设备 相应的按batch维度扔进模型,并将输出收集到output_device
    checkpoint加载时,需要提供适当的map_location参数,防止进程进入其他人的设备,如果缺少map_location,torch.load将首先
    把模型加载到CPU,然后将每个参数复制到相应的位置,本容器可以将输入数据在batch维度平均分配到各个设备上(数据并行),把模型在
    每个设备上复制一份(模型并行),每个模型副本负责处理一部分输入数据,在反向传播的过程中,每个节点的梯度将被平均.

    如果device_ids没有设置,DDP的device将默认设为所有可见的GPU,

    进程开始时,需要用os.environ['CUDA_VISABLE_DEVICES'] = local_rank 在进程中指定可见的GPU

    find_unused_parameters参数代表模型中不参与梯度更新的参数,需要设置为True来避免出错

    DDP同时还用了ring-reduce的思想,即所有GPU连成一个环,每个GPU只需要与上下游GPU通信,循环两次即可获得全局信息.

    DDP最核心的机制是梯度的全局同步,也就是reduce机制,所有参与梯度同步的参数将执行全局梯度同步,默认是梯度平均操作,即将所有卡
    上的梯度值进行平均,用平均值更新所有参数,只要保证参数的初始值一致,即可保证参数在各个GPU上的状态时刻保持一致,模型的参数不仅
    包括parameters 还包括buffer(不参与反向传播的参数),每次网络传播开始前,DDP会将buffer广播给其他节点,保持buffer的全局一致

    DDP只自动执行了梯度的全局同步,并没有对loss进行全局同步,所以只能看到单卡的loss,DDP提供了手动的reduce接口:all_reduce
    可以在每次打印loss前手动调用all_reduce实现loss的全局平均.

    1. 开启多进程,使用torch.Distributed.launch.py开启,也可以用torch.multiprocessing.spawn开启,每个进程需要分配一个
    rank,范围是0~GPU-1
    2. 用dist.init_process_group进行多进程初始化,指定backend,一般为nccl
    3. 用torch.cuda.set_device(rank)设定当前进程使用的GPU
    4. 使用DistributedSampler根据每个rank对每个进程的输入数据进行分配:self.train_sampler = DistributedSampler(train_dataset)
    并作为参数传给dataloader,这一步是为每个进程分配不同的数据
    5. 全局同步BN(syncBN)训练时在网络内部进行了归一化,为训练过程提供正则化,防止了中间层feature map的协方差偏移,有助于抑制过拟合
    使用BN不需要特别依赖于初始化参数,可以使用较大的学习率,加速模型的训练过程.
    6. 使用DDP对模型进行包装:model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank],find_unused_parameters=True)
    7. 使用DDP后,模型在每个GPU上都复制了一份,而且被包装了一层,之前的model变成了现在的model.module,在保存模型时要这样操作:
    加载模型只需要在构造DDP模型之前,在master节点上加载.


    '''
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # 给backbone部分设置单独的学习率
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # AdamW优化器
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # 训练集
    dataset_train = build_dataset(image_set='train', args=args)
    # 测试集
    dataset_val = build_dataset(image_set='val', args=args)
    # 分布式训练的sampler
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    # 打包batch
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    # TODO load 训练集 batch_sampler需要深入了解
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # 如果数据集是CoCo_panoptic,则需要调用相应的CoCo_val api
    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)
    # 如果要加载预训练全总并且frozen, 需要用自定义的load_state_dict来load frozen_weights来获得checkpoint
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    # 根据Path对象得到输出文件夹
    output_dir = Path(args.output_dir)
    # args.resume
    if args.resume:
        # 如果https开头,那么就通过下载的方式把云端的checkpoints load进CPU
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            # 不是http开头的话就用从本地路径load的方式torch.loadcheckpoint
            checkpoint = torch.load(args.resume, map_location='cpu')
        # 用系统的load_state_dict获得checkpoint
        model_without_ddp.load_state_dict(checkpoint['model'])
        # 如果checkpoint中有优化器和lr下降策略和epoch,则使用checkpoints中的训练策略
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
    # 如果eval-flag启动,则通过evaluate函数生成test_stats和CoCo_evaluator
    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            # save_on_master是存到主线程中的函数,将eval结果和pth保存
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return
    '''
    以下为训练进程:
    '''

    print("Start training")
    # 首先记录开始时间
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):  # 前闭后开
        if args.distributed:
            sampler_train.set_epoch(epoch)  # 每个dataloader的sampler要根据epoch不同而变化
            # 根据train_one_epoch函数得到train_stats
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']  # checkpoint保存点
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:  # lr学习策略drop之前
                # 把所有可能的checkpoints文件名append到checkpoint中
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                # 开始遍历
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({  # 在master进程里保存model optimizer lr_scheduler epoch args等参数
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        # 经过evaluate函数得到test_stats和CoCo_evaluator
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )
        # 在JSON文件里记录log
        #**双星号代表将任意数量的参数以字典的形式传入, item会返回字典里的一对key和value
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},#stats里的参数会被打印
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}
        # 在main_process里存储log.txt文件,将log_stats写入TXT中
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n") #json.dumps将log_stats从字典转换为json格式

            # for evaluation logs(此处的evaluator是CoCo)
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:  # ?是否会每个epoch都循环一次保存
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)
    # 记录所需要的完整时间
    total_time = time.time() - start_time
    # 转为字符串 print出来
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
