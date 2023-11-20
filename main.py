import datetime
import os
import sys
import argparse
import logging
import torch
import torch.utils.data
import torch.optim as optim
from torchsummary import summary
from traning import train, validate
from utils.data import get_dataset
from models.DSNet import DSNetSys
from torch.optim.lr_scheduler import StepLR, ExponentialLR

# import matplotlib
# matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='DSNet')

    # Network
    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, default="cornell", help='Dataset Name ("cornell" or "jacquard")')
    parser.add_argument('--dataset-path', type=str, default="F:\GraspLab\datasets\cornell", help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=1, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.9, help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=True, help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.,
                        help='交叉验证 Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--vis', type=bool, default=False, help='vis')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=100, help='Validation Batches')
    
    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/cornell-test', help='Training Output Directory')

    args = parser.parse_args()
    return args


def run():
    args = parse_args()
    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    net_desc = '{}_{}'.format(dt, '_'.join(args.description.split()))

    save_folder = os.path.join(args.outdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # tb = tensorboardX.SummaryWriter(os.path.join(args.logdir, net_desc))

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)  # if args.dataset=Cornell, 这句话代表 Dataset = CornellDataset
    train_dataset = Dataset(args.dataset_path,
                            start=0.0,
                            end=args.split,
                            ds_rotate=args.ds_rotate,
                            ds_shuffle=args.ds_shuffle,
                            random_rotate=True,
                            random_zoom=True,
                            include_depth=args.use_depth,
                            include_rgb=args.use_rgb)
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_dataset = Dataset(args.dataset_path,
                          start=args.split,
                          end=1.0,
                          ds_rotate=args.ds_rotate,
                          ds_shuffle=args.ds_shuffle,
                          random_rotate=False,
                          random_zoom=False,
                          include_depth=args.use_depth,
                          include_rgb=args.use_rgb)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    logging.info('Done')

    # Load Model
    logging.info('Loading Network...')
    input_channels = 1 * args.use_depth + 3 * args.use_rgb
    net = DSNetSys(in_chans=input_channels, embed_dim=96, drop_rate=0.6, attn_drop_rate=0.6, drop_path_rate=0.6, num_heads=[1, 2, 4, 8])
    device = torch.device("cuda:0")
    net = net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr=0.002)  # lr=0.0001 下一次尝试lr=0.01开始
    # optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4) # 0.0001
    # listy = [x * 2 for x in range(1, 1000, 5)]  # [2, 10, 18, 26, ..., 1998]
    # 创建一个多步学习率调度器，根据里程碑列表 listy 进行学习率的调整，每次乘以 0.5。
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=listy, gamma=0.5)
    # scheduler = StepLR(optimizer, step_size=300, gamma=0.1)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=0.0001)
    # scheduler = None
    logging.info('Done')

    # 加载之前训练好的权重
    pretrained_weights = torch.load(r"F:\GraspLab\DSNet-cloud\output\models\jacquard-RGBD-0.98\231021_1100_\epoch_04_iou_0.956_statedict.pt")
    net.load_state_dict(pretrained_weights)

    # 写入模型net的概要信息
    summary(net, (input_channels, 224, 224))
    f = open(os.path.join(save_folder, 'net.txt'), 'w')
    sys.stdout = f
    summary(net, (input_channels, 224, 224))
    sys.stdout = sys.__stdout__
    f.close()

    # 模型训练与训练时的验证
    best_iou = 0.0
    for epoch in range(args.epochs):
        logging.info('Beginning Epoch {:02d}'.format(epoch))
        print("current lr:", optimizer.state_dict()['param_groups'][0]['lr'])
        # for i in range(5000):
        train_results = train(epoch, net, device, train_data, optimizer, args.batches_per_epoch, vis=args.vis)
        scheduler.step()
        # Run Validation
        # logging.info('Validating...')
        test_results = validate(net, device, val_data, args.val_batches)
        # 打印：正确预测的数量、预测数量（对+错）、准确率
        logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                     test_results['correct'] / (test_results['correct'] + test_results['failed'])))
        # 计算并记录 IoU
        iou = test_results['correct'] / (test_results['correct'] + test_results['failed'])
        if epoch % 5 == 0 or iou > best_iou or iou > 0.9:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.3f' % (epoch, iou)))
            torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%02d_iou_%0.3f_statedict.pt' % (epoch, iou)))
            best_iou = iou
        if epoch == args.epochs - 1:
            torch.save(net, os.path.join(save_folder, 'epoch_%02d_iou_%0.3f' % (epoch, iou)))
            torch.save(net.state_dict(), os.path.join(save_folder, 'epoch_%02d_iou_%0.3f_statedict.pt' % (epoch, iou)))
        else:
            # Save an empty file with the same name format
            empty_file_path = os.path.join(save_folder, 'epoch_%02d_iou_%0.3f_empty' % (epoch, iou))
            with open(empty_file_path, 'w') as empty_file:
                pass  # Create an empty file


if __name__ == '__main__':
    run()
