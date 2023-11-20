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


logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description='DSNet')
    # Network
    parser.add_argument('--network', type=str, help='Path to saved network to evaluate')
    # Dataset & Data & Training
    parser.add_argument('--dataset', type=str, default="cornell", help='Dataset Name ("cornell" or "jacquard")')
    parser.add_argument('--dataset-path', type=str, default="F:\GraspLab\datasets\cornell", help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=0, help='Use Depth image for training (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1, help='Use RGB image for training (0/1)')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='交叉验证 Shift the start point of the dataset to use a different test/train split for cross validation.')
    parser.add_argument('--num-workers', type=int, default=8, help='Dataset workers')

    parser.add_argument('--batch-size', type=int, default=12, help='Batch size')
    parser.add_argument('--vis', type=bool, default=False, help='vis')
    parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=200, help='Batches per Epoch')
    parser.add_argument('--val-batches', type=int, default=32, help='Validation Batches')
    # Logging etc.
    parser.add_argument('--description', type=str, default='', help='Training description')
    parser.add_argument('--outdir', type=str, default='output/models/', help='Training Output Directory')

    args = parser.parse_args()
    return args

def run():
    args = parse_args()
    # Load Network
    # model = DSNetSys()
    # net = model.load_state_dict(torch.load(args.network))
    net = torch.load(args.network)
    device = torch.device("cuda:0")

    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)  # if args.dataset=Cornell, 这句话代表 Dataset = CornellDataset
    val_dataset = Dataset(args.dataset_path, start=args.split, end=1.0, ds_rotate=args.ds_rotate,
                          random_rotate=True, random_zoom=False,
                          include_depth=args.use_depth, include_rgb=args.use_rgb)
    val_data = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )
    logging.info('Done')

    test_results = validate(net, device, val_data, args.val_batches)
    # 打印：正确预测的数量、预测数量（对+错）、准确率
    logging.info('%d/%d = %f' % (test_results['correct'], test_results['correct'] + test_results['failed'],
                                 test_results['correct'] / (test_results['correct'] + test_results['failed'])))