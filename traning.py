import torch
import torch.utils.data
import torch.optim as optim
from utils.dataset_processing import evaluation
from models.common import post_process_output
from utils.visualisation.gridshow import gridshow
import logging
import cv2


def validate(net, device, val_data, batches_per_epoch):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param batches_per_epoch: Number of batches to run
    :return: Successes, Failures and Losses
    """

    # 不启用 BN 和 Dropout
    net.eval()

    results = {
        'correct': 0,
        'failed': 0,
        'loss': 0,
        'losses': {

        }
    }

    ld = len(val_data)

    with torch.no_grad():
        batch_idx = 0
        while batch_idx < batches_per_epoch:
            for x, y, didx, rot, zoom_factor in val_data:
                batch_idx += 1
                if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
                    break

                xc = x.to(device)
                yc = [yy.to(device) for yy in y]
                lossd = net.compute_loss(xc, yc)

                loss = lossd['loss']

                results['loss'] += loss.item() / ld
                for ln, l in lossd['losses'].items():
                    if ln not in results['losses']:
                        results['losses'][ln] = 0
                    results['losses'][ln] += l.item() / ld

                q_out, ang_out, w_out = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                            lossd['pred']['sin'], lossd['pred']['width'])

                s = evaluation.calculate_iou_match(q_out, ang_out,
                                                   val_data.dataset.get_gtbb(didx, rot, zoom_factor),
                                                   no_grasps=2,
                                                   grasp_width=w_out,
                                                   )

                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

    return results


def train(epoch, net, device, train_data, optimizer, batches_per_epoch, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {
        'loss': 0,
        'losses': {
        }
    }
    # 启用 BN 和 Dropout
    net.train()
    batch_idx = 0
    # Use batches per epoch to make training on different sized datasets (cornell/jacquard) more equivalent.
    # 使用 batches per epoch 使在不同大小的数据集（cornell/jacquard）上进行的训练更加等效
    while batch_idx < batches_per_epoch:
        #
        for x, y, _, _, _ in train_data:
            # print("shape:",x.shape)
            batch_idx += 1
            # if batch_idx >= batches_per_epoch:
            #     break
            # print("x_0:",x[0].shape,y[0][0].shape)
            # plt.imshow(x[0].permute(1,2,0).numpy())
            # plt.show()
            # plt.imshow(y[0][0][0].numpy())
            # plt.show()
            # 将输入数据和标签数据转移到指定的设备（例如 GPU）上进行计算
            xc = x.to(device)
            yc = [yy.to(device) for yy in y]
            # 调用模型的 compute_loss 方法计算损失，返回一个字典包含损失和各个损失的统计信息
            lossd = net.compute_loss(xc, yc)
            # 从损失字典中获取总损失
            loss = lossd['loss']
            # 每隔10个批次，打印损失信息
            if batch_idx % 10 == 0:
                logging.info('Epoch: {}, Batch: {}, Loss: {:0.4f}'.format(epoch, batch_idx, loss.item()))
            # 累计总损失
            results['loss'] += loss.item()
            # 将每个训练批次中计算得到的各个损失项的值
            # 累加到 results 字典中的相应损失项中，
            # 以便后续计算平均损失和统计信息
            for ln, l in lossd['losses'].items():
                if ln not in results['losses']:
                    results['losses'][ln] = 0
                results['losses'][ln] += l.item()
            # 清空优化器的梯度
            optimizer.zero_grad()
            # 反向传播计算梯度
            loss.backward()
            # 反向传播计算梯度
            optimizer.step()

            # Display the images
            if vis:
                imgs = []
                n_img = min(4, x.shape[0])
                for idx in range(n_img):
                    imgs.extend([x[idx,].numpy().squeeze()] + [yi[idx,].numpy().squeeze() for yi in y] + [
                        x[idx,].numpy().squeeze()] + [pc[idx,].detach().cpu().numpy().squeeze() for pc in
                                                      lossd['pred'].values()])
                gridshow('Display', imgs,
                         [(xc.min().item(), xc.max().item()), (0.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (0.0, 1.0)] * 2 * n_img,
                         [cv2.COLORMAP_BONE] * 10 * n_img, 10)
                cv2.waitKey(2)

    # 得到训练过程中的平均损失和各个损失项的平均值
    results['loss'] /= batch_idx
    for l in results['losses']:
        results['losses'][l] /= batch_idx

    return results
