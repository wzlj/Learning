import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from farmdataset import FarmDataset

from lr_scheduler import LR_Scheduler
from PIL import Image
from tqdm import tqdm
import os
import time
import numpy as np
# torch.backends.cudnn.enabled = False
from nn.unet import *
# from unet import UNet


classnames = ('0', '1', '2', '3')

def train(args, model, device, train_loader, optimizer, epoch, scheduler, model_path):
    model.train()
    criterion = nn.BCELoss()
    # print("==============after model.train()==================")
    # best_pred = 100000
    total_loss = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)

        scheduler(optimizer, batch_idx, epoch)
        optimizer.zero_grad()
        masks_pred = model(data)

        masks_pred = torch.sigmoid(masks_pred)
        masks_probs_flat = masks_pred.view(-1)

        n, h, w = target.size()
        # print("h, w", h, w)
        temp_label = torch.FloatTensor(n, 4, h, w).cuda()
        # print(temp_label.size())
        one = torch.Tensor([1.0]).cuda()
        zero = torch.Tensor([0.0]).cuda()
        for k in range(n):
            for i in range(4):
                temp_label[k, i, :, :] = torch.where(target[k, :, :] == i, one, zero)

        true_masks_flat = temp_label.view(-1)
        loss = criterion(masks_probs_flat, true_masks_flat)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % args.log_interval == 0:
            imgd = masks_pred.detach()[0, :, :, :].cpu()
            img = torch.argmax(imgd, 0).byte().numpy() * 80

            imgx = Image.fromarray(img).convert('L')
            imgxx = Image.fromarray(target.detach()[0, :, :].cpu().byte().numpy() * 80).convert('L')

            target_path = os.path.join(model_path, "target{}.bmp".format(batch_idx))
            predict_path = os.path.join(model_path, "predict{}.bmp".format(batch_idx))
            imgx.save(predict_path)
            imgxx.save(target_path)

    avg_loss = total_loss/len(train_loader.dataset)
    print('\nTrain Epoch: {} \ttotal_loss: {:.6f} \t avg_loss:{:.6f} \t lr :{:.7f}'.format(
         epoch, total_loss, avg_loss, scheduler.current_lr))

    return total_loss, avg_loss


def main():
    # import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Training settings
    parser = argparse.ArgumentParser(description='Scratch segmentation Example')
    parser.add_argument('--batch-size', type=list, default=[3, 1], metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='Seed',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optim', type=str, default='sgd', help="optimizer")
    parser.add_argument('--lr-scheduler', type=str, default='cos',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    # PATH Setting
    parser.add_argument('--train_path', type=list, default=["../data/train/data384_0.1", "../data/train/data512_0.1"])
    parser.add_argument('--label_path', type=list, default=["../data/train/label384_0.1", "../data/train/label512_0.1"])
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)


    device = torch.device("cuda" if use_cuda else "cpu")
    print('my device is :', device)

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    # train_loader = torch.utils.data.DataLoader(FarmDataset(train_path=args.train_path, label_path=args.label_path, istrain=True), batch_size=args.batch_size, shuffle=True,
    #                                            drop_last=True, **kwargs)
    #
    startepoch = 0
    # model = torch.load('./tmp/model{}'.format(startepoch)) if startepoch else UNet(3, 4).cuda()
    model = torch.load('./tmp/model{}'.format(startepoch)) if startepoch else \
        UNet(classnames, basenet='se_resnext101_32x4d', num_filters=16, num_logit_features=16, DecoderH=DecoderDeConv, DecoderL=DecoderSimpleNBN).cuda()

    #model.parallel()
    model = torch.nn.DataParallel(model).cuda()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Define Optimizer
    train_params = model.parameters()
    weight_decay = 5e-4
    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(train_params, lr=args.lr, momentum=args.momentum,
                                    weight_decay=weight_decay, nesterov=False)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(train_params, lr=args.lr, weight_decay=weight_decay, nesterov=False)
    else:
        raise NotImplementedError("Optimizer have't been implemented")

    scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                  args.epochs, 1600, lr_step=10, warmup_epochs=10)
    dataset = args.train_path[0].split('/')[-1]
    model_path = os.path.join('../data/model_{}_{}'.format(dataset, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    for epoch in range(startepoch, args.epochs + 1):

        if epoch > 40:
            train_loader = torch.utils.data.DataLoader(
                FarmDataset(train_path=[args.train_path[1], ], label_path=[args.label_path[1], ], istrain=True),
                batch_size=args.batch_size[1],
                shuffle=True,
                drop_last=True,
                **kwargs)
            total, avg = train(args, model, device, train_loader, optimizer, epoch, scheduler, model_path)
            torch.save(model, os.path.join(model_path, 'model{}_{}_{}'.format(epoch, total, avg)))
        else:
            train_loader = torch.utils.data.DataLoader(
                FarmDataset(train_path=[args.train_path[0], ], label_path=[args.label_path[0], ], istrain=True),
                batch_size=args.batch_size[0],
                shuffle=True,
                drop_last=True,
                **kwargs)
            total, avg = train(args, model, device, train_loader, optimizer, epoch, scheduler, model_path)
            if  epoch % 10 == 0:
                torch.save(model, os.path.join(model_path, 'model{}_{}_{}'.format(epoch, total, avg)))


if __name__ == '__main__':

    main()


