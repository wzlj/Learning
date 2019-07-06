import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from farmdataset import FarmDataset
from unet import UNet
from lr_scheduler import LR_Scheduler
from PIL import Image

import numpy as np
# torch.backends.cudnn.enabled = False


def train(args, model, device, train_loader, optimizer, epoch, scheduler):
    model.train()
    criterion = nn.BCELoss()
    print("++++++++++++ after model.train()")
    # best_pred = 100000
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # print("\n\n  ========================")
        # print(data)

        scheduler(optimizer, batch_idx, epoch)
        optimizer.zero_grad()
        masks_pred = model(data)
        # print('masks_pred size', masks_pred.size(),masks_pred)
        # print('target size', target.size(), target)
        # masks_pred = F.log_softmax(output, dim=1)

        # print(masks_pred)

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
        # best_pred = loss.item()
        optimizer.step()

        # time.sleep(0.6)#make gpu sleep
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \t lr :{:.7f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item(), scheduler.current_lr))
        # if epoch % 1 == 0:
            imgd = masks_pred.detach()[0, :, :, :].cpu()
            img = torch.argmax(imgd, 0).byte().numpy() * 80
            print(img.shape)
            imgx = Image.fromarray(img).convert('L')
            imgxx = Image.fromarray(target.detach()[0, :, :].cpu().byte().numpy() * 80).convert('L')

            imgx.save("./tmp/predict{}.bmp".format(batch_idx))
            imgxx.save('./tmp/real{}.bmp'.format(batch_idx))


def test(args, model, device, testdataset, issave=False):
    model.eval()
    test_loss = 0
    correct = 0
    evalid = [i + 7 for i in range(0, 2100, 15)]
    maxbatch = len(evalid)
    with torch.no_grad():
        for idx in evalid:
            data, target = testdataset[idx]
            data, target = data.unsqueeze(0).to(device), target.unsqueeze(0).to(device)
            # print(target.shape)
            target = target[:, :1472, :1472]
            output = model(data[:, :, :1472, :1472])
            output = F.log_softmax(output, dim=1)
            loss = nn.NLLLoss2d().to('cuda')(output, target)
            test_loss += loss

            r = torch.argmax(output[0], 0).byte()

            tg = target.byte().squeeze(0)
            tmp = 0
            count = 0
            for i in range(1, 4):
                mp = r == i
                tr = tg == i
                tp = mp * tr == 1
                t = (mp + tr - tp).sum().item()
                if t == 0:
                    continue
                else:
                    tmp += tp.sum().item() / t
                    count += 1
            if count > 0:
                correct += tmp / count

            if issave:
                Image.fromarray(r.cpu().numpy()).save('predict.png')
                Image.fromarray(tg.cpu().numpy()).save('target.png')
                input()

    print('Test Loss is {:.6f}, mean precision is: {:.4f}%'.format(test_loss / maxbatch, correct))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Scratch segmentation Example')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optim', type=str, default='sgd', help="optimizer")
    parser.add_argument('--lr-scheduler', type=str, default='cos',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print('my device is :', device)

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(FarmDataset(istrain=True), batch_size=args.batch_size, shuffle=True,
                                               drop_last=True, **kwargs)
    #
    startepoch = 0
    model = torch.load('./tmp/model{}'.format(startepoch)) if startepoch else UNet(3, 4).cuda()

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
                                  args.epochs, 157, lr_step=10, warmup_epochs=10)

    for epoch in range(startepoch, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, scheduler)
        if epoch > 50:
            torch.save(model, './tmp/model{}'.format(epoch))
        elif epoch % 10 == 0:
            torch.save(model, './tmp/model{}'.format(epoch))


if __name__ == '__main__':
    main()

    # model = UNet(3, 4).cuda()
    # input_arr = torch.randn(4, 3, 256, 256).cuda()
    # out = model(input_arr)
    # print(out.shape)
