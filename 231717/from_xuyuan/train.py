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
from nn.unet import UNet
# from nn.basenet import BASENET_CHOICES
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

    print('\nTrain Epoch: {} \ttotal_loss: {:.6f} \t avg_loss:{:.6f} \t lr :{:.7f}'.format(
         epoch, total_loss, total_loss/len(train_loader.dataset), scheduler.current_lr))


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
    # import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # Training settings
    parser = argparse.ArgumentParser(description='Scratch segmentation Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
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
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--optim', type=str, default='sgd', help="optimizer")
    parser.add_argument('--lr-scheduler', type=str, default='cos',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    # PATH Setting
    parser.add_argument('--train_path', type=str, default="../data/train/data256_0.1")
    parser.add_argument('--label_path', type=str, default="../data/train/label256_0.1")
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)


    device = torch.device("cuda" if use_cuda else "cpu")
    print('my device is :', device)

    kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(FarmDataset(train_path=args.train_path, label_path=args.label_path, istrain=True), batch_size=args.batch_size, shuffle=True,
                                               drop_last=True, **kwargs)
    #
    startepoch = 0
    # model = torch.load('./tmp/model{}'.format(startepoch)) if startepoch else UNet(3, 4).cuda()
    model = torch.load('./tmp/model{}'.format(startepoch)) if startepoch else UNet(classnames, basenet='se_resnext101_32x4d').cuda()

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
                                  args.epochs, len(train_loader.dataset), lr_step=10, warmup_epochs=10)
    dataset = args.train_path.split('/')[-1]
    model_path = os.path.join('../data/model_{}_{}'.format(dataset, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    for epoch in range(startepoch, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, scheduler, model_path)
        if epoch > 50:
            torch.save(model, os.path.join(model_path, 'model{}'.format(epoch)))
        elif epoch % 10 == 0:
            torch.save(model, os.path.join(model_path, 'model{}'.format(epoch)))


if __name__ == '__main__':

    main()

    # import time
    # print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # xray_classnames = ('0', '1', '2', '3')
    #
    # import argparse
    # parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--basenet", choices=BASENET_CHOICES, default='se_resnext101_32x4d', help='model of basenet')
    # parser.add_argument("--num-filters", type=int, default=16, help='num filters for decoder')
    # # parser.add_argument("--num_classes", type=int, default=5, help='num of output classes')
    # parser.add_argument("--classnames", type=str, default=xray_classnames, help='names of output classes')
    # parser.add_argument("--upscale-input", action='store_true',
    #                     help='scale input to make output the same size as original input')
    #
    # args = parser.parse_args()
    #
    # net = UNet(**vars(args))
    # # net = create("unet2_e_simple", "resnet101", xray_classnames)  #
    # print(net)
    # parameters = [p for p in net.parameters() if p.requires_grad]
    # n_params = sum(p.numel() for p in parameters)
    # print('N of parameters {} ({} tensors)'.format(n_params, len(parameters)))
    # encoder_parameters = [p for name, p in net.named_parameters() if p.requires_grad and name.startswith('encoder')]
    # n_encoder_params = sum(p.numel() for p in encoder_parameters)
    # print('N of encoder parameters {} ({} tensors)'.format(n_encoder_params, len(encoder_parameters)))
    # print('N of decoder parameters {} ({} tensors)'.format(n_params - n_encoder_params, len(parameters) - len(encoder_parameters)))
    #
    # x = torch.empty((1, 3, 128, 128))
    # y = net(x)
    # print(x.size(), '-->', y.size())

