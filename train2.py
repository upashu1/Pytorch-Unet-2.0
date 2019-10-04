import sys
import os
from optparse import OptionParser
import numpy as np

import torch
#import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import random
from eval import eval_net
from unet import UNet, UNet2
from utils import get_mic_ids, split_ids, split_train_val, get_mic_imgs_and_masks, batch, get_mic_test_imgs_and_masks

import timeit

def train_net(net,
              epochs=1,
              batch_size=2,
              lr=0.01,
              val_percent=.2,
              save_cp=True,
              gpu=False,startepoch=1):

    dir_img = 'micdata/train2/ds2/'
    dir_mask = 'micdata/train2/ds2/'
    dir_checkpoint = 'checkpoints_modelpth3/'
    
    print(dir_checkpoint)
    print(dir_img)
    print(dir_mask)

    ids = get_mic_ids(1,4,1)
    ids = split_ids(ids)

    iddataset = split_train_val(ids, val_percent)
    
    testids = get_mic_ids(251,270,1)
    

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    criterion = nn.BCELoss()
    start_time = timeit.default_timer()

    if (startepoch > 1):
        net.load_state_dict(torch.load(dir_checkpoint + 'CP{}.pth'.format(startepoch -1)), False)
        print('Model loaded from {}'.format(startepoch-1))
         
    for epoch in range(startepoch, startepoch+epochs):
        print('Starting epoch {}/{}.'.format(epoch, startepoch+epochs))
        start_time1 = timeit.default_timer()
        net.train()

        # reset the generators
        random.shuffle(iddataset['train'])
        train = get_mic_imgs_and_masks(iddataset['train'], dir_img, dir_mask)
        val = get_mic_imgs_and_masks(iddataset['val'], dir_img, dir_mask)

        epoch_loss = 0
        batch_count=0
        for i, b in enumerate(batch(train, batch_size)):
            #start_time2 = timeit.default_timer()
            net.train()
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b])

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()
            batch_count += 1

#            end_time2 = timeit.default_timer()
#            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))
#            print('one batch finished. Time taken = ', ((end_time2 - start_time2) / 60.) )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / batch_count))
        end_time1 = timeit.default_timer()
        print('Time taken = ', ((end_time1 - start_time1) / 60.) )

        if 1:
            val_dice = eval_net(net, val, gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))
            

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch ))
            print('Checkpoint {} saved !'.format(epoch))
            
    end_time = timeit.default_timer()
    print('training finished. Time taken = ', ((end_time - start_time) / 60.) )
    
    test = get_mic_test_imgs_and_masks(testids,dir_img, dir_mask)
    val_dice = eval_net(net, test, gpu)
    print('Dice Coeff on test images: {}'.format(val_dice))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=1, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=4,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=1e-30,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet2(n_channels=3, n_classes=1)

    #if args.load:
    #net.load_state_dict(torch.load("MODEL.pth", map_location='cpu'))
        #print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu, startepoch=68)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
