# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 15:11:21 2019

@author: Admin
"""

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from unet import UNet2
from utils import resize_and_crop, normalize, split_img_into_squares, hwc_to_chw, merge_masks, dense_crf




if __name__ == "__main__":
    
    im = Image.open("303.png")
    
    net = UNet2(n_channels=3, n_classes=1)
    
    net.load_state_dict(torch.load("CP67.pth"))
    
    net.eval()
    

    img = resize_and_crop(im, scale=1)
    img = normalize(img)

    #left_square, right_square = split_img_into_squares(img)

    img1 = hwc_to_chw(img)
    #right_square = hwc_to_chw(right_square)

    X_left = torch.from_numpy(img1).unsqueeze(0)
    #X_right = torch.from_numpy(right_square).unsqueeze(0)
    
    

    with torch.no_grad():
        output_left = net(X_left)
        #  output_right = net(X_right)

        left_probs = output_left.squeeze(0)
        # right_probs = output_right.squeeze(0)

       
        
        left_probs = left_probs.cpu().detach().numpy()
        #right_probs = tf(right_probs.cpu())

        left_mask_np = left_probs.squeeze()
        
    left_mask_np = np.uint8(np.abs(left_mask_np*255))
        
    im2 = Image.fromarray(left_mask_np)
        
    im2.save("out_temp.jpg")
        
