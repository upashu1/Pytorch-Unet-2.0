#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw

#function to read mask from text file made by ashu

def get_mic_mask_from_text(ids, dir, suffix='gt.txt'):
    
    for id,pos in ids:
#        if pos==1:
#            f=open((dir +id + suffix),"r")
#            words =f.read().split(',')
#            #words=list(map(lambda s:s.strip,words))
#            int_list = list(map(int,words))
#            a=np.array(int_list, dtype=np.float32)
#            a= a/255
#            a=np.reshape(a,(512,512))
#            yield a
         f=open((dir +id + suffix),"r")
         words =f.read().split(',')
         #words=list(map(lambda s:s.strip,words))
         int_list = list(map(int,words))
         a=np.array(int_list, dtype=np.float32)
         a= a/255
         a=np.reshape(a,(512,512))
         yield a[(pos % 4)*128 : (pos %4) *128 + 128, np.uint8(pos / 4)*128: np.uint8(pos /4)*128 + 128]


def get_mic_test_mask_from_text(ids, dir, suffix='gt.txt'):
    
    for id in ids:
#        if pos==1:
#            f=open((dir +id + suffix),"r")
#            words =f.read().split(',')
#            #words=list(map(lambda s:s.strip,words))
#            int_list = list(map(int,words))
#            a=np.array(int_list, dtype=np.float32)
#            a= a/255
#            a=np.reshape(a,(512,512))
#            yield a
         f=open((dir +id + suffix),"r")
         words =f.read().split(',')
         #words=list(map(lambda s:s.strip,words))
         int_list = list(map(int,words))
         a=np.array(int_list, dtype=np.float32)
         a= a/255
         a=np.reshape(a,(512,512))
         yield a
      
        

def mic_to_cropped_imgs(ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id,pos in ids:
#        if pos==1:
#            im = Image.open(dir + id + suffix)
#            a = np.array(im, dtype=np.float32)
#            a.shape
#            a.shape[0]
#            yield a
            
        im = Image.open(dir + id + suffix)
        a = np.array(im, dtype=np.float32)
        yield a[(pos % 4)*128 : (pos %4) *128 + 128, np.uint8(pos / 4)*128: np.uint8(pos /4)*128 + 128]


def mic_test_to_cropped_imgs(ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
#        if pos==1:
#            im = Image.open(dir + id + suffix)
#            a = np.array(im, dtype=np.float32)
#            a.shape
#            a.shape[0]
#            yield a
            
        im = Image.open(dir + id + suffix)
        a = np.array(im, dtype=np.float32)
        yield a


def mic_to_cropped_masks(ids, dir, suffix):
    """From a list of tuples, returns the correct cropped img"""
    for id,pos in ids:
#        if pos==1:
#            im = Image.open(dir + id + suffix)
#            
#            a = np.array(im, dtype=np.float32)
#            a = a/255
#            a.shape
#            a.shape[0]
#            yield np.mean(a, axis=-1)
            
        im = Image.open(dir + id + suffix)
        a = np.array(im, dtype=np.float32)
        a = a/255
#        a = np.mean(a, axis=-1)
        yield a[(pos % 4)*128 : (pos %4) *128 + 128, np.uint8(pos / 4)*128: np.uint8(pos /4)*128 + 128]

        
def get_mic_imgs_and_masks(ids, dir_img, dir_mask):
    """Return all the couples (img, mask)"""

    imgs = mic_to_cropped_imgs(ids, dir_img, '.png')

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)
    #testing for perfect reconstruction
#    masks = mic_to_cropped_masks(ids, dir_img, 'gt.png')

    masks = get_mic_mask_from_text(ids, dir_mask)
    

    return zip(imgs_normalized, masks)
    

def get_mic_test_imgs_and_masks(ids, dir_img, dir_mask):
    """Return all the couples (img, mask)"""

    imgs = mic_test_to_cropped_imgs(ids, dir_img, '.png')

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)
    #testing for perfect reconstruction
#    masks = mic_to_cropped_masks(ids, dir_img, 'gt.png')

    masks = get_mic_test_mask_from_text(ids, dir_mask)
    

    return zip(imgs_normalized, masks, ids)



def get_mic_ids(start=1, stop=20, interval=1):
    """Returns a list of the ids in the directory"""
    return (str(f) for f in range(start,stop,interval))

def get_mic_test_ids(start=1, stop=20, interval=1):
    """Returns a list of the ids in the directory"""
    return [str(f) for f in range(start,stop,interval)]

def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=16):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i)  for id in ids for i in range(n))


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield get_square(im, pos)

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '_mask.gif', scale)

    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '_mask.gif')
    return np.array(im), np.array(mask)


