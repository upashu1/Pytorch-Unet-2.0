# Pytorch-Unet-2.0  for noisy image segmentation
Improved UNet CNN Deep Learning Image Segmentation for noisy images. More accurate, More stable. Trained for microarray images.

#Note
#[#Download the repository from here](https://github.com/upashu1/Pytorch-UNet-2)



![input file](303.png) ![output file unet2](303outmaskunet2.png)

UNet 2.0 is a modified version of UNet for better segmentation even image is noisy. Below is the pictorial view difference between UNet and UNet 2.0.
![picutre of unet and unet2](Unet2.png)

[This](https://github.com/upashu1/Pytorch-UNet-2) is a forked version of https://github.com/milesial/Pytorch-UNet. To know more about it, 
[Click Here For Original Edition](https://github.com/milesial/Pytorch-UNet) 

## Usage
**Note : Use Python 3**

Download pretrained network [CP67.zip](https://storage.googleapis.com/static.wixstatic.com/raw/2895ae_32b397c6f49445329607aa7bd4f8ae64.zip?Expires=1570168609&GoogleAccessId=download-urls%40wixprivatemedia.iam.gserviceaccount.com&Signature=ET%2BnwU9vXrQ8mjfOLP2%2BBUqFZP8kAL3xyHtEU0tGdoXoJJ5xSzwNSSJMNpJISMaN1tlSw8I%2FOxHdlstdo8m17Pcgi48hTEaVJxsKmHad3BsXmTH%2BIfwPEJx2L4yQjwnh%2BCtj4NvRdf6WIhixx8WAilysHlAG0Y99Y7JOCoYkJ854TDEUcZ1CVR6IJ41LuzklQKy79b5UoruMzkbJ7svIRwXrhbTLwVK3h3enG7GQDN3QPiyErA7PdtPpkAescgRMFmtAi%2F4%2B7xPFDjc3NwGRkWWaBaTObjL4DAfHhmeGkKPr58J2lxGdr%2F2tmAhQNetBY97KJmeW%2Bb8KQL2OdJlFUw%3D%3D&response-content-disposition=inline%3B+filename%3D%22CP67.zip%22) for noisy microarray images from [here](http://wix.to/2cAQBBA)
Unzip it (CP67.zip) in same folder. It should be now CP67.pth
Use checkoutput.py program to check output on your images.

For training on your images use train2.py. For training, program assumes input image size 512 x 512 which is broken by program into 128 x 128 for training.

There is no limitation of image size on testing/predicting/checking output.
