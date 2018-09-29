
train models:
1. train_SegNet_standard  从网上下的SegNet模型, validation loss: 0.00619 
2. * train_UNet_standard 从网上下的UNet，用valid训练，结果崩了，一个模型都没有保存到
3. * train_UNet_same 跟上一个模型一样，用的same训练，还是崩了，但是保存到了几个模型，看validation loss也还行，但是最后还是崩，估计有问题
4. train_SegUNet_old_version 用的以前自己写的SegUNet的模型， 有5次maxpool，很好使, validation loss: 0.00503
5. train_SegUNet_old_version_randomcolor 跟上一个模型一样，用了random color，效果貌似没有上一个好, validation loss: 0.00529
6. train_SegUNet_new_version 以前的模型第5个downsample后直接用的upsample，这不合理。所以直接把第5个downsample删了。一共4个downsample。validation loss: 0.00497
7. train_SegUNet_old_version_coord_at_middle coord加在old version的当中。validation loss: 0.00497
8. train_SegUNet_new_version_coord_at_middle coord加在new version的当中。validation loss: 0.00495
9. train_UNet_pix2pix 用的pix2pix的UNet模型，速度很快，但是效果不好，可能是因为层数少了。validation loss: 0.13075
