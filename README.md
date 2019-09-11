# UDT_pytorch
This repository contains a Python *reimplementation* of the **Unsupervised Deep Tracking** 

Ning Wang, Yibing Song, Chao Ma, Wengang Zhou, Wei Liu, and Houqiang Li 

to appear in *CVPR 2019*

### Acknowledges

The results of this implementation may be slightly different from the original [UDT](http://github.com/594422814/UDT). The results in our paper are obtained using the MatconvNet implementation.

Our baseline method is DCFNet and many parts of the code are from [DCFNet_pytorch](https://github.com/foolwood/DCFNet_pytorch). For more details, you can refer to [DCFNet_pytorch](https://github.com/foolwood/DCFNet_pytorch). The main differences are (1) unsupervised data preprocessing (please check the ```train/dataset/``` folder); (2) our training code utilizes forward tracking and backward verification to train the model (please check the ```train/train_UDT.py``` file). (3) This implementation is a simplified version, and I will update it later.

### Requirements

Requirements for **PyTorch 0.4.0** and opencv-python

```shell
conda install pytorch torchvision -c pytorch
conda install -c menpo opencv
```

Training data (VID) and Test dataset (OTB).

### Test

```shell
cd UDT_pytorch/track 
ln -s /path/to/your/OTB2015 ./dataset/OTB2015
ln -s ./dataset/OTB2015 ./dataset/OTB2013
cd dataset & python gen_otb2013.py
python UDT.py --model ../train/work/checkpoint.pth.tar
```

### Train

1. Download training data. ([**ILSVRC2015 VID**](http://bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php#vid)) 

2. Prepare training data for `dataloader`.

   ```shell
   cd UDT_pytorch/train/dataset
   python parse_vid.py <VID_path>  # save all vid info in a single json
   python crop_image.py  # crop and generate a json for dataloader
   ```

3. Training. (on multiple ***GPUs*** :zap: :zap: :zap: :zap:)

   ```
   cd UDT_pytorch/train/
   CUDA_VISIBLE_DEVICES=0,1,2,3 python train_UDT.py
   ```

### Fine-tune hyper-parameter

1. After training, you can simple test the model with default parameter.

   ```shell
   cd UDT_pytorch/track/
   python UDT.py --model ../train/work/crop_125_2.0/checkpoint.pth.tar
   ```

2. Search a better hyper-parameter.

   ```shell
   CUDA_VISIBLE_DEVICES=0 python tune_otb.py  # run on parallel to speed up searching
   python eval_otb.py OTB2013 * 0 10000
   ```

### License
Licensed under an MIT license.

### Citation
If you find this work useful for your research, please consider citing our work and DCFNet:
```
@inproceedings{Wang_2019_Unsupervised,
    title={Unsupervised Deep Tracking},
    author={Wang, Ning and Song, Yibing and Ma, Chao and Zhou, Wengang and Liu, Wei and Li, Houqiang},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2019}
}

@article{wang17dcfnet,
    Author = {Qiang Wang, Jin Gao, Junliang Xing, Mengdan Zhang, Weiming Hu},
    Title = {DCFNet: Discriminant Correlation Filters Network for Visual Tracking},
    Journal = {arXiv preprint arXiv:1704.04057},
    Year = {2017}
}
```

### Contact
If you have any questions, please feel free to contact wn6149@mail.ustc.edu.cn



