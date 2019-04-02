# UDT_pytorch
This repository contains a Python *reimplementation* of the **Unsupervised Deep Tracking** 

## Requirements

```shell
git clone --depth=1 https://github.com/foolwood/DCFNet_pytorch
```

Requirements for **PyTorch 0.4.0** and opencv-python

```shell
conda install pytorch torchvision -c pytorch
conda install -c menpo opencv
```

Training data (VID) and Test dataset (OTB).

## Test

```shell
cd DCFNet_pytorch/track 
ln -s /path/to/your/OTB2015 ./dataset/OTB2015
ln -s ./dataset/OTB2015 ./dataset/OTB2013
cd dataset & python gen_otb2013.py
python DCFNet.py
```

## Train

1. Download training data. ([**ILSVRC2015 VID**](http://bvisionweb1.cs.unc.edu/ilsvrc2015/download-videos-3j16.php#vid)) 

   ```
   ./ILSVRC2015
   鈹溾攢鈹€ Annotations
   鈹偮犅?鈹斺攢鈹€ VID鈹溾攢鈹€ a -> ./ILSVRC2015_VID_train_0000
   鈹?         鈹溾攢鈹€ b -> ./ILSVRC2015_VID_train_0001
   鈹?         鈹溾攢鈹€ c -> ./ILSVRC2015_VID_train_0002
   鈹?         鈹溾攢鈹€ d -> ./ILSVRC2015_VID_train_0003
   鈹?         鈹溾攢鈹€ e -> ./val
   鈹?         鈹溾攢鈹€ ILSVRC2015_VID_train_0000
   鈹?         鈹溾攢鈹€ ILSVRC2015_VID_train_0001
   鈹?         鈹溾攢鈹€ ILSVRC2015_VID_train_0002
   鈹?         鈹溾攢鈹€ ILSVRC2015_VID_train_0003
   鈹?         鈹斺攢鈹€ val
   鈹溾攢鈹€ Data
   鈹偮犅?鈹斺攢鈹€ VID...........same as Annotations
   鈹斺攢鈹€ ImageSets
       鈹斺攢鈹€ VID
   ```

2. Prepare training data for `dataloader`.

   ```shell
   cd DCFNet_pytorch/train/dataset
   python parse_vid.py <VID_path>  # save all vid info in a single json
   python gen_snippet.py  # generate snippets
   python crop_image.py  # crop and generate a json for dataloader
   ```

3. Training. (on multiple ***GPUs*** :zap: :zap: :zap: :zap:)

   ```
   cd DCFNet_pytorch/train/
   CUDA_VISIBLE_DEVICES=0,1,2,3 python train_DCFNet.py
   ```


## Fine-tune hyper-parameter

1. After training, you can simple test the model with default parameter.

   ```shell
   cd DCFNet_pytorch/track/
   python DCFNet --model ../train/work/crop_125_2.0/checkpoint.pth.tar
   ```

2. Search a better hyper-parameter.

   ```shell
   CUDA_VISIBLE_DEVICES=0 python tune_otb.py  # run on parallel to speed up searching
   python eval_otb.py OTB2013 * 0 10000
   ```

## Citing DCFNet

If you find [**DCFNet**](https://arxiv.org/pdf/1704.04057.pdf) useful in your research, please consider citing:

```
@article{wang2017dcfnet,
  title={DCFNet: Discriminant Correlation Filters Network for Visual Tracking},
  author={Wang, Qiang and Gao, Jin and Xing, Junliang and Zhang, Mengdan and Hu, Weiming},
  journal={arXiv preprint arXiv:1704.04057},
  year={2017}
}
```
