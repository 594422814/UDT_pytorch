# UDT_pytorch
This repository contains a Python *reimplementation* of the **Unsupervised Deep Tracking** 

## Requirements

Requirements for **PyTorch 0.4.0** and opencv-python

```shell
conda install pytorch torchvision -c pytorch
conda install -c menpo opencv
```

Training data (VID) and Test dataset (OTB).

## Test

```shell
cd UDT_pytorch/track 
ln -s /path/to/your/OTB2015 ./dataset/OTB2015
ln -s ./dataset/OTB2015 ./dataset/OTB2013
cd dataset & python gen_otb2013.py
python UDT.py
```

## Train

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

## Fine-tune hyper-parameter

1. After training, you can simple test the model with default parameter.

   ```shell
   cd UDT_pytorch/track/
   python UDT --model ../train/work/crop_125_2.0/checkpoint.pth.tar
   ```

2. Search a better hyper-parameter.

   ```shell
   CUDA_VISIBLE_DEVICES=0 python tune_otb.py  # run on parallel to speed up searching
   python eval_otb.py OTB2013 * 0 10000
   ```

## Citing UDT and DCFNet



