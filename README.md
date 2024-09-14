# CAL Overview
PyTorch implementation of "**CAL: A Consistency-based Active Learning Strategy for Aerial Object Detection**".

## Installation
To get all the requirements, please run
```
pip install -r requirements.txt
```

## Usage
Take the dataset Visdrone2019 as example.

### YOLOv9

Training:
1. download the Visdrone2019 dataset [BaiDu](https://pan.baidu.com/s/1EE_mSVRuS_gsE4OMjpcUQA?pwd=4sk8)(Code:4sk8) and put it in the `dataset` folder.

2. get the init 500 images dataset from train:
```
python YOLOv9/tools/split_init_dataset.py 
```

3. download the [pre-trained weights](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt) of YOLOv9-c, and put it in the `YOLOv9` folder.

4. modify the `YOLOv9/data/VisDrone.yaml`:
```yaml
train: ../dataset/VisDrone_part/init
val: ../dataset/VisDrone/val
```

5. train the model to get the init weights:

```python
python YOLOv9/train_dual.py --weights YOLOv9/yolov9-c.pt
```
The trained weights can be downloaded from [BaiDu](https://pan.baidu.com/s/1uveTleRDReY85sO0cXxFIw?pwd=hhik)(Code:hhik).

6. sampling using the initial trained model:

```python
python YOLOv9/tools/final_disturbance.py  
```

Evaluation and Inference is same as [YOLOv9](https://github.com/WongKinYiu/yolov9/blob/main/README.md).

### Other Detectors
Similar to YOLOv9, [Faster-RCNN](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/faster_rcnn) and [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR) can be used in the same way.

For Faster-RCNN, The dataset should be formatted as the VOC dataset, `YOLOv9/utils/txt2xml.py` can help to convert the dataset.

For RT-DETRv2, The dataset should be formatted as the COCO dataset, `YOLOv9/utils/txt2json.py` can help to convert the dataset.


## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
* [https://github.com/WZMIAOMIAO/deep-learning-for-image-processing](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)
* [https://github.com/lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)

</details>


