# Run-YOLOv8-in-One-Line-on-Jetson

Do you want to deploy Yolo on Jetson using a single line command？This repo can help you achieve. Considering that red error messages during the environment setup process can potentially affect everyone's mood, we have decided to hide most of the warnings and error messages.

## Getting Start

Before deploying YOLOv8, please ensure that you have a [hardware device](https://www.seeedstudio.com/reComputer-J4011-p-5585.html?queryID=7e0c2522ee08fd79748dfc07645fdd96&objectID=5585&indexName=bazaar_retailer_products) with an installed [operating system](https://wiki.seeedstudio.com/reComputer_J4012_Flash_Jetpack/).

The only command you need to execute is:

```sh
git clone https://github.com/yuyoujiang/Run-YOLOv8-in-One-Line-on-Jetson && python Run-YOLOv8-in-One-Line-on-Jetson/run.py <password>
```

## Arguments Introduction

You can choose between different CV tasks, models, and deployment options like [official documents](https://docs.ultralytics.com/).
Please view the details using the following command:

```sh
python <path to this script>/run.py -h
```

- --password  : This is a positional arguments. A password may be required to install the necessary dependencies.
- --task : The CV task what you want to test. Supported "detect", "classify", "segment" and "pose". Default to detect. Refer to [here](https://docs.ultralytics.com/tasks/) for more information.
- --model : The model name. You can find the name of the model you want to use [here](https://docs.ultralytics.com/models/yolov8/#supported-modes). Default to yolov8n.
- --use_trt : Use tensorTR for inference. If can't find a tensorRT model, create one.
- --use_half : FP16 quantization when export a tensorRT model.
- --source : path to input video or camera id. Refer to [here](https://docs.ultralytics.com/modes/)
