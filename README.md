# Labelme2YOLO
This repo is for personal purposes. Feel free to use it if you need it. FYI, this is a mixture of pylabel and labelme2coco for reference.

## Instruction 
Follow the steps to convert your labelme data either to coco data or yolov5 data

### 1. Download the git
``` python
  git https://github.com/IanMin1022/Labelme2YOLO.git
```

### 2. From the git directory, use pip to install prerequisites
``` python
  pip install -r requirements.txt
```

### 3. Execute labelme2coco.py or labelme2yolo.py to convert your data
Arguments are as followed
``` python
  input_dir: path to input data (image/JSON data)
  output_dir: path for processed data
  train_rate: train and validation data rate (if train_rate = 0.85, train data is 85%)
```
#### For coco data
``` python
  python3 labelme2coco.py --input_dir=/content/dataset --output_dir /content/dataset
```

#### For yolov5 data
``` python
  python3 labelme2yolo.py --input_dir=/content/dataset/ --output_dir /content/dataset
```
