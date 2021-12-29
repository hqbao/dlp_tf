# EYEQ CHALLENGE RESPONSE

[TOC]

## Solution

- Apply Faster R-CNN with some improvements in Libra R-CNN for object detection
- Find people moving routes from detected heads, and define an entrance box to determine people get in/out by following clauses:
  - If a route start point is inside entrance box and end point is outside entrance box, that is an IN
  - If a route start point is outside entrance box and end point is inside entrance box, that is an OUT

**Result**

- Demo video:  [result.mp4](https://drive.google.com/file/d/1Xpr8OBNeAaEmjrG4pbudd_trXkoOBlJJ/view?usp=sharing)
- Object detection average precision: `86%`

## Setup

1. Clone this repository
2. Install dependencies (make sure Python 3.7 installed)

```
pip install -r requirements.txt
```

3. Prepare datasets & assets
   - annotations: [instances_eyeq.json](https://drive.google.com/file/d/1IwPJlvhA1IT3bUIm6iTKhcfVeJT1J0KW/view?usp=sharing)
   - images: [eyeq](https://drive.google.com/open?id=1J5fsS6v36YaNvCS39SPDhXm5sOa7dpYm)
   - [test.mp4](https://drive.google.com/file/d/1oTtuYGKTIYvX7NXmgkSs54Fiu9Vux5IC/view?usp=sharing)
   - [result.mp4](https://drive.google.com/file/d/1Xpr8OBNeAaEmjrG4pbudd_trXkoOBlJJ/view?usp=sharing)
   - [rpn_weights.h5](https://drive.google.com/file/d/1-k1BS6dXjxLKEc2MTDH2_ylOSigE0N7j/view?usp=sharing) (pre-trained model weights for inference)
   - [rpn_model.tflite](https://drive.google.com/file/d/1m4Ki63_XVFlBz12M2Woca_N0hjnXuyRf/view?usp=sharing) (pre-trained model for inference)
4. Make sure that the datasets along with the project folder as below

```
|- your_dir/
    |- maskrcnn
    |   |-output
    |       |- rpn_weights.h5
    |       |- rpn_model.tflite
    |- datasets
    |   |- coco
    |       |- annotations
    |           |- instances_eyeq.json
    |       |- images
    |           |- eyeq
    |- test.mp4
    |- result.mp4
```



## Get started

### 1. Get your datasets well with visualisation tools    

#### 1.1 Anchors

To see distribution of defined anchors

Run:

```bash
python3 visualize_anchors.py
```

To discover more, you can modify variables below:

- `asizes` (line 14): anchor sizes
- `all_anchor_points` (line 18): if true, all the anchors will be shown, else some of the anchors which are defined in `anchor_points` variable (line 19) will be shown.

#### 1.2 Labeled data

To see images with annotations

Run:

```bash
python3 visualize_coco_datasets.py
```

#### 1.3 Data generation (for RPN, non FPN)

To see selected foreground anchors and  selected background anchors.

Run:

```
python3 visualize_datagen_rpn_non_fpn.py
```

#### 1.4 Visualize some ResNet feature maps

Run:

```bash
python3 visualize_c2c3c4c5.py
```

### 2. Train & validation

#### 2.1 Get settings ready

Check `settings.py` and modify suitable parameters after getting your datasets well with visualization tools (For this challenge, just leave it like so).

#### 2.2 Train RPN (non FPN)

Run:

```bash
python3 train_rpn_non_fpn.py
```

(See loss history in logs file under `output/` folder)

#### 2.3 Precision

Run:

```bash
python3 presision_rpn_non_fpn.py
```

### 3. Inference

#### 3.1 RPN inference (non FPN)

Detect objects on images

 Run:

```bash
python3 inference_rpn_non_fpn.py
```

#### 3.2 Convert model to tensorflow lite format

Run: 

```bash
python3 tflite_convert.py
```

#### 3.3 Test

Place the test video in same root folder with maskrcnn folder

Run:

```bash
python3 test_rpn_non_fpn.py
```

After runing this, You will see a video `result.mp4`