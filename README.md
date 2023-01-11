# Live Video Facial Expression Detection

Using `fast.ai`, an Image Classifier was trained with facial expression data from kaggle. The convnet architecture from `timm` was used to provide optimised training speed and model performance. Initial model accuracy was extremely poor and attempts at improving this by tuning the learning rate were semi-successfully, however, the issue was later realised to be caused by poorly labelled data. The model achieved a success rate of 70% although this poor accuracy did not affect its performance in the final application.

`opencv` and `imutils` were used to stream video from a computer web cam and pass the image tensor into the trained model for prediction. The prediction was then rendered over the video frame alongside a bounding box of the face and displayed in a window to the user.

## Dependencies

- python 3.x

## Getting Started

1. Clone the repo:

```bash
git clone https://github.com/finnformic/video-facial-recognition.git
```

2. Change to working directory and setup python environment:

```bash
cd /foo/.../video-facial-recognition/
python3 -m venv venv
source venv/bin/activate
```

3. Install requirements:

```bash
pip3 install -r requirements.txt
```

4. Run application:

```bash
python3 main.py
```

## Features

- Detects seven facial expressions (neutral, happy, sad, disgust, fear, surprise, angry)
- Frame skipping implemented to speed up video

### Future features

- Change bounding box colour depending on expression
  - Happy (green)
  - Sad (blue)
  - Surprise (yellow)
  - Disgust (purple)
  - Fearful (orange)
  - Neutral (white)
  - Angry (red)
- Implement Flask app to deploy to the browser
