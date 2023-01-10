from fastai.vision.all import *
from imutils.video import VideoStream
import imutils
import cv2
import time

learn = load_learner('convnet_vision_model.pkl')
print('Model loaded successfully')

# video stream from front facing camera (src=0)
vs = VideoStream(src=0).start()
print('Video stream started successfully')

# detect frontal faces in frame
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 

time.sleep(2) # vid stream warm-up

# stream video loop
while True:
    # read video data  
    frame = vs.read()
    frame = imutils.resize(frame, width=600) # resive window
    frame = cv2.flip(frame, 1) # flip the frame horizontally

    # convert to greyscale for model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coord = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))[0]

    # extract co-ordinates from face detector   
    X, Y, w, h = face_coord
    H, W, _ = frame.shape
    X_1, X_2 = (max(0, X - int(w * 0.15)), min(X + int(1.15 * w), W))
    Y_1, Y_2 = (max(0, Y - int(h * 0.15)), min(Y + int(1.15 * h), H))
    img_cp = gray[Y_1:Y_2, X_1:X_2].copy()

    # model prediction - no transforms required
    pred, idx, probability = learn.predict(img_cp)
    # print(pred, probability)

    # determine size of prediction label
    padding = 5
    text_size, _ = cv2.getTextSize(pred, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    text_w, text_h = text_size

    # render background for prediction label
    cv2.rectangle(
        frame,
        (X_1, Y_1), 
        (X_1 + text_w + padding * 2, Y_1 - text_h - padding * 2), 
        (0, 255, 200), -1)

    # render prediction label
    cv2.putText(frame, pred, (X_1 + padding, Y_1 - padding), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # render box around face
    cv2.rectangle(frame, (X_1, Y_1), (X_2, Y_2), (0, 255, 200), 2)

    # display frame on window
    cv2.imshow("video stream", frame)

    # quit program
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# terminate window
vs.stop()
cv2.destroyAllWindows()