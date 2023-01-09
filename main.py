from fastai.vision import *
import imutils
from imutils.video import VideoStream
import cv2
import time


print('Starting video stream...')

# video stream from front facing camera (src=0)
vs = VideoStream(src=0).start()

# detects frontal faces in frame
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 

time.sleep(1)

flip = True

# while loop to stream video
while True:
    # read video data  
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to greyscale for model
    face_coord = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    for coords in face_coord:     
        X, Y, w, h = coords        
        H, W, _ = frame.shape        
        X_1, X_2 = (max(0, X - int(w * 0.3)), min(X + int(1.3 * w), W))        
        Y_1, Y_2 = (max(0, Y - int(0.3 * h)), min(Y + int(1.3 * h), H))        
        img_cp = gray[Y_1:Y_2, X_1:X_2].copy()        
        
        cv2.rectangle(                
            img=frame,                
            pt1=(X_1, Y_1),                
            pt2=(X_2, Y_2),                
            color=(128, 128, 0),                
            thickness=2,            
        )

        # prediction, idx, probability = learn.predict(Image(pil2tensor(img_cp, np.float32).div_(225)))     
        # cv2.putText(frame, str(prediction), (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (225, 255, 255), 2)

    # flip the frame horizontally
    if flip:
        frame = cv2.flip(frame, 1)
    
    # display frame on window
    cv2.imshow("frame", frame)

    # quit program
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# terminate window
vs.stop()
cv2.destroyAllWindows()