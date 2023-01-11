from flask import Flask, Response, render_template
from imutils.video import VideoStream
from fastai.vision.all import *
import threading
import imutils
import cv2
import time

output_frame = None
pred = 'neutral'
lock = threading.Lock()

app = Flask(__name__)

learn = load_learner('convnet_vision_model.pkl')
vs = VideoStream(src=0).start()
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 

def detect_expression():
    global vs, output_frame, lock, pred

    n_frames = 20
    frame_count = 0

    while True:
        frame = vs.read() # read video data
        frame = imutils.resize(frame, width=600) # resive window
        frame = cv2.flip(frame, 1) # flip the frame horizontally

        # convert to greyscale for model
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coord = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

        # prevents error if no face detected
        for coord in face_coord:
            # extract co-ordinates from face detector   
            X, Y, w, h = coord
            H, W, _ = frame.shape
            X_1, X_2 = (max(0, X - int(w * 0.15)), min(X + int(1.15 * w), W))
            Y_1, Y_2 = (max(0, Y - int(h * 0.15)), min(Y + int(1.15 * h), H))
            img_cp = gray[Y_1:Y_2, X_1:X_2].copy()

            # model prediction - no transforms required
            if not frame_count % n_frames:
                pred, idx, probability = learn.predict(img_cp)

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
            cv2.putText(
                frame, pred, 
                (X_1 + padding, Y_1 - padding), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 0, 0), 1)
            
            # render box around face
            cv2.rectangle(frame, (X_1, Y_1), (X_2, Y_2), (0, 255, 200), 2)

        frame_count += 1
        frame_count = frame_count % n_frames

        with lock:
            output_frame = frame.copy()

def generate_img():
    global output_frame, lock

    while True:
        # wait until lock is acquired
        with lock:
            # check if output frame is available
            if not output_frame.size:
                # bug CHECK THIS ####
                continue
            
            # encode output frame to .jpg
            flag, encoded_img = cv2.imencode('.jpg', output_frame)

            # ensure frame was successfully encoded
            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encoded_img) + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/camera_feed')
def camera_feed():
    return Response(generate_img(), 
    mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    t = threading.Thread(target=detect_expression)
    t.daemon = True
    t.start()

    app.run(debug=True, threaded=True, use_reloader=False, port=8000)

vs.stop()