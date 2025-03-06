import cv2
import numpy as np
from keras.models import load_model
from mtcnn import MTCNN
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time

detector = MTCNN()

model = load_model('face_recognition_model.h5')


class_names = ["mohan", "raghu", "unknown"]  


CONFIDENCE_THRESHOLD = 0.4  


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()
cap.set(3, 640)
cap.set(4, 480)

font = cv2.FONT_HERSHEY_COMPLEX

while True:
    success, frame = cap.read()
    if not success:
        continue

    faces = detector.detect_faces(frame)

    for face in faces:
        x, y, w, h = face['box']

        face_img = frame[y:y + h, x:x + w]

        face_img = cv2.resize(face_img, (224, 224))
        face_img = np.expand_dims(face_img, axis=0)
        face_img = preprocess_input(face_img)

        prediction = model.predict(face_img)
        classIndex = np.argmax(prediction)
        probabilityValue = np.max(prediction)

        if probabilityValue < CONFIDENCE_THRESHOLD:
            label = "Unknown"
        else:
            label = class_names[classIndex]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({round(probabilityValue * 100, 2)}%)",
                    (x, y - 10), font, 0.75, (255, 255, 255), 2)
        
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
time.sleep(2)