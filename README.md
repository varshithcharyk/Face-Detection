import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer

# Load pre-trained face detectation model
face_cascade = cv2.CascadeClassicfier(cv2.dara.haarcascades + 'haarcascade_frontalface_default.xml')

# Load face recognition model (example with a pre-trained model)
recognition_model = load_model('face_recognition_mode.h5')
normalizer = Normalizer(norm='l2')

# Function to detect faces
def detect_faces(image):
    grey = cv2.cvtcolor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighors=5,minSize=(30,30))
    return faces

# Function to preprocess face images for recognition
def preprocess_face(image, bbox):
    x, y, w, h = bbox
    face = image[y:y+h, x:x+w]
    face = cv2.resize(face, (160,160))  # Resize to model input size
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    return face

# Function to recognize faces
def recognize_face(face):
    embeddings = recognition_model.predict(face)
    normalized_embeddings = normalizer.transform(embeddings)
    return normalized_embeddings

# Main function for face detection and recognition
def main():
    video_capture = cv2.VideoCapture(0)  # Use webcame

    print("Starting face detection and recognition. press 'q' to exit.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Detect faces
        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            # Draw a rectangle around detected faces
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Preprocess the face for recognition
            face = preprocess_face(frame, (x, y, w, h))

            # Recognize the face
            embeddings = recognize_face(face)
            # Placeholder for matching against a database (not implemented)
            label = "Recognized" if embeddings is not None else "Unknown"

            # Display the video frame
            cv2.imshow('Video', frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    video_capture.release()
    cv2.destroyAllWindowa()

if __name__ == "__main__":
    main()  
