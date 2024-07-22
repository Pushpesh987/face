import cv2
import numpy as np

# Check if cv2.face is available
if not hasattr(cv2, 'face'):
    raise Exception("cv2.face module is not available. Ensure you have opencv-contrib-python installed.")

# Load the trained model and label map
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_recognizer.yml')
label_map = np.load('label_map.npy', allow_pickle=True).item()

# Debugging: Print the label map
print(f"Label Map: {label_map}")

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to recognize faces in an image
def recognize_faces(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = gray_image[y:y + h, x:x + w]
        label_id, confidence = face_recognizer.predict(face)
        print(f"Predicted label ID: {label_id}, Confidence: {confidence}")
        if confidence < 100:  # You can adjust the confidence threshold as needed
            custom_id = label_map.get(label_id, "Unknown")
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, custom_id, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return image

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = recognize_faces(frame)
    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
