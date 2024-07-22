import cv2
import os
import numpy as np

# Check if cv2.face is available
if not hasattr(cv2, 'face'):
    raise Exception("cv2.face module is not available. Ensure you have opencv-contrib-python installed.")

# Path to the dataset
dataset_path = 'dataset/'

# Initialize face recognizer with tuned parameters
face_recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Custom IDs mapping
custom_id_map = {
    "person1": "4sf21cd022",
    "person3": "4sf21ec003",
    # Add more mappings as needed
}

# Function to read images and labels from the dataset
def get_images_and_labels(dataset_path):
    image_paths = []
    labels = []
    label_map = {}
    id_to_label = {}  # Map numeric labels to custom IDs

    # Iterate through each person's folder
    for label_id, person_name in enumerate(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue
        custom_id = custom_id_map.get(person_name)
        if custom_id is None:
            continue
        label_map[label_id] = custom_id  # Map label id to custom ID
        id_to_label[custom_id] = label_id
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image_paths.append(image_path)
            labels.append(label_id)
    
    return image_paths, labels, label_map

# Get image paths and labels
image_paths, labels, label_map = get_images_and_labels(dataset_path)

# Debugging: Print the image paths and labels
print(f"Image Paths: {image_paths}")
print(f"Labels: {labels}")

# List to hold the images and corrected labels
images = []
corrected_labels = []
for image_path, label in zip(image_paths, labels):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = img[y:y + h, x:x + w]
        images.append(face)
        corrected_labels.append(label)
        # Debugging: Display the detected face
        cv2.imshow("Detected Face", face)
        cv2.waitKey(100)  # Display each face for 100ms

# Convert labels to numpy array
corrected_labels = np.array(corrected_labels)

# Debugging: Print the number of images and labels
print(f"Number of images: {len(images)}")
print(f"Number of labels: {len(corrected_labels)}")

# Train the model
face_recognizer.train(images, corrected_labels)

# Save the model and the label map
face_recognizer.save('face_recognizer.yml')
np.save('label_map.npy', label_map)

print("Model training completed and saved.")
cv2.destroyAllWindows()
