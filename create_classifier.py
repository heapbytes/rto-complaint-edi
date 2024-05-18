import numpy as np
import os
import cv2

# Method to train custom classifier to recognize face
def train_classifier(name):
    # Read all the images in the dataset
    dataset_path = './dataset/'

    faces = []
    ids = []

    # Iterate through the dataset directory to read images
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):  # Assuming images are in jpg or png format
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
                if img is not None:
                    # Resize the image to a fixed size (e.g., 100x100)
                    img = cv2.resize(img, (100, 100))
                    faces.append(img)
                    ids.append(int(os.path.splitext(file)[0].split('_')[1]))  # Extract ID from filename

    # Convert lists to numpy arrays
    faces = np.array(faces)
    ids = np.array(ids)

    # Train classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)

    # Save classifier
    classifier_path = f'./classifier.xml'
    clf.write(classifier_path)
    print(f"Classifier saved to: {classifier_path}")

# Example usage:
if __name__ == "__main__":
    train_classifier("custom")


