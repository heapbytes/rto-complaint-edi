import face_recognition as fr
import os
import cv2
import numpy as np

class FaceRecognizer:
    def __init__(self):
        self.faces = self.get_encoded_faces()

    def get_encoded_faces(self):
        encoded = {}

        for dirpath, dnames, fnames in os.walk("./dataset"):
            for f in fnames:
                if f.endswith(".jpg") or f.endswith(".png"):
                    face = fr.load_image_file("dataset/" + f)
                    # Check if any face is detected in the image
                    face_encodings = fr.face_encodings(face)
                    if len(face_encodings) > 0:
                        encoding = face_encodings[0]
                        encoded[f.split(".")[0]] = encoding

        return encoded

    def classify_face(self, im):
        faces_encoded = list(self.faces.values())
        known_face_names = list(self.faces.keys())

        img = cv2.imread(im, 1)

        face_locations = fr.face_locations(img)
        unknown_face_encodings = fr.face_encodings(img, face_locations)

        face_names = []
        for face_encoding in unknown_face_encodings:
            matches = fr.compare_faces(faces_encoded, face_encoding)
            name = "Unknown"

            face_distances = fr.face_distance(faces_encoded, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

        print("Recognized faces:")
        for name in face_names:
            print(name)

        return face_names

# Example usage:
if __name__ == "__main__":
    recognizer = FaceRecognizer()
    recognizer.classify_face("Extracted/uploaded_image.jpg")






'''
class FaceRecognizer:
    def __init__(self, face_repository_path="./dataset/"):
        self.face_repository_path = face_repository_path
        self.faces_encoded = {}
        self.known_face_names = []
        self.load_encoded_faces()

    def load_encoded_faces(self):
        for dirpath, dnames, fnames in os.walk(self.face_repository_path):
            for f in fnames:
                if f.endswith(".jpg") or f.endswith(".png"):
                    file_path = os.path.join(dirpath, f)
                    try:
                        face = fr.load_image_file(file_path)
                        encoding = fr.face_encodings(face)
                        if len(encoding) > 0:
                            self.faces_encoded[f.split(".")[0]] = encoding[0]
                            self.known_face_names.append(f.split(".")[0])
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")


    def classify_face(self, im):
        print('HIT HERE SUCCESSFULLY')
        img = fr.load_image_file(im)
        face_locations = face_recognition.face_locations(img)
        unknown_face_encodings = face_recognition.face_encodings(img, face_locations)
        face_names = []
        for face_encoding in unknown_face_encodings:
            matches = face_recognition.compare_faces(list(self.faces_encoded.values()), face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(list(self.faces_encoded.values()), face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                # Only assign a name if the match is within a certain tolerance
                if face_distances[best_match_index] < 0.6:  # Adjust this threshold as needed
                    name = self.known_face_names[best_match_index]
            face_names.append(name)
        return face_names


if __name__ == "__main__":

    recognizer = FaceRecognizer()
    print(recognizer.classify_face("./Extracted/uploaded_image.jpg")) 
'''