import cv2
from time import time

def main_app(name, timeout=5):
    face_cascade = cv2.CascadeClassifier('./Face_cascade.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(f"./classifier.xml")
    
    # Read the image from file
    image = cv2.imread("Extracted/new.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        id, confidence = recognizer.predict(roi_gray)
        confidence = 100 - int(confidence)
        print(confidence)
        if confidence > -3:
            text = 'Recognized: ' + name.upper()
            font = cv2.FONT_HERSHEY_PLAIN
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            image = cv2.putText(image, text, (x, y-4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
        else:
            text = "Unknown Face"
            font = cv2.FONT_HERSHEY_PLAIN
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
            image = cv2.putText(image, text, (x, y-4), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.imshow("image", image)
    cv2.waitKey(2000)  # Display the recognized face for 2 seconds

    cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    main_app("Your Name")

