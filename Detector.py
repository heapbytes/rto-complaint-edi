import cv2
from time import time

def main_app(name, timeout=5):
    face_cascade = cv2.CascadeClassifier('./Face_cascade.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(f"./classifier.xml")
    cap = cv2.VideoCapture(0)

    start_time = time()
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id, confidence = recognizer.predict(roi_gray)
            confidence = 100 - int(confidence)
            print('\n\n\nCOFIDENCE : ', confidence)
            if confidence > 30:
                text = 'Recognized: ' + name.upper()
                font = cv2.FONT_HERSHEY_PLAIN
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow("image", frame)
                cv2.waitKey(10000)  # Display the recognized face for 2 seconds
                cap.release()
                cv2.destroyAllWindows()
                return
            else:
                text = "Unknown face"
                font = cv2.FONT_HERSHEY_PLAIN
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("image", frame)

        elapsed_time = time() - start_time
        if elapsed_time >= timeout:
            break

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    main_app("Your Name")


      
