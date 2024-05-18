import cv2

def recognize_faces(classifier_path):
    # Load the pre-trained classifier
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read(classifier_path)

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()  # Capture frame-by-frame

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use the classifier to detect faces
        #face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        face_cascade = cv2.CascadeClassifier('./Face_cascade.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangles around the detected faces and try to recognize them
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Recognize the face
            predicted_id, _ = clf.predict(roi_gray)

            # Display the recognized person's ID
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = f"Person {predicted_id}"
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)

            # Draw a rectangle around the face
            color = (255, 0, 0)  # BGR color format
            stroke = 2  # Thickness of the rectangle border
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, stroke)

        # Display the frame
        cv2.imshow('Recognizing Faces', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    classifier_path = './classifier.xml'  # Path to your trained classifier
    recognize_faces(classifier_path)


