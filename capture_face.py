import cv2
import os
import time

def capture_face_SHORT(face_n):
# Create a directory to store the dataset if it doesn't exist
    dataset_dir = 'dataset'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Initialize the count for captured images
    count = 0

    while count < 50:
        #time.sleep(0.2)
        ret, frame = cap.read()  # Capture frame-by-frame

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangles around the faces and capture images
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Save the captured face to the dataset directory
            count += 1
            face_name = f'{face_n}_{count}.jpg'
            cv2.imwrite(os.path.join(dataset_dir, face_name), frame[y:y+h, x:x+w])
            #print(f"Captured image {face_name}")

    # Capture images until count reaches 100

        # Display the frame
        cv2.imshow('Capturing Faces', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

