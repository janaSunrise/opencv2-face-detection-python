import cv2

from . import HAARCASCADE_CONFIG

# Define the variables
cap = cv2.VideoCapture(0)
classifier = cv2.CascadeClassifier(HAARCASCADE_CONFIG)

if __name__ == "__main__":
    while True:
        # Read the frame
        _, img = cap.read()

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = classifier.detectMultiScale(
            img_gray, scaleFactor=1.1, minNeighbors=4
        )

        # Draw the rectangle around each face
        for x, y, w, h in detections:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Detection", img)

        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xFF

        if k == 27:
            break

    # Release the VideoCapture object
    cap.release()
