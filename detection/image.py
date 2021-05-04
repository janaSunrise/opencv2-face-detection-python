import cv2

from . import IMAGE_PATH_PREFIX, HAARCASCADE_CONFIG

# Define the variables
IMAGE_NAME = "000.jpg"

classifier = cv2.CascadeClassifier(HAARCASCADE_CONFIG)

if __name__ == "__main__":
    # Get the image
    img = cv2.imread(IMAGE_PATH_PREFIX + IMAGE_NAME)

    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detections = classifier.detectMultiScale(img_grayscale, scaleFactor=1.1, minNeighbors=6)

    # Get the face co-ordinates
    print(f"Co-ordinates:\n{detections}")

    # Now let's plot the face co-ordinates
    for x, y, w, h in detections:
        cv2.rectangle(img, (x, y), (x + w, y + h), (64, 224, 208), 2)

    # Time to show the Image with detection zones.
    cv2.imshow("Face detection", img)
    cv2.waitKey()
