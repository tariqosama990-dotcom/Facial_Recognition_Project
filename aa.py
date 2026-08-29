import cv2

# Open the default camera
camera = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not camera.isOpened():
    print("Cannot access camera")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = camera.read()

    if not ret:
        print("Cannot read frame")
        break

    # Display the camera image
    cv2.imshow("Camera Test", frame)

    # Press Q to close the camera
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()