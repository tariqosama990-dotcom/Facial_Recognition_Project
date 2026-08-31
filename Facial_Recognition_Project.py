import cv2

#Definition and download of the Haar Cascade file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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


    #convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #Face Detection
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    #Draw a square around the face
    for(x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 2)

     # Display the camera image
    cv2.imshow("Camera Test", frame)
        
    # Press Q to close the camera
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
print("ggggg")
print("ggggg")