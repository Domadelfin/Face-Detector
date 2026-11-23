import cv2

# Load the pre-trained face detector that comes with OpenCV

face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_path)


if face_cascade.empty():
    print("Could not load the face detector")
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open the webcam")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find faces in the grayscale image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=15,
        minSize = (60,60)
    )

    # Draw a box around each face on the COLOR frame
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, f"Faces: {len(faces)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Face Detection", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        print(faces)

cap.release()
cv2.destroyALlWindows()
        