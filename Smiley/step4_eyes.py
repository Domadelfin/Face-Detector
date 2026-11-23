import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")

if face_cascade.empty() or eye_cascade.empty():
    print("Could not load a detector")
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

    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors=5, minSize=(60,60))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop out just the face (the "region of interest")
        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        # Look for eyes only inside the face
        eyes = eye_cascade.detectMultiScale(face_gray[0:h // 2, :], scaleFactor = 1.1,
                                            minNeighbors = 10, minSize=(20, 20))

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh),
                          (255, 0, 0), 2)

        # Show what an emotion model would "see": a tiny 48x48 grayscale face
        tiny = cv2.resize(face_gray, (48, 48))
        preview = cv2.resize(tiny, (240, 240),
                             interpolation=cv2.INTER_NEAREST)
        cv2.imshow("What the model sees", preview)

    cv2.imshow("Face + Eyes", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite("screenshot.png", frame)
        print("Saved screenshot.pnh")

cap.release()
cv2.destroyAllWindows()