import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml")

if face_cascade.empty() or smile_cascade.empty():
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

    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.1,
                                          minNeighbors=6, minSize=(60,60))

    for(x, y, w, h) in faces:
        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        # The mouth is in the bottom half of the face
        mouth_top = h // 2
        mouth_gray = face_gray[mouth_top:h, :]

        smiles = smile_cascade.detectMultiScale(mouth_gray, scaleFactor=1.7,
                                                minNeighbors=35, minSize=(25,25))

        # Draw the smile boxes (shifted down by mouth_top)
        for(sx, sy, sw, sh) in smiles:
            cv2.rectangle(face_color,
                          (sx, mouth_top + sy),
                          (sx + sw, mouth_top + sy + sh),
                          (0, 255, 255), 2)

        if len(smiles) > 0:
            label = "Smiling :)"
            color = (0, 255, 0)
        else:
            label = "Not smiling"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Smile Detector", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyALlWindows()