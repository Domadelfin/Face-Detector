import cv2

#Opens the default webcam (camera number 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open the webcam")
    exit()

while True:
    success, frame = cap.read()

    if not success:
        print("Could not read a frame")
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('p'):
        print(frame.shape, frame[0,0])

cap.release()
cv2.destroyAllWindows()