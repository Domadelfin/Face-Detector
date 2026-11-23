import cv2
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open the webcam")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)

    # Creates a grayscale copy of the frame 
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Draw a rectangle
    cv2.rectangle(frame, (200,200), (440, 450), (0, 0, 255), 2)

    cv2.rectangle(grey, (200, 200), (440, 450), (255, 0, 0), 2)

    # Draws a filled red circle at the center of the frame
    cv2.circle(frame, (320, 240), 5, (0, 0, 255), -1)

    #Writes text in the top-left corner
    cv2.putText(frame, "Hello!", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Color", frame)
    cv2.imshow("Gray", grey)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    if key == ord('p'):
        print("color:", frame.shape, " grey: ", grey.shape, grey[0,0])

cap.release()
cv2.destroyAllWindows()