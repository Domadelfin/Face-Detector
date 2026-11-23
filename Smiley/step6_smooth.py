import cv2
from collections import deque

# Settings to tune
ALPHA = 0.3         # how quickly the box follows new detections (0 to 1)
MAX_MISSED = 10     # frames to keep the box after the face disappears
HISTORY = 18        # how many recent frames vote on "smiling"

def smooth(old, new):
    return ALPHA * new + ( 1 - ALPHA) * old

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

# Memory that lasts between frames
smooth_box = None                       # the smoothed (x, y, w, h) or None
missed = 0                              # frames in a row with now face
smile_history = deque(maxlen=HISTORY)   # recent True/False smile results

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(60,60))

    if len(faces) > 0:
        # Pick the largest face (biggest width * height)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        missed = 0

        # Thin gray box: the raw detection, for comparison
        cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 1)

        if smooth_box is None:
            smooth_box = (x, y, w, h)
        else:
            old_x, old_y, old_w, old_h = smooth_box
            smooth_box = (smooth(old_x, x), smooth(old_y, y),
                          smooth(old_w, w), smooth(old_h, h))

        # Check for a smile in the bottom half of the face
        face_gray = gray[y:y + h, x:x + w]
        mouth_gray = face_gray[ h // 2:h, :]
        smiles = smile_cascade.detectMultiScale(mouth_gray, scaleFactor=1.7,
                                                minNeighbors=35, minSize=(25, 25))
        smile_history.append(len(smiles) > 0)
    else:
        missed += 1
        if missed > MAX_MISSED:
            smooth_box = None
            smile_history.clear()

    if smooth_box is not None:
        bx, by, bw, bh = [int(v) for v in smooth_box]

        smile_ratio = sum(smile_history) / len(smile_history)
        if smile_ratio > 0.5:
            label = f"Smiling ({smile_ratio:.0%})"
            color = (0, 255, 0)
        else:
            label = f"Not smiling ({smile_ratio:.0%})"
            color = (0, 0, 255)

        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2)
        cv2.putText(frame, label, (bx, max(by - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0/7, color, 2)

    cv2.imshow("Smoothed Smile Detector", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()