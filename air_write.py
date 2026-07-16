import cv2
import mediapipe as mp
import numpy as np
import os
import time

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None
prev_point = None

# Colors
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
BLACK = (0, 0, 0)

draw_color = GREEN
eraser = False

os.makedirs("saved_notes", exist_ok=True)

while True:
    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    # ---------- Buttons ----------
    cv2.rectangle(frame, (0, 0), (80, 50), RED, -1)
    cv2.rectangle(frame, (90, 0), (170, 50), GREEN, -1)
    cv2.rectangle(frame, (180, 0), (260, 50), BLUE, -1)
    cv2.rectangle(frame, (270, 0), (350, 50), (255, 255, 255), -1)
    cv2.putText(frame, "ERASE", (360, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)

    if result.multi_hand_landmarks:

        hand = result.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        h, w, _ = frame.shape

        index = hand.landmark[8]
        thumb = hand.landmark[4]

        ix = int(index.x*w)
        iy = int(index.y*h)

        tx = int(thumb.x*w)
        ty = int(thumb.y*h)

        cv2.circle(frame,(ix,iy),8,(0,255,255),-1)

        distance = np.sqrt((ix-tx)**2+(iy-ty)**2)

        # -------- Color Buttons --------
        if iy < 50:

            if ix < 80:
                draw_color = RED
                eraser=False

            elif 90 < ix <170:
                draw_color = GREEN
                eraser=False

            elif 180 < ix <260:
                draw_color = BLUE
                eraser=False

            elif 270 < ix <350:
                draw_color = BLACK
                eraser=False

            elif ix>360:
                eraser=True

            prev_point=None

        else:

            if distance<40:

                if prev_point is None:
                    prev_point=(ix,iy)

                color = BLACK if eraser else draw_color

                thickness = 30 if eraser else 5

                cv2.line(canvas,
                         prev_point,
                         (ix,iy),
                         color,
                         thickness)

                prev_point=(ix,iy)

            else:
                prev_point=None

    else:
        prev_point=None

    output=cv2.add(frame,canvas)

    cv2.putText(
        output,
        "Pinch to Draw | S=Save | C=Clear | Q=Quit",
        (10,470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,255,255),
        2
    )

    cv2.imshow("Air Writing AI",output)

    key=cv2.waitKey(1)&0xFF

    if key==ord('c'):
        canvas=np.zeros_like(frame)

    elif key==ord('s'):
        filename=f"saved_notes/note_{int(time.time())}.png"
        cv2.imwrite(filename,canvas)
        print("Saved:",filename)

    elif key==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()