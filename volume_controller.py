import cv2
import mediapipe as mp
import math
import numpy as np
import os
import time
import subprocess
import urllib.request


# ==========================================
# 1. DOWNLOAD MEDIAPIPE HAND MODEL
# ==========================================

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task"
)

if not os.path.exists(MODEL_PATH):

    print("Downloading hand tracking model...")

    try:
        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )

        print("Hand model downloaded successfully.")

    except Exception as error:

        print("Could not download model:")
        print(error)

        exit()


# ==========================================
# 2. MEDIAPIPE SETUP
# ==========================================

BaseOptions = mp.tasks.BaseOptions

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.7,

    min_hand_presence_confidence=0.7,

    min_tracking_confidence=0.7
)


# ==========================================
# 3. WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    exit()


# ==========================================
# 4. VOLUME SETTINGS
# ==========================================

MIN_DISTANCE = 30

MAX_DISTANCE = 220

current_volume = 50

last_volume = -1

start_time = time.monotonic()


# ==========================================
# 5. MAC VOLUME CONTROL
# ==========================================

def set_volume(volume):

    global last_volume

    volume = max(
        0,
        min(100, int(volume))
    )

    # Don't repeatedly send the same value
    if volume == last_volume:
        return

    subprocess.run(

        [
            "osascript",
            "-e",
            f"set volume output volume {volume}"
        ],

        stdout=subprocess.DEVNULL,

        stderr=subprocess.DEVNULL
    )

    last_volume = volume


# ==========================================
# 6. DRAW HAND LANDMARKS
# ==========================================

def draw_hand(frame, hand):

    height, width, _ = frame.shape

    points = []


    # Draw 21 landmarks

    for landmark in hand:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        points.append(
            (x, y)
        )

        cv2.circle(

            frame,

            (x, y),

            4,

            (0, 255, 0),

            -1
        )


    # Hand connections

    connections = [

        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (5, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (9, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (13, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (0, 17)
    ]


    for start, end in connections:

        cv2.line(

            frame,

            points[start],

            points[end],

            (0, 255, 0),

            2
        )


# ==========================================
# 7. START MEDIAPIPE
# ==========================================

with HandLandmarker.create_from_options(
    options
) as landmarker:


    # ======================================
    # MAIN LOOP
    # ======================================

    while True:


        # -------------------------------
        # Read webcam
        # -------------------------------

        success, frame = cap.read()


        if not success:

            print(
                "Could not read webcam."
            )

            break


        # Mirror camera

        frame = cv2.flip(
            frame,
            1
        )


        height, width, _ = (
            frame.shape
        )


        # -------------------------------
        # Convert image
        # -------------------------------

        rgb = cv2.cvtColor(

            frame,

            cv2.COLOR_BGR2RGB
        )


        image = mp.Image(

            image_format=(
                mp.ImageFormat.SRGB
            ),

            data=rgb
        )


        # -------------------------------
        # Timestamp
        # -------------------------------

        timestamp = int(

            (
                time.monotonic()
                - start_time
            ) * 1000
        )


        # -------------------------------
        # Detect hand
        # -------------------------------

        result = (
            landmarker.detect_for_video(
                image,
                timestamp
            )
        )


        # =================================
        # IF HAND FOUND
        # =================================

        if result.hand_landmarks:


            hand = (
                result.hand_landmarks[0]
            )


            # Draw landmarks

            draw_hand(
                frame,
                hand
            )


            # -----------------------------
            # Thumb
            # -----------------------------

            thumb = hand[4]


            thumb_x = int(

                thumb.x * width

            )

            thumb_y = int(

                thumb.y * height

            )


            # -----------------------------
            # Index finger
            # -----------------------------

            index = hand[8]


            index_x = int(

                index.x * width

            )

            index_y = int(

                index.y * height

            )


            # -----------------------------
            # Draw fingertips
            # -----------------------------

            cv2.circle(

                frame,

                (thumb_x, thumb_y),

                10,

                (255, 0, 255),

                -1
            )


            cv2.circle(

                frame,

                (index_x, index_y),

                10,

                (255, 0, 255),

                -1
            )


            # -----------------------------
            # Draw line
            # -----------------------------

            cv2.line(

                frame,

                (thumb_x, thumb_y),

                (index_x, index_y),

                (255, 0, 255),

                3
            )


            # =================================
            # CALCULATE DISTANCE
            # =================================

            distance = math.hypot(

                index_x - thumb_x,

                index_y - thumb_y
            )


            # =================================
            # DISTANCE → VOLUME
            # =================================

            volume = np.interp(

                distance,

                [
                    MIN_DISTANCE,
                    MAX_DISTANCE
                ],

                [
                    0,
                    100
                ]
            )


            volume = int(volume)


            current_volume = volume


            # =================================
            # CHANGE MAC VOLUME
            # =================================

            set_volume(
                volume
            )


            # -----------------------------
            # Show distance
            # -----------------------------

            cv2.putText(

                frame,

                f"Distance: {int(distance)} px",

                (30, 90),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (255, 255, 255),

                2
            )


        # =================================
        # DISPLAY VOLUME
        # =================================

        cv2.putText(

            frame,

            f"Volume: {current_volume}%",

            (30, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 0),

            2
        )


        # =================================
        # VOLUME BAR
        # =================================

        bar_x = 30

        bar_y = 130

        bar_width = 350

        bar_height = 30


        # Background

        cv2.rectangle(

            frame,

            (
                bar_x,
                bar_y
            ),

            (
                bar_x + bar_width,
                bar_y + bar_height
            ),

            (50, 50, 50),

            -1
        )


        # Filled volume

        filled = int(

            (
                current_volume
                / 100
            ) * bar_width

        )


        cv2.rectangle(

            frame,

            (
                bar_x,
                bar_y
            ),

            (
                bar_x + filled,
                bar_y + bar_height
            ),

            (0, 255, 0),

            -1
        )


        # Border

        cv2.rectangle(

            frame,

            (
                bar_x,
                bar_y
            ),

            (
                bar_x + bar_width,
                bar_y + bar_height
            ),

            (255, 255, 255),

            2
        )


        # =================================
        # INSTRUCTIONS
        # =================================

        cv2.putText(

            frame,

            "Close fingers = LOW | Spread = HIGH",

            (30, 205),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2
        )


        cv2.putText(

            frame,

            "Press Q to exit",

            (30, 240),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (0, 255, 255),

            2
        )


        # =================================
        # SHOW CAMERA
        # =================================

        cv2.imshow(

            "AI Hand Gesture Volume Controller",

            frame
        )


        # =================================
        # EXIT
        # =================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()

print(
    "Volume Controller stopped."
)