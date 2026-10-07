# AI Hand Gesture Volume Controller

A real-time, touchless volume controller that uses **Computer Vision and AI-based hand tracking** to control the macOS system volume with hand gestures.

## Features

- Real-time webcam processing
- Hand landmark detection using MediaPipe
- Thumb and index finger tracking
- Fingertip distance calculation
- 0–100% system volume control
- Real-time volume bar and hand landmarks
- macOS system-volume integration
- Automatic MediaPipe model download

## How It Works

```text
Webcam
   ↓
OpenCV
   ↓
MediaPipe Hand Landmarker
   ↓
21 Hand Landmarks
   ↓
Thumb + Index Finger
   ↓
Distance Calculation
   ↓
Distance → Volume (0–100%)
   ↓
macOS System Volume
```

### Gesture Control

| Gesture | Result |
|---|---|
| Thumb and index finger close | Low volume |
| Fingers moderately separated | Medium volume |
| Fingers spread apart | High volume |
| Press `Q` | Exit |

## Technologies

- Python 3.11
- OpenCV
- MediaPipe 0.10.35
- NumPy
- macOS AppleScript (`osascript`)
- Visual Studio Code
- Git and GitHub

## Requirements

- macOS
- Python 3.11
- Webcam
- Internet connection for the first MediaPipe model download

## Project Structure

```text
AI_hand-gesture-volume-controller/
│
├── .gitignore
├── README.md
├── requirements.txt
└── volume_controller.py
```

`hand_landmarker.task` is downloaded automatically by the application and is intentionally excluded from Git.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mnithish11/AI_hand-gesture-volume-controller.git
cd AI_hand-gesture-volume-controller
```

### 2. Check Python

```bash
python3.11 --version
```

### 3. Create the virtual environment

On an Apple Silicon Mac with Homebrew:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
```

### 4. Activate it

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main dependencies are:

```text
mediapipe==0.10.35
opencv-python
numpy
```

## Run the Application

```bash
python volume_controller.py
```

Or:

```bash
./.venv/bin/python volume_controller.py
```

The webcam window will open. Move your hand in front of the camera and change the distance between your thumb and index finger.

## Volume Calculation

The distance between the thumb and index fingertip is calculated using:

```text
d = √((x₂ - x₁)² + (y₂ - y₁)²)
```

The project uses:

```python
MIN_DISTANCE = 30
MAX_DISTANCE = 220
```

and maps the distance to 0–100%:

```python
volume = np.interp(
    distance,
    [MIN_DISTANCE, MAX_DISTANCE],
    [0, 100]
)
```

MediaPipe landmarks used:

```text
Landmark 4 → Thumb fingertip
Landmark 8 → Index fingertip
```

## macOS Volume Control

The application controls the system volume using:

```bash
osascript -e "set volume output volume 50"
```

Python executes this through the `subprocess` module.

## Camera

The current implementation uses:

```python
cap = cv2.VideoCapture(0)
```

If an iPhone Continuity Camera opens instead of the MacBook camera, disable Continuity Camera on the iPhone or modify the camera-selection logic.

## Privacy

Camera frames are processed locally by the application. The hand-tracking model runs locally after it is downloaded. The application does not intentionally upload camera footage to a remote server.

## Limitations

- Primarily designed for macOS.
- Requires a working webcam.
- Poor lighting can reduce detection accuracy.
- Complex backgrounds can affect tracking.
- Current implementation tracks one hand.
- Multiple cameras can cause the wrong camera to be selected.

## Future Enhancements

- Multi-hand gesture recognition
- Gesture-based play/pause and media controls
- Screen brightness control
- Gesture-based mouse control
- Voice + gesture control
- Windows and Linux support
- Smart-home control
- Advanced gesture classification

## Testing

| Test | Expected Result |
|---|---|
| Start application | Camera window opens |
| Show hand | Hand landmarks appear |
| Bring fingers closer | Volume decreases |
| Spread fingers | Volume increases |
| Move hand | Landmarks track movement |
| Press Q | Application exits |

## Learning Outcomes

This project demonstrates practical knowledge of:

- Python programming
- Computer Vision
- OpenCV
- MediaPipe
- Hand landmark detection
- NumPy
- Euclidean distance
- Real-time video processing
- Human-Computer Interaction
- Virtual environments
- Python package management
- macOS automation
- Git and GitHub
- Debugging

## Author

**M Nithish Shanker Goud**

Computer Science Engineering  
Ajeenkya DY Patil University × NIAT

## License

This project is created for educational and demonstration purposes.

## Project Description

> An AI-powered touchless Human-Computer Interaction system that uses real-time hand landmark detection and thumb-index finger distance to control system audio volume.
