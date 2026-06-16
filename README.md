# NAO_team
# NAO Line Follower — Carrera Individual

Autonomous line-following system for the **NAO V4** robot, built for the RoboCup competition. The robot navigates a lane defined by two white lines on a dark surface, detects a horizontal finish line, and is controlled entirely via the robot's head touch sensors.

## Architecture

All processing runs **onboard the NAO** (Intel Atom Z530). A separate debug viewer runs on a laptop over Ethernet/WiFi during development.

```
[ NAO V4 ]
  carrera.py (Python 2.7)
  ├── Bottom camera → grayscale threshold → column scan → lane center
  ├── PID controller → angular velocity correction
  ├── Finish line detection (horizontal white line)
  └── TCP socket → compressed JPEG frames
          │
          │ Ethernet / WiFi
          ▼
[ Laptop ]
  debug_viewer.py (Python 3)
  └── Annotated live feed (cv2.imshow)
```

## Files

| File | Runs on | Description |
|---|---|---|
| `carrera.py` | NAO (Python 2.7) | Main program: vision, PID, locomotion |
| `debug_viewer.py` | Laptop (Python 3) | Live annotated camera feed |

## Requirements

**NAO:** Python 2.7, OpenCV 2.4.x, NumPy — all pre-installed on NAOqi OS.

**Laptop:** Python 3, OpenCV (`pip install opencv-python`), NumPy.

## Usage

```bash
# Upload to the NAO
scp carrera.py nao@<NAO_IP>:/home/nao/

# Run on the NAO
ssh nao@<NAO_IP>
python carrera.py

# Run the debug viewer 
python3 debug_viewer.py <NAO_IP>
```

Press the **front head button** to start. Press it again to stop.

## Configuration

All tunable parameters are grouped at the top of `carrera.py`:

| Parameter | Description |
|---|---|
| `THRESH` | White line detection threshold |
| `ROI_TOP` | Fraction of frame used for lane detection |
| `KP / KI / KD` | PID gains |
| `FWD_SPEED` | Maximum forward speed |
| `FWD_START / RAMP_TIME` | Startup velocity ramp |
| `META_THRESH` | Finish line brightness threshold |
| `META_MIN_ROW_FILL` | Minimum horizontal pixel span to detect finish line |
| `DEBUG_ENABLED` | Set to `False` to free CPU |

