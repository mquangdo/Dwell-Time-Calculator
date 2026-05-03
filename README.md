# Dwell-Time-Calculator
https://github.com/user-attachments/assets/23c86802-0a58-41d1-b143-caa92f9c4022

## Overview

Dwell-Time-Calculator is a computer vision tool that tracks how long objects (vehicles, people, etc.) spend inside user-defined polygon zones in a video stream. It uses [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection, [ByteTrack](https://github.com/ifzhang/ByteTrack) for multi-object tracking (via the `supervision` library), and overlays per-object dwell-time labels directly onto the output video.

## Features

- **Polygon zone support** – define one or more zones via a JSON config file
- **Two timer modes** – `ClockBasedTimer` (wall-clock time) and `FPSBasedTimer` (frame-count based)
- **Annotated video output** – colored bounding boxes and `MM:SS` dwell-time labels per object
- **Live FPS display** – real-time frame rate shown on each output frame
- **Zone drawing tool** – interactive utility to draw and save zone polygons on a video frame
- **YouTube downloader** – helper script to fetch source videos from YouTube

## Project Structure

```
Dwell-Time-Calculator/
├── src/
│   └── run.py               # Main entry point – detection, tracking, annotation
├── utils/
│   ├── general.py           # Zone config loader, frame generator, helper functions
│   └── timers.py            # FPSBasedTimer and ClockBasedTimer implementations
├── tools/
│   ├── draw_zones.py        # Interactive tool to draw polygon zones on a video frame
│   └── download_from_yt.py  # Download a YouTube video via pytube
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:

```
opencv-python
supervision>=0.20.0
ultralytics<=8.3.40
pytube
```

Install them with:

```bash
pip install -r requirements.txt
```

## Configuration

Before running, create a `zones_config.json` file that defines the polygon zones. Each zone is a list of `[x, y]` coordinates:

```json
[
  [[100, 200], [400, 200], [400, 500], [100, 500]],
  [[500, 200], [800, 200], [800, 500], [500, 500]]
]
```

You can use the interactive zone-drawing tool to create this file:

```bash
python tools/draw_zones.py
```

- **Left-click** to add a point to the current polygon
- Press **Enter** to finish the current polygon and start a new one
- Press **S** to save zones to `zones_config.json`
- Press **Escape** or **Q** to quit

## Usage

Edit the `source_video_path` and `target_video_path` variables in `src/run.py`, then run:

```bash
python src/run.py
```

Key parameters in the `main()` call:

| Parameter          | Description                                      | Example          |
|--------------------|--------------------------------------------------|------------------|
| `weights`          | YOLO model weights file                          | `yolov8m.pt`     |
| `device`           | Compute device                                   | `cuda` or `cpu`  |
| `confidence`       | Detection confidence threshold (0–1)             | `0.5`            |
| `iou`              | NMS IoU threshold (0–1)                          | `0.7`            |
| `classes`          | COCO class IDs to track (e.g. car=2, bus=5)      | `[2, 5, 6, 7]`   |
| `source_video_path`| Path to the input video                          | `input.mp4`      |
| `target_video_path`| Path for the annotated output video              | `output.mp4`     |

## Downloading a Source Video from YouTube

```bash
python tools/download_from_yt.py --url "https://www.youtube.com/watch?v=<id>" \
    --output_path ./videos --file_name traffic.mp4
```
