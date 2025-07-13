from typing import List

from ultralytics import YOLO

import supervision as sv

import cv2
import numpy as np

from utils.timers import ClockBasedTimer
from utils import find_in_list, get_stream_frames_generator, load_zones_config


source_video_path = '/kaggle/input/traffic-video/4K Road traffic video for object detection and tracking - free download now - Karol Majek (720p h264) (online-video-cutter.com).mp4'
target_video_path = 'output.mp4'

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)

def main(
    weights: str,
    device: str,
    confidence: float,
    iou: float,
    classes: List[int],
    source_video_path: str = source_video_path,
    target_video_path: str = target_video_path,
) -> None:
    model = YOLO(weights)
    tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
    
    frames_generator = get_stream_frames_generator(source_video_path)
    
    video_info = sv.VideoInfo.from_video_path(source_video_path)

    fps_monitor = sv.FPSMonitor()

    polygons = load_zones_config
    
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    
    timers = [ClockBasedTimer() for _ in zones]
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frames_generator:
            fps_monitor.tick()
            fps = fps_monitor.fps
    
            results = model(frame, verbose=False, device=device, conf=confidence)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[find_in_list(detections.class_id, classes)]
            detections = detections.with_nms(threshold=iou)
            detections = tracker.update_with_detections(detections)
    
            annotated_frame = frame.copy()
            annotated_frame = sv.draw_text(
                scene=annotated_frame,
                text=f"{fps:.1f}",
                text_anchor=sv.Point(40, 30),
                background_color=sv.Color.from_hex("#A351FB"),
                text_color=sv.Color.from_hex("#000000"),
            )
    
            for idx, zone in enumerate(zones):
                annotated_frame = sv.draw_polygon(
                    scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
                )
    
                detections_in_zone = detections[zone.trigger(detections)]
                time_in_zone = timers[idx].tick(detections_in_zone)
                custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

                annotated_frame = COLOR_ANNOTATOR.annotate(
                    scene=annotated_frame,
                    detections=detections_in_zone,
                    custom_color_lookup=custom_color_lookup,
                )
                labels = [
                    f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                    for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
                ]
                annotated_frame = LABEL_ANNOTATOR.annotate(
                    scene=annotated_frame,
                    detections=detections_in_zone,
                    labels=labels,
                    custom_color_lookup=custom_color_lookup,
                )
                sink.write_frame(annotated_frame)

main(weights='yolov8m.pt', device='cuda', confidence=0.5, iou=0.7, classes=[2, 5, 6, 7], source_video_path=source_video_path, target_video_path=target_video_path)