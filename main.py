import cv2
import time
import signal
import sys
from ultralytics import YOLO

stats = []

def save_stats():
    with open('stats.txt', 'w') as f:
        f.write("Per-frame statistics:\n")
        f.write("Frame, Processing Time (s), Detections\n")
        for stat in stats:
            f.write(f"{stat['frame']}, {stat['processing_time']:.2f}, {stat['detections']}\n")
        f.write("\nSummary:\n")
        f.write(f"Total frames processed: {len(stats)}\n")
        total_detections = sum(stat['detections'] for stat in stats)
        f.write(f"Total detections: {total_detections}\n")
        total_time = sum(stat['processing_time'] for stat in stats)
        video_length = len(stats) / fps
        average_time_per_frame = total_time / len(stats)
        average_detections_per_frame = total_detections / len(stats)
        f.write(f"Output video length: {video_length:.2f} seconds\n")
        f.write(f"Average time per frame: {average_time_per_frame:.2f} seconds\n")
        f.write(f"Average detections per frame: {average_detections_per_frame:.2f}\n")

def signal_handler(sig, frame):
    cap.release()
    out.release()
    save_stats()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

model = YOLO('yolov5nu.pt')
video_path = 'car.mp4'
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_start = time.time()
    results = model.predict(frame, imgsz=320, device='cpu')
    detections_in_frame = 0
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            label = model.names[int(box.cls[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            detections_in_frame += 1
    out.write(frame)
    frame_end = time.time()
    stats.append({
        'frame': frame_count + 1,
        'processing_time': frame_end - frame_start,
        'detections': detections_in_frame
    })
    frame_count += 1
    print(f"Processed frame {frame_count}, Time: {frame_end - frame_start:.2f}s, Detections: {detections_in_frame}")

cap.release()
out.release()
save_stats()
