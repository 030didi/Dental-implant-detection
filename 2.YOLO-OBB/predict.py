import ultralytics

ultralytics.checks()
import os
from ultralytics import YOLO

model = YOLO(
    r"path"
)  # yolov8n.pt、yolov8s.pt、yolov8n-seg.pt、yolov8s-world.pt ...

results = model.predict(
        source=r"path",  
        conf=0.7,
        save=True,
        save_txt=True,
        save_conf=True,
        visualize=False,
    )


