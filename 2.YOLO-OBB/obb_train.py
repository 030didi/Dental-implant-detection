import torch
import numpy as np
import ultralytics
from ultralytics import YOLOWorld, YOLO  
import multiprocessing  

if __name__ == "__main__":
    multiprocessing.freeze_support()  

    model = YOLO("models/yolo8n-obb.pt")  

    results = model.train(  
        data=r"data.yaml",  
        imgsz=640,  
        epochs=150,  
        patience=50,  
        batch=1,  
	optimizer="AdamW",
        project="Yolov8_obb_rain_res",  
        name="final_v8obb",
    )
