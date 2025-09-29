import torch
import numpy as np
import ultralytics
from ultralytics import YOLO 
import multiprocessing  
import os

print(os.path.exists("yolo/yaml/data.yaml"))  # Returns False if the path is incorrect.

if __name__ == "__main__":
    multiprocessing.freeze_support()  

    # Step 1 : Load pre-trained model
    model = YOLO("models/yolov8n.pt")

    # Step 2 : Train the model
    results = model.train(
        data=r"yolo/yaml/data.yaml",          # Training Mission File
        imgsz=1024,                           
        epochs=150,                           
        patience=50,                          
        batch=1,                              
        lr0=0.0005,                           
        optimizer="AdamW",                    
        project="yolov8_Judgment of missing teeth",  
    )
