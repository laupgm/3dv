from ultralytics import YOLO
import numpy as np
from metrics_functions import *
import statistics as stats
from operator import itemgetter

if __name__ == '__main__':

    # Load a pretrained YOLO11n model
    model = YOLO("yolo11l.pt")

    # Train the model on the COCO8 dataset for 100 epochs
    train_results = model.train(data="3dv-1\data.yaml", epochs=100, imgsz=640)

    # Evaluate the model's performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    #model=YOLO('runs\segment/train7\weights/best.pt')
    #results = model("dataset/3dv-1/test\images\student-48_jpg.rf.2485b3d54d4a31d8d7a83c1a520c8f5c.jpg")  # Predict on an image
    #results[0].show()  # Display results

