import os
import pickle
from utils import YOLOVideoProcessor
import time
from ultralytics import YOLO


def main():
    file_path = "parking-spots.pkl"
    if not os.path.exists(file_path):
        print("No save detected. Please run select-regions.py to define polygon regions first.")
        exit()
        
    with open("parking-spots.pkl", 'rb') as file:
        stored_polygons = pickle.load(file)
        img_dimensions = pickle.load(file)

    print("Polygons:", stored_polygons)
    print("Image dimensions:", img_dimensions)

    processor = YOLOVideoProcessor(YOLO("model/yolov8s.pt"), "data/videos/parking1.mp4", 30, True, True,
                                   classes=[2, 3, 5, 7],  polygons=stored_polygons,  
                                #    img_width=img_dimensions[0], img_height=img_dimensions[1],
                                   )
    
    try:
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        processor.stop()


if __name__ == "__main__":
    main()