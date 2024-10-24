import threading
import cv2
import time
from ultralytics import YOLO
from ultralytics.engine.results import Results
from matplotlib.path import Path
import numpy as np
from shapely import box, Polygon


class VideoCapture:
    """
    VideoCapture
    ------------
    This class is designed to be a wrapper for cv2.VideoCapture to streamline
    video feed handling by using threading.

    The primary function is that the read() function is run constantly in a separate thread.
    Locks and Events are used to allow this class to be thread-safe.


    Parameters:
        video_source (str): Path to a video file or a video feed URL (rtsp/rtmp/http).
        capped_fps (bool): If True, caps the frame rate (default is False). Set to true for file playback. 
        framerate (int): Frame rate for video file playback (used if capped_fps is True).
        restart_on_end (bool): If True, restarts video file playback when it ends (default is False).
    """
    
    last_frame = None
    last_ready = False
    lock = threading.Lock()
    stop_event = threading.Event()
    start_event = threading.Event()
    fps = 30
    video_source = None
    capped_fps = False
    restart_on_end = False

    # make sure capped_fps is False for case of rstp/rtmp url 
    def __init__(self, video_source:str, framerate:int=30, capped_fps:bool=False, restart_on_end:bool=False):
        self.fps : int = framerate
        self.video_source : str = video_source
        self.capped_fps : bool = capped_fps
        self.restart_on_end : bool = restart_on_end
        self.cap : cv2.VideoCapture = cv2.VideoCapture(video_source)
        self.thread : threading.Thread = threading.Thread(target=self.__capture_read_thread__)
        self.thread.daemon = True
        self.thread.start()

    def __capture_read_thread__(self):
        """
        Continuously reads frames from the video source in a separate thread.

        This method is intended to run in a separate thread and is not meant to be called directly.
        It reads frames as soon as they are available, and handles video restart if specified.
        If capped_fps is True, it waits to maintain the specified frame rate.

        The method stops when stop_event is set or if the video source cannot provide frames and restart_on_end is False.
        """
        while not self.stop_event.is_set():
            if self.start_event.is_set():
                self.last_ready, last_frame = self.cap.read()
                if self.last_ready:
                    with self.lock:
                        self.last_frame = last_frame
                else:
                    self.last_frame = None

                # print(self.video_source, " frame read: ", self.last_ready)
            
                if not self.last_ready and self.restart_on_end:  # restart if file playback video ended
                    with self.lock:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
                # only wait in case of video files, else keep reading frames without delay to prevent packet drop/burst receive
                if self.capped_fps:
                    time.sleep(1 / self.fps)
        return
    
    def start(self):
        """
        Start the video capture reading frames.
        """
        print("Capture thread has started.")
        self.start_event.set()
        # sleep to allow
        time.sleep(1/self.fps) # sleep for a tiny bit to allow capture thread to start properly

    def read(self):
        """
        Retrieve the latest frame from the video capture. Skips frames automatically 
        if they are not read in time. Allows for realtime video playback.

        Returns:
            [success boolean, frame or None]
        """
        try:
            if self.start_event.is_set():
                # with self.lock:
                if (self.last_ready is not False) and (self.last_frame is not None):
                    return [self.last_ready, self.last_frame.copy()]
            else:
                if not self.stop_event.is_set():
                    self.start()

            return [False, None]
        except Exception as e:
            raise ValueError(f"Error encountered by read() function: {e}")
        
    def isOpened(self):
        """
        Check if the video source is opened.
        """
        return self.cap.isOpened()    

    def open(self, video_source):
        """
        Open a new video source.
        
        Args:
            video_source (str): Path to the new video source.
        """
        self.video_source = video_source
        with self.lock:
            self.cap.open(self.video_source)

    def release(self):
        """
        Stop the video capture and release resources.
        """
        self.stop_event.set()
        self.restart_on_end = False
        self.thread.join(2)
        
        with self.lock:
            self.cap.release()



class YOLOVideoProcessor:
    """
    YOLOVideoProcessor
    ------------------
    This class processes video frames using YOLO for object detection.
    It uses the VideoCapture class to manage video feeds and applies YOLO
    detection on each frame. Processing is done in separate threads.

    Parameters:
        video_source (str): Path to a video file or a video feed URL.
        yolo_model (YOLO): An instance of a YOLO model from Ultralytics.
        framerate (int): Frame rate for video file playback.
        capped_fps (bool): If True, caps the frame rate.
        restart_on_end (bool): If True, restarts video file playback when it ends.
        confidence (float): Confidence threshold of YOLO model to perform OCR on track object.
        classes (list[int]): The class list of YOLO of objects to track. Empty list/track all by default.
        pixel_padding (int): Number of pixels to pad the bounding box region.
        img_width/img_height (int): Dimensions of output display window.
        top_k (int): Store the k highest confidence images for sake of performing OCR. Default is 5.
    """
    
    def __init__(self, yolo_model: YOLO, video_source: str, framerate: int = 30, 
                 capped_fps: bool = False, restart_on_end: bool = False, confidence:float=0.75, 
                 classes:list=[2,3,5,7], pixel_padding:int=5, img_width:int=800, img_height:int=600,
                 annotations:bool=True, IoU_threshold:float=0.4, polygons:dict = {}):
        
        # video capture
        self.video_capture = VideoCapture(video_source, framerate, capped_fps, restart_on_end)
        self.img_width = img_width
        self.img_height = img_height

        # yolo basics
        self.yolo_model = yolo_model
        self.classes = classes # defaulted to detect cars/vehicles/motorbikes
        self.confidence = confidence
        self.padding = pixel_padding
        self.annotations : bool = annotations
        # basic polygons { x: [(),(),()] }
        self.polygons : dict = polygons
        # shapely polygons with areas pre-calculated (makes calculating IoU much more efficient)
        self.shapely_polygons = {keys: Polygon(polygon) for keys, polygon in polygons.items()}
        # using path objects to check whether points are contained inside them or not
        self.polygon_paths = {keys: Path(polygon) for keys, polygon in polygons.items()}
        # polygons already converted to cv2 requirements for sake of drawing them with minimal processing at time of need
        self.drawing_polygons = {keys: np.array(polygon, dtype=np.int32).reshape((-1,1,2)) for keys, polygon in polygons.items()}
        # empty parking spots = polygons to draw, occupied parking spots = don't draw
        self.occupied_polygons = {} # the parking spots which have a car on them in the current frame
        self.iou_threshold = IoU_threshold
        self.tracked_ids = []

        # frame processing thread
        self.stop_event = threading.Event()
        self.processing_thread = threading.Thread(target=self.__process_frames__)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    
    def __draw_polygons(self, frame, color=(80,120,0), alpha=0.5):
        """
        Draw polygons in each frame to represent empty parking spots.
        
        Parameters:
            frame (numpy.ndarray): The original frame to draw polygons on
            color (tuple(int, int, int)): BGR tuple for fill colour of polygons
            alpha (float): b/w 0 and 1 to select level of transparency of parking spots

        Returns:
            output_frame (numpy.ndarray): Frame with only unoccupied parking spots shown
        """
        # temp image to store polygons so we can add transparency via weighted addition
        polygon_image = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

        # drawing loop - draw only unoccupied spots
        for key, parking_spot_pts in self.drawing_polygons.items():
            if key in self.occupied_polygons.keys(): # if a spot is occupied, do not draw it
                continue

            cv2.fillPoly(polygon_image, [parking_spot_pts], color=color) 
        # fully visible original frame, partially visible polygon image for transparency
        output_image = cv2.addWeighted(frame, 1, polygon_image, alpha, 1)
        return output_image
    

    def calculate_iou(self, bbox:list|tuple, polygon:int):
        """
        Calculate IoU for a box and a polygon.

        Parameters:
            bbox (tuple | list): xyxy as a tuple or list
            polygon (int): key of polygon to use for intersection calculation 
        """
        bbox_shape : Polygon = box(bbox[0], bbox[1], bbox[2], bbox[3])
        polygon_shape : Polygon = self.shapely_polygons[polygon]

        # calculate intersection
        intersection = bbox_shape.intersection(polygon_shape)

        # get areas
        inter_area = intersection.area
        bbox_area = bbox_shape.area
        poly_area = polygon_shape.area

        union_area = bbox_area + poly_area - inter_area

        return inter_area / union_area if union_area != 0 else 0



    def __car_in_parking_spot(self, car_id, bbox):
        """
        This function checks if the car with given xywh is within a parking spot.

        Parameters:
            car_id (int): The id of the detected car as determined by YOLO
            bbox (list | tuple): list or tuple of xyxy determined  by YOLO
        """
        # using IoU w/ threshold to determine if a car is on a spot
        for key, _ in self.shapely_polygons.items():
            if self.calculate_iou(bbox, key) >= self.iou_threshold:
                self.occupied_polygons[key] = car_id

        ## centroid logic
        # car_center = (x_center, y_center)
        # for key, polygon in self.polygon_paths.items():
        #     if polygon.contains_point(car_center):
        #         # print(f"{car_id} contained in spot {key}")
        #         self.occupied_polygons[key] = car_id # certain spot contains certain car, future-proofing
        #         # break # car is in one parking spot, no need to check more (car could be in multiple adjacent spots at once)


    def __new_object_in_frame(self, id:int):
        """
        All functionality for when a new object is being tracked. 
        """
        # print("new obj detected")
        self.tracked_ids.append(id)
    

    def __object_being_tracked(self, id:int):
        """
        All functionality for an object that is actively being tracked.
        """
        return None
    

    def __object_left_frame(self, id:int):
        """
        All functionality to deal with a tracked object that has left the feed. Pass ids of all objects to start with.

        Deletes the entry from the tracked ids list.
        """
        self.tracked_ids.remove(id)
        # print(f"\nObject left frame: {id}\n")
    

    def __process_frames__(self):
        """
        Continuously processes and displays frames from the video source using YOLO.
        Run in a separate thread. Enables ease of running multiple feeds very easily.
        """
        # get feed name to display window title
        feed_name = self.video_capture.video_source
        self.video_capture.start()
        
        while not self.stop_event.is_set():
            ret, frame = self.video_capture.read()
            
            if not ret:
                print("Frame not read in yolo vid proc")
                time.sleep(0.5)
                continue

            frame = cv2.resize(frame, (self.img_width, self.img_height))
            annotated_frame = frame

            results = self.yolo_model.track(frame, stream=True, persist=True, verbose=False, classes=self.classes)

            for res in results:

                # all ids detected by yolo
                current_ids = res.boxes.id.int().cpu().tolist() if res.boxes.id is not None else []

                # items previously tracked that are no longer tracked ==> they have left the frame(/are obstructed)
                ids_left_frame = set(self.tracked_ids) - set(current_ids)
                for id in ids_left_frame:
                    self.__object_left_frame(id)
                    
                # if no one detected in single result, skip
                if res.boxes.id is None:
                    continue
                
                # functionality for if an object is detected at all
                self.__yolo_detection_processing(res, frame)

                if self.annotations: # only draw annotations if set
                    annotated_frame = res.plot() # plot all objects that are detected
            
            annotated_frame = self.__draw_polygons(annotated_frame)
            cv2.imshow(feed_name, annotated_frame)

            self.occupied_polygons = {}

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()


    def __yolo_detection_processing(self, res : Results, original_frame : np.ndarray):
        """
        Separate out the detection processing code for modularity.

        Args:
            res (ultralytics Results): single item in the list of results returned by yolo.track()
            original_frame (np.ndarray): the original frame being processed
        """
        # for each detection in the result
        for detection in res.boxes:

            # det_conf = detection.conf[0]
            # if det_conf <= self.confidence:
            #     continue

            # id of detection by yolo, always unique
            det_id = int(detection.id[0])
            
            # centroid + obj width/height
            # x_center, y_center, width, height = detection.xywh[0]
            x1,y1,x2,y2 = detection.xyxy[0]
            bbox = (int(x1),int(y1),int(x2),int(y2))
            # always try to check if car in parking spot
            self.__car_in_parking_spot(det_id, bbox)
            
            # specific behaviours for an object that is being tracked/is newly tracked
            if det_id in self.tracked_ids:
                self.__object_being_tracked(det_id)

            else:
                self.__new_object_in_frame(det_id)


    def stop(self):
        """
        Stop the video processing and release resources.
        """
        self.stop_event.set()
        self.processing_thread.join()
        self.video_capture.release()
