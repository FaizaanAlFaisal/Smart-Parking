import threading
import cv2
import time
from typing import Tuple
from ultralytics import YOLO
from ultralytics.engine.results import Results
from matplotlib.path import Path
import numpy as np
import os


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
    last_ready = None
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
                if (self.last_ready is not None) and (self.last_frame is not None):
                    return [self.last_ready, self.last_frame.copy()]
            else:
                if not self.stop_event.is_set():
                    print("Start event not set.")
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
                 classes:list=[], pixel_padding:int=5, img_width:int=800, img_height:int=600,
                 polygons:dict = {}):
        
        # video capture
        self.video_capture = VideoCapture(video_source, framerate, capped_fps, restart_on_end)
        self.img_width = img_width
        self.img_height = img_height

        # yolo basics
        self.yolo_model = yolo_model
        self.classes = classes
        self.confidence = confidence
        self.padding = pixel_padding
        # basic polygons { x: [(),(),()] }
        self.polygons : dict = polygons
        # using path objects to check whether points are contained inside them or not
        self.polygon_paths = {keys: Path(polygon) for keys, polygon in polygons.items()}
        # polygons already converted to cv2 requirements for sake of drawing them with minimal processing at time of need
        self.drawing_polygons = {keys: np.array(polygon, dtype=np.int32).reshape((-1,1,2)) for keys, polygon in polygons.items()}
        # empty parking spots = polygons to draw, occupied parking spots = don't draw
        self.occupied_polygons = []
        self.tracked_ids = []

        # frame processing thread
        self.stop_event = threading.Event()
        self.processing_thread = threading.Thread(target=self.__process_frames__)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    
    def __draw_polygons(self):
        """
        Draw polygons in each frame to represent empty parking spots.
        """


    def __new_object_in_frame(self, id:int):
        """
        All functionality for when a new object is being tracked. 
        """
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
        print(f"\nObject left frame: {id}\n")
    

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

            results = self.yolo_model.track(frame, stream=True, persist=True, verbose=False, classes=[0])

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
                annotated_frame = res.plot() # plot all objects that are detected
                
            cv2.imshow(feed_name, annotated_frame)

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

            det_conf = detection.conf[0]
            if det_conf <= self.confidence:
                continue

            # id of detection by yolo, always unique
            det_id = int(detection.id[0])
            
            # bounding box and crop of object detected
            xmin, ymin, xmax, ymax = detection.xyxy[0]
            bbox = (int(xmin), int(ymin), int(xmax), int(ymax))
            
            # if object has not been detected till now
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
