# Image-Segmentation-using-YOLOv8

This is a simple implementation of YOLO segmentation for real-time object detection and segmentation in video streams. The code uses the YOLO (You Only Look Once) object detection model with segmentation capabilities to process video frames and display the results in a window. The model predicts bounding boxes and segmentation contours for each object detected in the video frame.

Dependencies

    OpenCV: pip install opencv-python
    YOLO (Ultralytics): pip install ultralytics

Usage

    Download the YOLOv8 segmentation model weights file yolov8l-seg.pt from the official Ultralytics repository or train your own model.
    Run the script: python main.py
    
    Link to the pre-trained model: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt

Code Overview
YOLOSegmentation class

This class initializes the YOLO model and provides a detect method for object detection and segmentation in images.

    __init__(self, model_path): Initializes the YOLO model with the provided model path.
    detect(self, img): Detects objects in the given image, returning bounding boxes, class IDs, segmentation contours, and confidence scores.

main function

The main function initializes the YOLOSegmentation class, captures video frames from the webcam, and processes the frames using the YOLO model. The processed frames are displayed in a window.

    modelobj = YOLOSegmentation('yolov8l-seg.pt'): Initialize the YOLOSegmentation class with the YOLOv8 segmentation model weights.
    cap = cv2.VideoCapture(0): Capture video from the default camera (webcam).
    frame = cv2.resize(frame, None, fx=0.7, fy=0.7): Resize the captured frame.
    bboxes, classes, segmentations, scores = modelobj.detect(frame): Detect objects and their segmentation contours in the frame.
    cv2.imshow("Video Feed", frame): Display the processed frame with bounding boxes and segmentation contours.
    cv2.waitKey(1) & 0xFF == ord('q'): Exit the loop and close the video feed window when the 'q' key is pressed.
