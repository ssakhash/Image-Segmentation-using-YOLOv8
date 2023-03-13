import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter

class YOLOSegmentation:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):
        height, width, _ = img.shape
        results = self.model.predict(source=img.copy(), save=False, save_txt=False)
        result = results[0]
        segmentation_contours_idx = []
        for seg in result.masks.segments:
            # contours
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)
            segmentation_contours_idx.append(segment)

        bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
        class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
        scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
        return bboxes, class_ids, segmentation_contours_idx, scores


def main():
    # Initialize YOLO model
    modelobj = YOLOSegmentation('yolov8l-seg.pt')

    # Initialize video capture object
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open video capture object")
        return

    while True:
        ret, frame = cap.read()
        # Check if a frame was successfully captured
        if not ret:
            print("Unable to capture frame")
            break
            
        frame = cv2.resize(frame, None, fx=0.7, fy=0.7)
        bboxes, classes, segmentations, scores = modelobj.detect(frame)

        # Draw bounding boxes and segmentation contours on the frame
        class_counts = dict(Counter(classes))
        color_dict = {}
        for idx, (class_id, count) in enumerate(class_counts.items()):
            color_dict[class_id] = tuple(np.random.randint(0, 255, 3).tolist())

        for bbox, classid, seg, score in zip(bboxes, classes, segmentations, scores):
            (x1, y1, x2, y2) = bbox
            if score > 0.5:
                mask = np.zeros_like(frame)
                color = color_dict.get(classid, (255, 255, 255))
                cv2.fillPoly(mask, [seg], color)
                frame = cv2.addWeighted(frame, 1, mask, 0.5, 0)

        cv2.imshow("Video Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()