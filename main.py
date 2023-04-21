
import cv2
from ultralytics import YOLO
import random
from tracker import Tracker
import time

cap = cv2.VideoCapture(0)

ret, frame = cap.read()

model = YOLO("yolov8n.pt")
tracker = Tracker()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]
detection_threshold = 0.5

classes = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

fps = 0
frame_count = 0
start_time = time.time()

while ret:
    frame_count += 1
    results = model(frame,conf=0.5,verbose=False)

    for result in results:        
        detections = []
        for r in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)

            if classes[class_id] == 'car':
                if score > detection_threshold:
                    detections.append([x1, y1, x2, y2, score])
                
        if(detections!=[]):
            tracker.update(frame, detections)
            
            for track in tracker.tracks:
    
                bbox = track.bbox
                x1, y1, x2, y2 = bbox
                track_id = track.track_id
                

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                cv2.putText(frame, "ID:"+str(track_id),(int(x1),int(y1)-10),cv2.FONT_HERSHEY_PLAIN,1, (colors[track_id % len(colors)]), 2)
    


    cv2.putText(frame, f'FPS: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xff == 27:
        break


    ret, frame = cap.read()

    if time.time() - start_time >= 1:
        fps = frame_count
        frame_count = 0
        start_time = time.time()

cap.release()
cv2.destroyAllWindows()
