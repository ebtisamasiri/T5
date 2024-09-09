from ultralytics import YOLO
import cv2
# from google.colab.patches import cv2_imshow
import cvzone #provides easy-to-use functions built on top of OpenCV
import math

cap = cv2.VideoCapture("Video/car2.mov")

model = YOLO('best2.pt')

#from traffic dataset
classNames = ['ambulance', 'army vehicle', 'auto rickshaw', 'bicycle', 'bus', 'car', 'garbagevan',
              'human hauler', 'minibus', 'minivan', 'motorbike', 'pickup', 'policecar', 'rickshaw',
              'scooter', 'suv', 'taxi', 'three wheelers -CNG-', 'truck', 'van', 'wheelbarrow'
              ]

# mask = 2.imread('mask.png')
while True:
    success, img = cap.read()
    # imgRegion = cv2.bitwise_and(img, mask)
    results = model(img, stream=True) #imgRegion
    for r in results:
        boxes = r.boxes #getting the bounding box for each of the results
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0] #xywh
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3) #3->thickness
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100 #* 100)) / 100  to make 2 decimal places

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in ["car", "bus", "truck", "motorbike", "train"] and conf > 0.25:
            # if currentClass == 'car' or currentClass == 'bus' or currentClass == 'truck' \
            #         or currentClass == 'motorbike' and conf > 0.3 :

                cvzone.putTextRect(img,
                                   f'{currentClass} {conf}',
                                   (x1,y1), #(max(0, x1), max(35, y1)),
                                   scale=1,
                                   thickness=2,
                                   offset=5)
                cvzone.cornerRect(img, (x1, y1, w, h), l=8)

    # cv2_imshow(img)
    cv2.imshow("Video", img)
    cv2.waitKey(1)