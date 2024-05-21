import cv2
import numpy as np
from main import YOLOv8_face
import argparse
import random
rtsp_url = "rtsp://admin:ADMIN@123@192.168.0.3:554/Streaming/Channels/101"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: No cam")
    exit()

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

def checkFrameVideo():
    pass

def isPointIn(x,y,dstimg,polygon):
    print("inside is point")
    h,w,_=dstimg.shape


    #draw polygon
    cv2.polylines(dstimg,[polygon],isClosed=True,color=(0,255,255),thickness=2)

    #check for point
    inside= cv2.pointPolygonTest(np.array(polygon, np.int32), (int(x),int(y)), False) >= 0

    print(f"Test point inside: ${inside} ")
    return inside

parser = argparse.ArgumentParser()
parser.add_argument('--imgpath', type=str, default='images/4.png', help="image path")
parser.add_argument('--modelpath', type=str, default='weights/yolov8n-face.onnx',
                    help="onnx filepath")
parser.add_argument('--confThreshold', default=0.45, type=float, help='class confidence')
parser.add_argument('--nmsThreshold', default=0.5, type=float, help='nms iou thresh')
args = parser.parse_args()

# Initialize YOLOv8_face object detector
YOLOv8_face_detector = YOLOv8_face(args.modelpath, conf_thres=args.confThreshold, iou_thres=args.nmsThreshold)


# Draw polygon
polygon = [(random.randint(0,w),random.randint(0,h)) for _ in range(random.randint(3,10))]
polygon = np.array(polygon,np.int32).reshape((-1,1,2))

# Draw detections
while True:
    ret,frame= cap.read()
    if not ret:
        break
    # srcimg = cv2.imread(frame)
    # Detect Objects
    boxes, scores, classids, kpts = YOLOv8_face_detector.detect(frame)
    dstimg = YOLOv8_face_detector.draw_detections(frame, boxes, scores, kpts)

    nose_kp = kpts[0]
    print(kpts)
    x,y,c=nose_kp[0],nose_kp[1],nose_kp[2]
    isPointIn(int(x),int(y),dstimg,polygon)
    cv2.imwrite('result.jpg', dstimg)
    winName = 'Deep learning face detection use OpenCV'
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, dstimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
