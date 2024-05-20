import cv2
import numpy as np
from main import YOLOv8_face
import argparse
import random

def is_point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(np.array(polygon, np.int32), point, False) >= 0


def isPointIn(x,y,dstimg):
    print("inside is point")
    h,w,_=dstimg.shape
    polygon = [(random.randint(0,w),random.randint(0,h)) for _ in range(random.randint(3,10))]
    polygon = np.array(polygon,np.int32).reshape((-1,1,2))


    #draw polygon
    cv2.polylines(dstimg,[polygon],isClosed=True,color=(0,255,255),thickness=2)

    #check for point
    inside = is_point_in_polygon((int(x),int(y)),polygon)
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
srcimg = cv2.imread(args.imgpath)

# Detect Objects
boxes, scores, classids, kpts = YOLOv8_face_detector.detect(srcimg)

# Draw detections
dstimg = YOLOv8_face_detector.draw_detections(srcimg, boxes, scores, kpts)

nose_kp = kpts[0]
print(kpts)
x,y,c=nose_kp[0],nose_kp[1],nose_kp[2]
isPointIn(int(x),int(y),dstimg)

#cv2.imwrite('result.jpg', dstimg)
winName = 'Deep learning face detection use OpenCV'
cv2.namedWindow(winName, 0)
cv2.imshow(winName, dstimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
