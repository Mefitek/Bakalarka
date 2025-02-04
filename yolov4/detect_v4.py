import cv2
import numpy as np

### PATHS ####
IMG_PATH = "C:\\Users\\mefit\\Desktop\\BAK\\test\\qr_pos_10_jpg.rf.ffe42b0784adcf25dbd4f1d15547b09f.jpg"
CLASSES_PATH = "C:\\Users\\mefit\\Desktop\\BAK\\yolov4\\yolov4-tiny\\QR_points.names"
YOLOv4_CONFIG = "C:\\Users\\mefit\\Desktop\\BAK\\yolov4\\yolov4-tiny\\yolo-tiny.cfg"
YOLOv4_WEIGHTS = "C:\\Users\\mefit\\Desktop\\BAK\\yolov4\\yolov4-tiny\\backup\\Yolo-tiny_best.weights"

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]


### Load image
img = cv2.imread(IMG_PATH)


# read object category list
with open(CLASSES_PATH, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
print("Classe: " +str(classes))

# create random color to draw for each class 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# load model
net = cv2.dnn.readNet(YOLOv4_WEIGHTS, YOLOv4_CONFIG)

# get the output layers from YOLO architecture for reading output predictions
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

# helper function for drawing bounding boxes
def draw_prediction(image, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(image, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(image, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, confidence, color, 2)

# running inference on input frame
def run_inference(image):
    
    Width = image.shape[1]
    Height = image.shape[0]

    blob = cv2.dnn.blobFromImage(image, 1/255, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    
    for i in indices:
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        print("[INFO] detected {} with bbox {}".format(str(classes[class_ids[i]]),
                                [[int(x),int(y)], [int(x+w),int(y+h)]]))
        draw_prediction(image, class_ids[i], confidences[i], int(x),
                        int(y), int(x+w), int(y+h))
        
run_inference(img)

cv2.imshow('Image', img)
cv2.waitKey(0)


