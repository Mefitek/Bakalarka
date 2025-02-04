from PIL import Image #library link: https://pillow.readthedocs.io/en/stable/
import numpy
import cv2 #https://pypi.org/project/opencv-python/; pro funkce imread(), threshold(), práci s Yolov4
from math import sqrt
import imutils # rotate piture
from pyzbar.pyzbar import decode #https://pypi.org/project/pyzbar/


# function to get the color of brightest pixel in a PIL image
def get_brightest(image):
   # convert img to an array (if it isn't one already)
    if isinstance(image, numpy.ndarray): image = image
    else: image = numpy.asarray(image)

    orig = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # convert to gray
    # apply a Gaussian blur - to prevent detecting noise as brightest pixel
    image = cv2.GaussianBlur(image, (5, 5), 0)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(image)

    image = orig.copy()
    image = Image.fromarray(image)
    color = image.getpixel(maxLoc)
    return color

# function crop an image using the given normalized coordinates
def my_crop(img_input, x1,y1,x2,y2, normalized = True ,add_quiet = True, show=False):
    
    # checks img_input type, convert to PIL image if needed
    if isinstance(img_input, str):
        img = Image.open(img_input) # Opens a image in RGB mode
    elif isinstance(img_input, numpy.ndarray):
        img = Image.fromarray(img_input) # Convert array to img
    else:
        img = img_input

    width, height = img.size

    #un-normalize dormalized coordinates
    if normalized:
        x1=x1*width 
        x2=x2*width
        y1=y1*height
        y2=y2*height
        
    if(not add_quiet):
       img_crop = img.crop((x1, y1, x2, y2))
       return img_crop #returns cropped image

    #add a 10 % extra space around cropped image to accommodate for QR code's Quiet Zone

    # coordinates that assume the +10% margin
    x1_m=(int)(x1-(0.1*(x2-x1)))
    x2_m=(int)(x2+(0.1*(x2-x1)))
    y1_m=(int)(y1+(0.1*(y1-y2)))
    y2_m=(int)(y2-(0.1*(y1-y2)))

    if ((x2_m-x1_m) < width) and ((y1_m-y2_m) < height): #if the requiered area isn't bigger that original, use Image.crop() function
        img_crop = img.crop((x1_m, y1_m, x2_m, y2_m))
    else: #otherwise we add white margin, as the crop() function would add black margin
        img1 = img.crop((x1, y1, x2, y2))
        width_1 = (int)((x2-x1)*1.2)
        height_1 = (int)((y2-y1)*1.2)
        lightest_color = get_brightest(img1)
        img_crop = Image.new(img.mode, (width_1, height_1), lightest_color)
        img_crop.paste(img1, ((int)((x2-x1)*0.1), (int)((y2-y1)*0.1)))
    
    if(show):
        img_crop.show()

    return img_crop #returns cropped image

# function that reads a .txt file containing a label and returns it as a list
def read_label(label_path, print_arr=False):
    list_of_coordinates = []
    coordinates = []
    with open(label_path) as f:
        lines = f.readlines()
        i=0
        for line in lines:
            words = line.split()
            j=0
            for word in words:
                if(j==0):
                    number = int(word)
                else:
                    number = float(word)
                j=j+1
                coordinates.append(number)
            list_of_coordinates.append(coordinates)
            coordinates = []
            i=i+1

    if print_arr==True:
        print()
        for coord in list_of_coordinates:
            print(coord)

    return list_of_coordinates

# function uses the crop() function to crop out the images considering the label list format from my_det)) function
def crop_by_labels(labels, img_path, add_quiet = True, show=False):
    imgs_crop = []
    j=0
    for i in labels:
        if(j%2 == 1):
            imgs_crop.append(my_crop(img_path, labels[j][0], labels[j][1], labels[j][2], labels[j][3], True, add_quiet, show))
        j=j+1
    return imgs_crop #return list of cropped images

# crop the qr code base on list of sorted position markers
def crop_by_pos(img, pos, add_quiet = True, show=False):
    # calculate the coordinates based on position markers
    x1 = min(pos[0][1], pos[0][3], pos[1][1], pos[1][3], pos[2][1], pos[2][3])
    y1 = min(pos[0][2], pos[0][4], pos[1][2], pos[1][4], pos[2][2], pos[2][4])
    x2 = max(pos[0][1], pos[0][3], pos[1][1], pos[1][3], pos[2][1], pos[2][3])
    y2 = max(pos[0][2], pos[0][4], pos[1][2], pos[1][4], pos[2][2], pos[2][4])

    if pos[0][1] < pos[1][1]: x2 = x2 + (pos[1][1]-pos[0][1])
    else: x2 = x2 + (pos[0][1]-pos[1][1])
    if pos[1][2] > pos[2][2]: y2 = y2 + (pos[1][2]-pos[2][2])
    else: y2 = y2 + (pos[2][2]-pos[1][2])

    if show:
        print("Positions for cropping (x1, y1, x2, y2):")
        print(x1)
        print(y1)
        print(x2)
        print(y2)

    # crop the image using the my_crop function
    img_crop = my_crop(img, x1,y1,x2,y2, normalized=False, add_quiet=add_quiet, show=False)

    pos_new = pos
    i=0
    for p in pos:
        pos_new[i][1] = pos[i][1]-x1
        pos_new[i][2] = pos[i][2]-y1
        pos_new[i][3] = pos[i][3]-x1
        pos_new[i][4] = pos[i][4]-y1
        if add_quiet:
            pos_new[i][1]=pos_new[i][1] + (x2-x1)/10
            pos_new[i][2]=pos_new[i][2] + (y2-y1)/10
            pos_new[i][3]=pos_new[i][3] + (x2-x1)/10
            pos_new[i][4]=pos_new[i][4] + (y2-y1)/10
        i=i+1

    if show:
        for p_n in pos_new:
            draw_prediction(img_crop, "bod", int(p_n[1]), int(p_n[2]), int(p_n[3]), int(p_n[4]))
        img_crop.show()

    return img_crop, pos_new

def xywh_to_xyxy(abs_list_of_coords, print_arr=False):

    i=0
    xyxy = []

    for coord in abs_list_of_coords:
        xyxy_1 = [0,0,0,0,0]

        x = abs_list_of_coords[i][1]
        y = abs_list_of_coords[i][2]
        w = abs_list_of_coords[i][3]
        h = abs_list_of_coords[i][4]

        x1 =  x - w/2
        x2 =  x + w/2
        y1 = y - h/2
        y2 = y + h/2

        xyxy_1[0] = abs_list_of_coords[i][0]
        xyxy_1[1] = x1
        xyxy_1[2] = y1
        xyxy_1[3] = x2
        xyxy_1[4] = y2

        xyxy.append(xyxy_1)

        i=i+1

    if print_arr==True:
        print()
        for coord in xyxy:
            print(coord)

    return xyxy

def coords_to_rel(list_of_coords, cropped_img):
    width = cropped_img.size[0]
    height = cropped_img.size[1]

    rel_list_of_coords = []
    rel_coords = [0,0,0,0,0]
    
    i=0
    for coords in list_of_coords:
       rel_coords = [0,0,0,0,0]
       rel_coords[0] = list_of_coords[i][0]
       rel_coords[1] = list_of_coords[i][1]/width
       rel_coords[2] = list_of_coords[i][2]/height
       rel_coords[3] = list_of_coords[i][3]/width
       rel_coords[4] = list_of_coords[i][4]/height
       rel_list_of_coords.append(rel_coords)
       i=i+1

    return rel_list_of_coords

def img_to_bw(img, pos, show=False):
    ### Check if image is of type numpy.ndarray, if not convert it
    if not isinstance(img, numpy.ndarray):
        img_arr = numpy.asarray(img)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    else: img_arr = img
    
    x1 = min(pos[0][1], pos[0][3], pos[1][1], pos[1][3], pos[2][1], pos[2][3])
    y1 = min(pos[0][2], pos[0][4], pos[1][2], pos[1][4], pos[2][2], pos[2][4])
    x2 = max(pos[0][1], pos[0][3], pos[1][1], pos[1][3], pos[2][1], pos[2][3])
    y2 = max(pos[0][2], pos[0][4], pos[1][2], pos[1][4], pos[2][2], pos[2][4])

    if pos[0][1] < pos[1][1]: x2 = x2 + (pos[1][1]-pos[0][1])
    if pos[1][2] > pos[2][2]: y2 = y2 + (pos[1][2]-pos[2][2])
    
    crop_img = img_arr[int(y1):int(y2),int(x1):int(x2)]
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) #converting cropped image to gray image

    average_value = numpy.mean(crop_img) #getting average value of a pixel in greyscale cropped qr code

    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY) #converting original image to gray image

    #converting image to black and white with threshold of average value of pixel in cropped img
    (thresh, bw_img) = cv2.threshold(img_arr, average_value, 255, cv2.THRESH_BINARY)


    bw_img = Image.fromarray(bw_img)

    if show:
        bw_img.show()

    return bw_img

def check_bounds(rel_coords):
    
    fixed_coords = []
    temp_coords = [0,0,0,0,0]
    
    i=0
    for coords in rel_coords:
       temp_coords = [0,0,0,0,0]
       j=0
       for coord in coords:
            if coord<0.0: temp_coords[j]=0.0
            elif coord>1.0: temp_coords[j]=1.0
            else: temp_coords[j] = coord

            j=j+1

       fixed_coords.append(temp_coords)
       i=i+1
       
    return fixed_coords

def xyxy_to_xywh(coords_xyxy, print_arr=False):
    
    i=0
    xywh = []

    for coord in coords_xyxy:
        xywh_1 = [0,0,0,0,0]

        x1 = coords_xyxy[i][1]
        y1 = coords_xyxy[i][2]
        x2 = coords_xyxy[i][3]
        y2 = coords_xyxy[i][4]

        x =  (x1+x2)/2
        y =  (y1+y2)/2
        w = abs(x1-x2)
        h = abs(y1-y2)

        xywh_1[0] = coords_xyxy[i][0]
        xywh_1[1] = x
        xywh_1[2] = y
        xywh_1[3] = w
        xywh_1[4] = h

        xywh.append(xywh_1)
        i=i+1


    if print_arr==True:
            print()
            for coord in xywh:
                print(coord)

    return xywh

# converts relative (0.0-0.1) coordinates [0,c,c,c,c] to absolute ones 
def coords_to_abs(list_of_coordinates, image_path, print_arr=False):
    img = Image.open(image_path)
    width = img.size[0]
    height = img.size[1]
    abs_list_of_coords = []
    abs_coords = [0,0,0,0,0]
    
    i=0
    for coords in list_of_coordinates:
       abs_coords = [0,0,0,0,0]
       abs_coords[0] = list_of_coordinates[i][0]
       abs_coords[1] = list_of_coordinates[i][1]*width
       abs_coords[2] = list_of_coordinates[i][2]*height
       abs_coords[3] = list_of_coordinates[i][3]*width
       abs_coords[4] = list_of_coordinates[i][4]*height
       abs_list_of_coords.append(abs_coords)
       i=i+1
    
    if print_arr==True:
        print()
        for coord in abs_list_of_coords:
            print(coord)

    img.close()
    return abs_list_of_coords

# same as coords_to_abs, but assumes [c,c,c,c] and not [0,c,c,c,c]
def coords_to_abs1(list_of_coordinates, image_path, print_arr=False):
    img = Image.open(image_path)
    width = img.size[0]
    height = img.size[1]
    abs_list_of_coords = []
    abs_coords = [0,0,0,0]
    
    i=0
    for coords in list_of_coordinates:
       abs_coords = [0,0,0,0]
       abs_coords[0] = list_of_coordinates[i][0]*width
       abs_coords[1] = list_of_coordinates[i][1]*height
       abs_coords[2] = list_of_coordinates[i][2]*width
       abs_coords[3] = list_of_coordinates[i][3]*height
       abs_list_of_coords.append(abs_coords)
       i=i+1
    
    if print_arr==True:
        print()
        for coord in abs_list_of_coords:
            print(coord)

    img.close()
    return abs_list_of_coords

#as coords_to_abs, but input is image instead of image_path
def coords_to_abs2(list_of_coordinates, img, print_arr=False): 

    width = img.size[0]
    height = img.size[1]
    abs_list_of_coords = []
    abs_coords = [0,0,0,0,0]
    
    i=0
    for coords in list_of_coordinates:
       abs_coords = [0,0,0,0,0]
       abs_coords[0] = list_of_coordinates[i][0]
       abs_coords[1] = list_of_coordinates[i][1]*width
       abs_coords[2] = list_of_coordinates[i][2]*height
       abs_coords[3] = list_of_coordinates[i][3]*width
       abs_coords[4] = list_of_coordinates[i][4]*height
       abs_list_of_coords.append(abs_coords)
       i=i+1
    
    if print_arr==True:
        print()
        for coord in abs_list_of_coords:
            print(coord)

    img.close()
    return abs_list_of_coords

# calculate distance between two 2D points
def dist_2D(x1,y1,x2,y2):
    distance = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)
    distance = sqrt(distance)
    return distance

def sort_pos_markers(list_of_coords, show):
    sorted_coords=[]

    if not (len(list_of_coords)==3): # if there aren't exactly 3 position markers, return False
        print("The number of detected position markers is: " + str(len(list_of_coords)))
        return False
    
    # points a b c in format [confidence, x1, y1, x2, y2]
    point_a = list_of_coords[0]
    point_b = list_of_coords[1]
    point_c = list_of_coords[2]


    # points in format [average x, average y]
    a=[(point_a[1]+point_a[3])/2,(point_a[2]+point_a[4])/2]
    b=[(point_b[1]+point_b[3])/2,(point_b[2]+point_b[4])/2]
    c=[(point_c[1]+point_c[3])/2,(point_c[2]+point_c[4])/2]
    centroid=[(a[0]+b[0]+c[0])/3,(a[1]+b[1]+c[1])/3]

    if show:
        print("Coordinates before sorting AAA:")
        print(a)
        print(b)
        print(c)

    # distances between points a b c
    dist_a_b = dist_2D(a[0], a[1], b[0], b[1])
    dist_a_c = dist_2D(a[0], a[1], c[0], c[1])
    dist_b_c = dist_2D(b[0], b[1], c[0], c[1])

    # find the longest distance → corner position mark = mark 2
    if (dist_a_b >= dist_a_c) and (dist_a_b >= dist_b_c):
        longest=dist_a_b # if the distance between a nd b is the longest → the diagonal
        mark_2 = c # then point c is the corner position mark → mark 2
        unknown1 = a
        unknown2 = b
    elif (dist_a_c >= dist_a_b) and (dist_a_c >= dist_b_c):
        longest=dist_a_c
        mark_2 = b
        unknown1 = a
        unknown2 = c
    else:
        longest=dist_b_c
        mark_2 = a
        unknown1 = b
        unknown2 = c
    
    # find in which rotation the qr code is based (360°/8 = 8 major rotations)
    # depending on the rotation assign the the position markers
    if unknown1[0] <= mark_2[0] :
        if unknown2[0] >= mark_2[0] :
            if mark_2[0] <= centroid[0]:
                if unknown1[1] <= unknown2[1]: mark_1 = unknown1
                else: mark_1 = unknown2
            else:
                if unknown1[1] >= unknown2[1]: mark_1 = unknown1
                else: mark_1 = unknown2
        else:
                if unknown1[1] >= unknown2[1]: mark_1 = unknown1
                else: mark_1 = unknown2
    else:
        if unknown2[0] <= mark_2[0]:
            if mark_2[0] >= centroid[0]:
                if unknown1[1] >= unknown2[1]: mark_1 = unknown1
                else: mark_1 = unknown2
            else:
                if unknown1[1] <= unknown2[1]: mark_1 = unknown1
                else: mark_1 = unknown2
        else:
                if unknown1[1] <= unknown2[1]: mark_1 = unknown1
                else: mark_1 = unknown2

    if mark_1 == unknown1 : mark_3 = unknown2
    else: mark_3 = unknown1
    
    # append the position marks to list sorted_coords in order mark_3, mark_2, mark_1
    if (mark_3 == a):
        sorted_coords.append(point_a)
        if (mark_2 == b):
            sorted_coords.append(point_b)
            sorted_coords.append(point_c)
        else:
            sorted_coords.append(point_c)
            sorted_coords.append(point_b)
    elif (mark_3 == b):
        sorted_coords.append(point_b)
        if (mark_2 == a):
            sorted_coords.append(point_a)
            sorted_coords.append(point_c)
        else:
            sorted_coords.append(point_c)
            sorted_coords.append(point_a)
    else:
        sorted_coords.append(point_c)
        if (mark_2 == a):
            sorted_coords.append(point_a)
            sorted_coords.append(point_b)
        else:
            sorted_coords.append(point_b)
            sorted_coords.append(point_a)

    if show:
        print("Coordinates after sorting:")
        for c in sorted_coords:
            print(c)

    # return the sorted position markers 
    return sorted_coords

# use YOLOv4 model to get position markers of qr code
def get_QR_pos(img, classes_path, config_path, weights_path, show):
    ### PATHS ####

    CONFIDENCE_THRESHOLD = 0.6
    NMS_THRESHOLD = 0.6 # Non-maximum Suppression threshold

    ### Check if image is of type numpy.ndarray, if not convert it
    img_arr = img
    if not isinstance(img, numpy.ndarray):
        img_arr = numpy.asarray(img)

    # read object category list
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # load model
    net = cv2.dnn.readNet(weights_path, config_path)

    # get the output layers from YOLO architecture for reading output predictions
    def get_output_layers(net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers

    # running inference on input frame
    def run_inference(image, show):
        
        Width = image.shape[1]
        Height = image.shape[0]

        blob = cv2.dnn.blobFromImage(image, 1/255, (416,416), (0,0,0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = numpy.argmax(scores)
                confidence = scores[class_id]
                if confidence > CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        
        return_boxes = [] # list of bounding boxes that is to be returned by funciton

        for i in indices:
            box = boxes[i]

            x = box[0] 
            y = box[1]
            w = box[2] 
            h = box[3] 

            return_box=[0,0,0,0,0] # iteam of return_boxes list
            return_box[0] = confidences[i] # index 0 = confidence
            return_box[1] = x # index 1 = x1
            return_box[2] = y # index 2 = y1
            return_box[3] = x+w # index 3 = x2
            return_box[4] = y+h # index 4 = y2

            if show:
                print("[INFO] detected {} with bbox {}".format(str(classes[class_ids[i]]),
                                        [[int(x),int(y)], [int(x+w),int(y+h)]]))
                draw_prediction(image, str(classes[class_ids[i]]), int(x),
                                int(y), int(x+w), int(y+h))
            
            return_boxes.append(return_box)
            
        return return_boxes
            
    points = run_inference(img_arr, show)
    points = sort_pos_markers(points, False)

    if show:
        cv2.imshow('Image', img_arr)
        cv2.waitKey(0)

    return points

# helper function for drawing bounding boxes on image (numpy array)
def draw_prediction(image, class_label, x1, y1, x2, y2):
    # convert img to an array (if it isn't one already)
    if isinstance(image, numpy.ndarray): image = image
    else:
        image = numpy.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    label = class_label
    color = (171,15,15)
    cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)
    cv2.putText(image, label, (x1-10,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.imshow('Image', image)
    cv2.waitKey(0)

# return base value of an angle
def angle_deviation(angle, dev_angle):
    deviation = (angle - dev_angle + 180) % 360 - 180
    deviation = abs(deviation)
    return deviation

def rotate_qr(img, pos_markers, show):
    
    # get position markers (calculate center of each of the three marks)
    pos_1 = [(pos_markers[0][1]+pos_markers[0][3])/2, (pos_markers[0][2]+pos_markers[0][4])/2]
    pos_2 = [(pos_markers[1][1]+pos_markers[1][3])/2, (pos_markers[1][2]+pos_markers[1][4])/2]
    pos_3 = [(pos_markers[2][1]+pos_markers[2][3])/2, (pos_markers[2][2]+pos_markers[2][4])/2]
    centroid = [(pos_1[0]+pos_2[0]+pos_3[0])/3, (pos_1[1]+pos_2[1]+pos_3[1])/3]

    # Calculate the angle of rotations (aligning x or y rotation between point 1-2 or 2-3)
    angle_1 = -(numpy.degrees(numpy.arctan2(pos_3[0]-pos_2[0], pos_3[1]-pos_2[1])))
    angle_2 = -(numpy.degrees(numpy.arctan2(pos_1[0]-pos_2[0], pos_1[1]-pos_2[1])))
    angle_3 = numpy.degrees(numpy.arctan2(pos_3[1]-pos_2[1], pos_3[0]-pos_2[0]))
    angle_4 = numpy.degrees(numpy.arctan2(pos_1[1]-pos_2[1], pos_1[0]-pos_2[0]))

    # Average the angle rotations
    angle_a = (angle_1 + angle_4)/2
    angle_b = (angle_2 + angle_3)/2

    if pos_2[0] > centroid[0]:
        if pos_1[1] >= pos_3[1] : angle = angle_a
        else: angle = angle_b
    else:
        if pos_1[1] >= pos_3[1] : angle = angle_b
        else: angle = angle_a

    if show:
        print("\nANGLE = ")
        print(angle)

    # convert img to an array (if it isn't one already)
    if isinstance(img, numpy.ndarray): img_arr = img
    else:
        img_arr = numpy.asarray(img)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

    # give image padding for rotation
    img = Image.fromarray(img_arr)
    width, height = img.size
    dimension = max(width, height)
    img_padding = Image.new(img.mode, (dimension, dimension), (255,255,255))
    img_padding.paste(img, ((int)((dimension-width)/2),(int)((dimension-height)/2)))
    img_arr = numpy.asarray(img_padding)

    # rotate the image using imutils function rotate
    img_rotated = imutils.rotate(img_arr, angle=angle)

    # display the rotated picture
    if(show):
        cv2.imshow('Image', img_rotated)
        cv2.waitKey(0)

    # convert rotated image back to type PIL image
    img_rotated = Image.fromarray(img_rotated)

    return img_rotated

def decode_qr(img):
    decoded_objects = decode(img) # Decode the QR code
    decoded_data = [obj.data.decode('utf-8') for obj in decoded_objects] # Extraxt only the decoded data
    print(decoded_data)

    
    

def img_to_bw_adaptive(img, show=False):
    ### Check if image is of type numpy.ndarray, if not convert it
    if not isinstance(img, numpy.ndarray):
        img_arr = numpy.asarray(img)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
    else: img_arr = img
    
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY) #converting original image to gray image

    bw_img = cv2.adaptiveThreshold(img_arr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, 2)

    bw_img = Image.fromarray(bw_img)

    if show:
        bw_img.show()

    return bw_img