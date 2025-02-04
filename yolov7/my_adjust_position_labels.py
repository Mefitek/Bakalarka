from my_det import my_detect
from my_functions import *
import glob # procházení adresářů

# formatting a list by throwing out every second entry in it (which is confidence stored separately)
def labels_cut_conf(labels):
    labels_2 = []
    i=0
    for l in labels:
        if(i%2==1):
            labels_2.append(labels[i])
        i=i+1
    return labels_2

# Get the global paths of all .jpg and .txt files, saving them as arrays
imgs = glob.glob(r"C:\Users\mefit\Desktop\BAK\yolov4\QR_pos_darknet_2\valid\*.jpg")
points_labels = glob.glob(r"C:\Users\mefit\Desktop\BAK\yolov4\QR_pos_darknet_2\valid\*.txt")

n=0
for img1 in imgs:

    img = imgs[n]
    points_label = points_labels[n]
    print("Obrazek:\t" + str(img))
    print("Textak: \t" + str(points_label))

    ###########################
    # QR code detection
    ###########################

    labels = my_detect(img) # array of qr_code labels
    crops = crop_by_labels(labels, img, False, False) # array of cropped images of qrcode labels
    labels = labels_cut_conf(labels) # removing confidence labels
    labels = coords_to_abs1(labels, img, False) # convert coordinates to absolutes

    ###########################
    # Process the QR Position markers
    ###########################

    coords = read_label(points_label, False) # read the coordinates from .txt file and save them as array
    abs_coords = coords_to_abs(coords, img, False) # convert coordinates to absolutes
    coords_xyxy = xywh_to_xyxy(abs_coords, False) # convert coordinated to xyxy format

    ###########################
    # Move the Positional markers
    ###########################

    if(len(labels)==1): # if there's more than 1 qr code it's hard to say which position marker's belong to it
        if(len(coords_xyxy)<=3): # if there are not exactly 3 position marker's, the same problem as above applies
            # reading coordinates of detected QR code
            x1 = labels[0][0]
            y1 = labels[0][1]
            x2 = labels[0][2]
            y2 = labels[0][3]

            print(str(x1) +" "+ str(y1) +" "+ str(x2) +" "+ str(y2))
            # substracting detected coordinates from the position marker coordinates
            coords_xyxy2 = []

            i = 0
            for coord in coords_xyxy:
                temp_coord=[0,0,0,0,0]
                temp_coord[1] = coord[1] - x1
                temp_coord[2] = coord[2] - y1
                temp_coord[3] = coord[3] - x1
                temp_coord[4] = coord[4] - y1
                coords_xyxy2.append(temp_coord)
                i=i+1
            
            coords_rel = coords_to_rel(coords_xyxy2, crops[0]) # convert coordinates to relative ones
            coords_rel_fix = check_bounds(coords_rel)
            coords_rel_fix = xyxy_to_xywh(coords_rel_fix)

            path = "C:\\Users\\mefit\\Downloads\\testik\\"
            image_name = "qr_pos_" + str(n+1120) + ".jpg"
            (crops[0]).save(path+image_name)

            label_name = "qr_pos_" + str(n+1120) + ".txt"

            with open((path+label_name), 'w') as f:
                iter = 0
                for coord in coords_rel_fix:
                    label_txt = "0 " +str(coord[1]) + " " + str(coord[2]) + " " + str(coord[3]) + " " + str(coord[4])
                    f.write(label_txt)
                    if iter<2:
                        f.write("\n")
                    iter=iter+1
            f.close()

    print("\n" + str(n) + "\n")
    n=n+1