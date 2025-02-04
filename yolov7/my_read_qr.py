from my_det import my_detect
import my_paths
from my_functions import *

#HERE STARTS THE MAIN PROGRAM
#img_path = my_paths.IMG_PATH8
img_path = r"C:\Users\mefit\Downloads\testik.jpg"
print("Processing image: " +str(img_path))

labels = my_detect(img_path) # array of qr_code labels

crops = crop_by_labels(labels, img_path, add_quiet=True, show=False) # array of cropped images of qrcode labels

print("\nQR codes detected:")
print(str(len(crops)))
print()

i=0
for qr_code in crops: # for each qr code detected and cropped in the picture
    qr = qr_code
    i=i+1
    print("Now processing QR code number " + str(i) + ":")

    points = get_QR_pos(qr_code, my_paths.v4_CLASSES_PATH, my_paths.v4_CONFIG, my_paths.v4_WEIGHTS, show=False)
    if isinstance(points, bool): continue # if the points variable is False, go to next cycle (there were not exactly 3 position markers)
   
    # sort position markers list in order 1 2 3 (1=down left marker, 2=upper left, 3=upper right)
    points = sort_pos_markers(points, show=False)
    if isinstance(points, bool): continue

    # image rotations
    #qr = rotate_qr(qr_code, points, show=False)

    # after image rotation get new position markers
    points = get_QR_pos(qr, my_paths.v4_CLASSES_PATH, my_paths.v4_CONFIG, my_paths.v4_WEIGHTS, show=False)
    if isinstance(points, bool): continue
    points = sort_pos_markers(points, show=False)
    if isinstance(points, bool): continue

    # crop the image using crop_by_pos()
    qr, points = crop_by_pos(qr, points, add_quiet=True, show=False)

    qr_adapt = img_to_bw_adaptive(qr, show=False)
    qr = img_to_bw(qr, points, show=False)
    
    print("\nReading Raw image:")
    decode_qr(Image.open(img_path))

    # docode using Pyzbar library
    print("\nReading transformed qr code: (Binarization with average threshold)")
    decode_qr(qr)
    print("\nReading transformed qr code: (Binarization with adaptive threshold)")
    decode_qr(qr_adapt)

    #qr.show()
    qr_adapt.show()

    print("\n")

    