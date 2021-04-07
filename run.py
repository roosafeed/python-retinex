import os
import msr
import cv2


img_folder = "images"
out_folder = "output"

# constants
sig_list = [15, 80, 250]    #list of sigma
w = 1/3                     #weight
k = (0, 0)                  #kernel size
a = 125                     #alpha
G = 192
b = -30
bet = 46                    #beta

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

if not os.path.exists(img_folder):
    os.makedirs(img_folder)

img_list = os.listdir(img_folder)

if(len(img_list) == 0):
    print("The directory is empty. \nSave the input images to the 'images' folder.")
    exit()

for img in img_list:
    image = cv2.imread(os.path.join(img_folder, img))
    if image is None:
        # print(image)
        print(str(img) + " is not a supported file format")
        continue
    print(str(img) + "...")
    out = msr.MSRCR(image, sig_list, G, b, a, bet, w, k)
    loc = out_folder + "/msr_" + str(img)
    iw = cv2.imwrite(loc, out)
    if(iw):
        print("image saved to " + loc)
    cv2.imshow("Image", image)
    cv2.imshow("Retinex", out)
    cv2.waitKey()

cv2.destroyAllWindows()