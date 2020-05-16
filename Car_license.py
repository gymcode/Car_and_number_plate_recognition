# importing all the packages to be used
import cv2
import imutils
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ------------------------Image Detection------------------#
img_path = "car_images/licensed_car143.jpeg"

image = cv2.imread(img_path)
image = imutils.resize(image, width=400, height=400)

cv2.imshow('original image', image)
cv2.waitKey(0)


# function to show images based on title and image type
def show_function(title, images):
    img = cv2.UMat(images)
    cv2.imshow(title, img)


# changing image to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show_function("Grayscale Conversion", gray)
cv2.waitKey(0)

# filtering noises out of the image
gray_scale = cv2.bilateralFilter(gray, 11, 17, 17)
show_function("Bilateral Filter", gray_scale)
cv2.waitKey(0)

# finding the canny edges
edged = cv2.Canny(gray, 170, 200)
show_function("Canny Edges", edged)
cv2.waitKey(0)

cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(len(cnts))


image_copy = image.copy()
cv2.drawContours(image_copy, cnts, -1, (0, 255, 0), 3)
show_function("Showing all Contours", image_copy)
cv2.waitKey(0)

if len(cnts) < 60:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:len(cnts)]
else:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

print(len(cnts))
NumberPlateCnt = None

image_copy = image.copy()
cv2.drawContours(image_copy, cnts, -1, (0, 255, 0), 3)
show_function("Top " + str(len(cnts)) + " Contours", image_copy)
cv2.waitKey(0)

# ----------------Character Segmentation -------------------------#

idx = "plate"
for contours in cnts:
    perimeter = cv2.arcLength(contours, True)
    approx = cv2.approxPolyDP(contours, 0.02 * perimeter, True)
    if len(approx) == 4:
        NumberPlateCnt = approx

        x, y, w, h = cv2.boundingRect(contours)
        new_img = image[y:y + h, x: x + w]
        cv2.imwrite(idx + '.png', new_img)
        break


if NumberPlateCnt is None:
    print("No contour pertaining to the plate was found")
else:
    cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)
    show_function("Final; image with Number plate Detected", image)
    cv2.waitKey(0)

    License_Plate_path = "plate.png"
    License_Plate = cv2.imread(License_Plate_path)
    License_Plate = imutils.resize(License_Plate, width=200, height=200)

    # showing the license plate
    show_function("License Plate", License_Plate)

    # ---------------------Character recognition -----------------#

    text = pytesseract.image_to_string(License_Plate, lang="eng")
    print("The License Plate is: ", text)



