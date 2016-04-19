import cv2


def showImage(
        img,
        namedWindow="",
        width=640, height=480
):
    cv2.namedWindow(namedWindow, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(namedWindow, width, height)
    cv2.imshow(namedWindow, img)


image = cv2.imread("Page_09_Pattern_24.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale

showImage(gray, 'gray')

_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)  # threshold

showImage(thresh, 'thresh')

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

dilated = cv2.dilate(thresh, kernel, iterations=0)  # dilate

showImage(dilated, 'dilated')

_, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # get contours

# for each contour found, draw a rectangle around it on original image
for contour in contours:
    # get rectangle bounding contour
    [x, y, w, h] = cv2.boundingRect(contour)

    # discard areas that are too large
    # if h > 300 and w > 300:
    #     continue

    # discard areas that are too small
    if h < 40 or w < 40:
        continue

    # draw rectangle around contour on original image
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)

showImage(image, 'Extract Object')

while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
