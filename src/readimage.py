import cv2
import pytesseract
import numpy as np


def extract_data(img):
    line_data = []

    img_height = img.shape[0]
    img_width = img.shape[1]

    height = int(img_height * (49 / 528))
    width = int(img_width * (267 / 1160))

    num_width = int(img_width * (73 / 1160))
    last_col_width = int(img_width * (86 / 1160))

    pNamesImg = img[height:-1, :width]

    boxes = pytesseract.image_to_data(pNamesImg, config='--psm 6')

    for count, data in enumerate(boxes.splitlines()):
        if count > 0:
            data = data.split()
            if len(data) == 12:
                print(data[11])
                x, y, w, h, content = int(data[6]), int(data[7]), int(data[8]), int(data[9]), data[11]
                cv2.rectangle(img, (x, height + y), (w + x, height + h + y), (0, 255, 0), 1)
                line_data.append([content])

    num_img = img[height:-1, width:-last_col_width]

    conf = "--psm 7 digits"

    print(num_img.shape)
    print(height)
    print(num_width)

    numbers = []
    for row in range(0, num_img.shape[0], height):
        for col in range(0, num_img.shape[1], num_width):

            single_num_img = num_img[row:row + height, col:col + num_width]

            x_off, y_off = col, row

            boxes = pytesseract.image_to_data(single_num_img, config=conf)

            for count, data in enumerate(boxes.splitlines()):
                if count > 0:
                    data = data.split()
                    if len(data) == 12:
                        # print(data[11])
                        x, y, w, h, content = int(data[6]), int(data[7]), int(data[8]), int(data[9]), data[11]
                        cv2.rectangle(img,
                                      (x_off + width + x, y_off + height + y),
                                      (x_off + width + w + x, y_off + height + h + y),
                                      (0, 255, 0),
                                      1)
                        cv2.putText(img, content, (x_off + width, y_off + height),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                        numbers.append(content)

    numbers = np.array(numbers)
    numbers = np.reshape(numbers, (-1, 11))
    print(numbers)
    for i, x in enumerate(numbers):
        line_data[i] += numbers[i].tolist()
    print(line_data)

    return img


def img_preparation(img):
    img = cv2.pyrUp(img)
    img = cv2.medianBlur(img, 5)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 150])
    upper_white = np.array([0, 0, 255])

    mask = cv2.inRange(img, lower_white, upper_white)
    img = cv2.bitwise_and(img, img, mask=mask)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 13, 2)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.pyrDown(img)

    return img


def main():
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    img = cv2.imread(r".\..\target.png")

    img = img_preparation(img)

    img = extract_data(img)

    cv2.imshow("Extracted box", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
