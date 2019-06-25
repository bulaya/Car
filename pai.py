import io

from hyperlpr import *
import sys
import importlib
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog


class Car:
    def __init__(self):
        self.flag = 0
        self.path = ''
        self.path_ = ''
        self.window = tk.Tk()
        self.window.title('车牌识别')
        self.label_img = tk.Label(self.window)
        fm1 = tk.Frame(self.window)
        self.fm2 = tk.Frame(self.window)
        fm3 = tk.Frame(self.window)

        self.path = tk.StringVar()

        Ltop = tk.Label(fm1, text="请选择图片路径")
        B1 = tk.Button(self.fm2, text="路径选择", command=self.selectPath)
        self.E1 = tk.Entry(self.fm2, textvariable=self.path, bd=5)
        B2 = tk.Button(self.fm2, text="确定", command=self.readpic)
        self.Lbot1 = tk.Label(fm3, text="车牌：")

        Ltop.pack(side=tk.TOP)
        B1.pack(side=tk.LEFT)
        self.E1.pack(side=tk.LEFT)
        B2.pack(side=tk.LEFT)
        self.Lbot1.pack(side=tk.BOTTOM)

        fm1.pack(side=tk.TOP)
        self.fm2.pack(side=tk.TOP)
        fm3.pack(side=tk.TOP)
        self.label_img.pack()

    def selectPath(self):
        self.path_ = filedialog.askopenfilename()
        self.path.set(self.path_)

    def readpic(self):
        image = cv2.imread(self.path_)
        plate = self.detect(image)
        self.Lbot1['text'] = HyperLPR_PlateRecogntion(image)[0][0]
        cv2.imwrite('img.jpg', plate, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

        img = Image.open('img.jpg', 'r')
        global ph
        ph = ImageTk.PhotoImage(img)
        self.label_img["image"]=ph


        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def run(self):
        importlib.reload(sys)
        self.window.mainloop()

    def detect(self, image):
        # 定义分类器
        cascade_path = 'cascade.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        # 修改图片大小
        resize_h = 600
        height = image.shape[0]
        scale = image.shape[1] / float(height)
        image = cv2.resize(image, (int(scale * resize_h), resize_h))
        # 转为灰度图
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 车牌定位
        car_plates = cascade.detectMultiScale(image_gray, 1.1, 2, minSize=(36, 9), maxSize=(36 * 40, 9 * 40))
        print("检测到车牌数", len(car_plates))
        if len(car_plates) > 0:
            for car_plate in car_plates:
                x, y, w, h = car_plate
                plate = image[y - 2: y + h + 2, x - 2: x + w + 2]
                # cv2.imshow("plate", plate)
                cv2.rectangle(image, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 0, 0), 2)
        # cv2.imshow("image", image)
        return plate


if __name__ == '__main__':
    car = Car()
    car.run()
