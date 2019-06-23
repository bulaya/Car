from hyperlpr import *
import sys
import importlib
import cv2
import tkinter as tk
from tkinter import filedialog


def selectPath():
    global path_
    path_ = filedialog.askopenfilename()
    path.set(path_)


def readpic():
    image = cv2.imread(path_)
    detect(image)
    print(HyperLPR_PlateRecogntion(image)[0][0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run():
    importlib.reload(sys)

    global path
    window = tk.Tk()
    window.title('车牌识别')

    fm1 = tk.Frame(window)
    fm2 = tk.Frame(window)
    fm3 = tk.Frame(window)

    path = tk.StringVar()

    Ltop = tk.Label(fm1, text="请选择图片路径")
    B1 = tk.Button(fm2, text="路径选择", command=selectPath)
    E1 = tk.Entry(fm2, textvariable=path, bd=5)
    B2 = tk.Button(fm2, text="确定", command=readpic)
    Lbot1 = tk.Label(fm3, text="学号：xxx 姓名：xxx")
    Lbot2 = tk.Label(fm3, text="学号：xxx 姓名：xxx")

    Ltop.pack(side=tk.TOP)
    B1.pack(side=tk.LEFT)
    E1.pack(side=tk.LEFT)
    B2.pack(side=tk.LEFT)
    Lbot1.pack(side=tk.BOTTOM)
    Lbot2.pack(side=tk.BOTTOM)

    fm1.pack(side=tk.TOP)
    fm2.pack(side=tk.TOP)
    fm3.pack(side=tk.TOP)

    sw = window.winfo_screenwidth()
    # 得到屏幕宽度
    sh = window.winfo_screenheight()
    # 得到屏幕高度
    ww = 300
    wh = 100
    # 窗口宽高为100
    x = (sw - ww) / 2
    y = (sh - wh) / 3
    window.geometry("%dx%d+%d+%d" % (ww, wh, x, y))

    window.mainloop()


def detect(image):
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
            cv2.imshow("plate", plate)
            cv2.rectangle(image, (x - 2, y - 2), (x + w + 2, y + h + 2), (255, 0, 0), 2)
    cv2.imshow("image", image)


if __name__ == '__main__':
    run()
