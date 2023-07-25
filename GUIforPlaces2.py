# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import Recognize_Program

image = None

def open_image():
    global image
    # 打开文件选择器
    file_path = filedialog.askopenfilename()
    # 加载图像并在标签中显示
    image = Image.open(file_path)

    # 将图像限制在一定大小
    max_size = 500
    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_width = int(width * (max_size / height))
            new_height = max_size
        image = image.resize((new_width, new_height), Image.ANTIALIAS)

    photo = ImageTk.PhotoImage(image)
    pic_label.configure(image=photo)
    pic_label.image = photo

    # 清除上一张图片的识别结果
    result_label0.config(text='')
    result_label1.config(text='')
    result_label2.config(text='')
    result_label3.config(text='')
    result_label4.config(text='')

def recognize_image():
    # 执行图像识别的程序
    result0,result1,result2,result3,result4 = Recognize_Program.recognize(image)
    result_label0.config(text=result0)
    result_label1.config(text=result1)
    result_label2.config(text=result2)
    result_label3.config(text=result3)
    result_label4.config(text=result4)

# 主窗口
root = Tk()
root.title("Image Recognition System")
root.geometry("800x700+50+50")

# 导入图像按钮
button = tk.Button(root, text="Choose Image", command=open_image)
button.pack()

# 执行识别按钮
recognize_button = tk.Button(root, text="Recognize", command=recognize_image)
recognize_button.pack()

# 图像标签
pic_label = tk.Label(root)
pic_label.pack()
# 识别结果标签
result_label0 = tk.Label(root)
result_label0.pack()
result_label1 = tk.Label(root)
result_label1.pack()
result_label2 = tk.Label(root)
result_label2.pack()
result_label3 = tk.Label(root)
result_label3.pack()
result_label4 = tk.Label(root)
result_label4.pack()

# 运行应用程序的主循环
root.mainloop()

