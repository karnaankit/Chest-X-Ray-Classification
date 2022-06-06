from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tkinter.ttk import *


def read_image(fname):
    file = open(fname, "r")
    f = file.read()
    file.close()
    return f


# detect


def callback():
    try:
        filename = filedialog.askopenfilename(title='Upload X-Ray Image')

        # filename and DIR name
        IMAGE_DIR = filename[:-17]
        image_name = filename[-17:]

        # outputs
        # labels_to_show = labels[:4]
        # util.compute_gradcam(model, image_name, IMAGE_DIR, df, labels, labels_to_show, run_number=1)
        #
        # labels_to_show = labels[4:8]
        # util.compute_gradcam(model, image_name, IMAGE_DIR, df, labels, labels_to_show, run_number=2)
        #
        # labels_to_show = labels[8:12]
        # util.compute_gradcam(model, image_name, IMAGE_DIR, df, labels, labels_to_show, run_number=3)
        #
        # labels_to_show = labels[12:]
        # util.compute_gradcam(model, image_name, IMAGE_DIR, df, labels, labels_to_show, run_number=4)
        # try:
        #     output_root = Toplevel()
        #     output_root.geometry('946x768')
        #
        #     # inside frame1
        #     img1 = Image.open('output_image1.png')
        #     width1, height1 = img1.size
        #     width_new1 = int(width1 / 1.25)
        #     height_new1 = int(height1 / 1.25)
        #     img_resized1 = img1.resize((width_new1, height_new1))
        #     i1 = ImageTk.PhotoImage(img_resized1)
        #     b1 = Label(output_root, image=i1)
        #     b1.grid(row=0, column=0)
        #
        #     img2 = Image.open('output_image2.png')
        #     width2, height2 = img2.size
        #     width_new2 = int(width2 / 1.25)
        #     height_new2 = int(height2 / 1.25)
        #     img_resized2 = img2.resize((width_new2, height_new2))
        #     i2 = ImageTk.PhotoImage(img_resized2)
        #     b2 = Label(output_root, image=i2)
        #     b2.grid(row=1, column=0)
        #
        #     img3 = Image.open('output_image3.png')
        #     width3, height3 = img3.size
        #     width_new3 = int(width3 / 1.25)
        #     height_new3 = int(height3 / 1.25)
        #     img_resized3 = img3.resize((width_new3, height_new3))
        #     i3 = ImageTk.PhotoImage(img_resized3)
        #     b3 = Label(output_root, image=i3)
        #     b3.grid(row=2, column=0)
        #
        #     img4 = Image.open('output_image4.png')
        #     width4, height4 = img4.size
        #     width_new4 = int(width4 / 1.25)
        #     height_new4 = int(height4 / 1.25)
        #     img_resized4 = img4.resize((width_new4, height_new4))
        #     i4 = ImageTk.PhotoImage(img_resized4)
        #     b4 = Label(output_root, image=i4)
        #     b4.grid(row=3, column=0)
        #
        #     output_root.mainloop()
    except:
        pass



root = Tk()
root.title("X-Ray Classification")
background = PhotoImage(file="back.png")
Label(root, image=background).place(x=0, y=0)
root.geometry("480x459")
# root.configure(background='black')

# detect button
detect = PhotoImage(file=r"ui_classify.png")
Button(root, text="detect", image=detect, command=callback).place(x=150, y=350)

# close button
# close = PhotoImage(file = r"ui_close.png")
# Button(root, text = "close",image = close, command = root.destroy).place(x=0,y=0)

root.mainloop()
