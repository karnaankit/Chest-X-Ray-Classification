from numpy import *
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('darkgrid')
import util


from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model


from tkinter import *
from PIL import ImageTk, Image  
from tkinter import filedialog 
from tkinter.ttk import * 

labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']

freq_pos=np.array([0.02406894, 0.0212051,  0.1201283,  0.00201105, 0.04108648, 0.05422193,
 0.17651401, 0.01266451, 0.04798513, 0.05641117, 0.03028027, 0.01471374,
 0.10424357, 0.02074689])
freq_neg=np.array([0.97593106, 0.9787949,  0.8798717,  0.99798895, 0.95891352, 0.94577807,
 0.82348599, 0.98733549, 0.95201487, 0.94358883, 0.96971973, 0.98528626,
 0.89575643, 0.97925311])

pos_weights = freq_neg
neg_weights = freq_pos


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    def weighted_loss(y_true, y_pred):
        loss = 0.0
        
        for i in range(len(pos_weights)):
            loss_pos = -1 * K.mean(pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon))
            loss_neg = -1 * K.mean(neg_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon))
            loss += loss_pos + loss_neg
        return loss
    return weighted_loss


base_model = DenseNet121(weights='imagenet_weights_no_top.h5', include_top = False)

x = base_model.output

x = GlobalAveragePooling2D()(x)

predictions = Dense(len(labels), activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights),metrics=['accuracy'])

model.load_weights("test.h5")
df = pd.read_csv("train-small.csv")


#UI start

def read_image(fname):    
    file = open(fname,"r")
    f = file.read()
    file.close() 
    return f

#detect


def callback():   
  try:
    filename = filedialog.askopenfilename(title ='Upload X-Ray Image')

    #filename and DIR name
    IMAGE_DIR = filename[:-17]
    image_name = filename[-17:]
    
    #outputs
    labels_to_show = labels[:4]
    util.compute_gradcam(model, image_name, IMAGE_DIR, df, labels, labels_to_show, run_number=1)
    
    labels_to_show =labels[4:8]
    util.compute_gradcam(model, image_name, IMAGE_DIR, df, labels, labels_to_show, run_number=2)
    
    labels_to_show =labels[8:12]
    util.compute_gradcam(model, image_name, IMAGE_DIR, df, labels, labels_to_show, run_number=3)
    
    labels_to_show =labels[12:]
    util.compute_gradcam(model, image_name, IMAGE_DIR, df, labels, labels_to_show, run_number=4)
    try:
        output_root = Toplevel()
        output_root.geometry('946x768')

        # inside frame1
        img1 = Image.open('output_image1.png')
        width1, height1 = img1.size
        width_new1 = int(width1/1.25)
        height_new1 = int(height1/1.25)
        img_resized1 = img1.resize((width_new1,height_new1))
        i1 = ImageTk.PhotoImage(img_resized1)
        b1 = Label(output_root, image=i1)
        b1.grid(row=0, column=0)

        img2 = Image.open('output_image2.png')
        width2, height2 = img2.size
        width_new2 = int(width2/1.25)
        height_new2 = int(height2/1.25)
        img_resized2 = img2.resize((width_new2,height_new2))
        i2 = ImageTk.PhotoImage(img_resized2)
        b2 = Label(output_root, image=i2)
        b2.grid(row=1, column=0)

        img3 = Image.open('output_image3.png')
        width3, height3 = img3.size
        width_new3 = int(width3/1.25)
        height_new3 = int(height3/1.25)
        img_resized3 = img3.resize((width_new3,height_new3))
        i3 = ImageTk.PhotoImage(img_resized3)
        b3 = Label(output_root, image=i3)
        b3.grid(row=2, column=0)

        img4 = Image.open('output_image4.png')
        width4, height4=img4.size
        width_new4 = int(width4/1.25)
        height_new4 = int(height4/1.25)
        img_resized4 = img4.resize((width_new4, height_new4))
        i4 = ImageTk.PhotoImage(img_resized4)
        b4 = Label(output_root, image=i4)
        b4.grid(row=3, column=0)

        output_root.mainloop()

    except:
      print('Exception Occured!!')
    
  except:
    print("Exception Occured!!")


root = Tk() 
root.title("X-Ray Classification")
background = PhotoImage(file="back.png")
Label(root, image=background).place(x=0, y=0)
root.geometry("480x459")

#detect button
detect = PhotoImage(file=r"ui_classify.png")
Button(root, text="detect", image=detect, command=callback).place(x=150, y=350)



root.mainloop()
