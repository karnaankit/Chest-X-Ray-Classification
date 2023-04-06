import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras import backend as K
from tensorflow.keras.layers import  Dense
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.models import Model
import util
import streamlit as st

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
        # initialize loss to zero
        loss = 0.0
        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class
            loss_pos = -1 * K.mean(pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon))
            loss_neg = -1 * K.mean(neg_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon))
            loss += loss_pos + loss_neg

        return loss

    return weighted_loss


base_model = DenseNet121(weights='imagenet_weights_no_top.h5', include_top = False)

x = base_model.output

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)

# and a logistic layer
predictions = Dense(len(labels), activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights),metrics=['accuracy'])
model.load_weights("finalized.h5")
df = pd.read_csv("train-small.csv")
IMAGE_DIR = "nih/images-small/"

def main():
    st.title("Chest X-Ray Image Classification")
    uploaded_files = st.file_uploader("Choose X-Ray image", type=["png"], accept_multiple_files=False)
    try:
        if uploaded_files:
            name = uploaded_files.name
            st.write("Classifying...")
            labels_to_show = labels[:4]
            util.compute_gradcam(model, name, IMAGE_DIR, df, labels, labels_to_show)
            st.image('output_image.png', use_column_width=True)
            labels_to_show = labels[4:8]
            util.compute_gradcam(model, name, IMAGE_DIR, df, labels, labels_to_show)
            st.image('output_image.png', use_column_width=True)
            labels_to_show = labels[8:12]
            util.compute_gradcam(model, name, IMAGE_DIR, df, labels, labels_to_show)
            st.image('output_image.png', use_column_width=True)
            labels_to_show = labels[12:]
            util.compute_gradcam(model, name, IMAGE_DIR, df, labels, labels_to_show)
            st.image('output_image.png', use_column_width=True)
            st.write("Classification Complete")
    except Exception as e:
        st.write("Error: ", e)


if __name__ == "__main__":
    main()
