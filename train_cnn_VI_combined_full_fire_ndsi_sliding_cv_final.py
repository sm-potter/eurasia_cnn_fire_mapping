#!/usr/bin/env python
# coding: utf-8

# Import necessary packages
from __future__ import division
import pandas as pd
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.python.lib.io import file_io
from tensorflow.python.keras.optimizer_v2.adam import Adam
import os
import segmentation_models as sm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate, Conv2DTranspose, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Input, AvgPool2D
from tensorflow.keras.models import Model
from keras_unet_collection import models
import geopandas as gpd
# import tensorflow_addons as tfa
import logging
import time


# Record the start time
start_time = time.time()

# Set the fold number
fold = 0

# Load min-max normalization values
min_max = pd.read_csv("/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/l8_sent_collection2_global_min_max_cutoff_proj.csv").reset_index(drop=True)
min_max = min_max[['6', '7', '8']]
print(min_max)

# Function to normalize data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

def normalize_meanstd(a, axis=None): 
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std

# Function to get file paths based on AOI and ID
def get_file_paths(df):
    file_paths = []
    for aoi in df['AOI'].unique():
        ids = df[df['AOI'] == aoi]['ID'].tolist()
        if aoi == 'anna':
            base_dir = '/explore/nobackup/people/spotter5/cnn_mapping/Russia/anna_monthly_ndsi_sliding_subs_0_128/'
        elif aoi == 'NBAC':
            base_dir = '/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/nbac_monthly_ndsi_sliding_subs_0_128/'
        elif aoi == 'MTBS':
            base_dir = '/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/mtbs_monthly_ndsi_sliding_subs_0_128/'
        else:
            continue  # Skip unknown AOI

        # List all files in the base_dir
        all_files = os.listdir(base_dir)

        # Filter files whose IDs are in ids
        filtered_files = [
            os.path.join(base_dir, name) for name in all_files 
            if int(name.split('_')[-1].split('.')[0]) in ids
        ]
        file_paths.extend(filtered_files)
    return file_paths

# Read training, validation, and testing dataframes for each AOI
# For Eurasia (anna)
training_df_anna = pd.read_csv(f'/explore/nobackup/people/spotter5/cnn_mapping/Russia/train_fold_{fold}.csv')
validation_df_anna = pd.read_csv(f'/explore/nobackup/people/spotter5/cnn_mapping/Russia/val_fold_{fold}.csv')
testing_df_anna = pd.read_csv(f'/explore/nobackup/people/spotter5/cnn_mapping/Russia/test_fold_{fold}.csv')

training_df_anna['AOI'] = 'anna'
validation_df_anna['AOI'] = 'anna'
testing_df_anna['AOI'] = 'anna'

# For NBAC and MTBS
training_df_nbac = pd.read_csv(f'/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/train_fold_{fold}.csv')
validation_df_nbac = pd.read_csv(f'/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/val_fold_{fold}.csv')
testing_df_nbac = pd.read_csv(f'/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/test_fold_{fold}.csv')

# Combine dataframes
training_df = pd.concat([training_df_anna, training_df_nbac], ignore_index=True)
validation_df = pd.concat([validation_df_anna, validation_df_nbac], ignore_index=True)
testing_df = pd.concat([testing_df_anna, testing_df_nbac], ignore_index=True)

# Create a unique identifier by combining AOI and ID
training_df['unique_id'] = training_df['AOI'].astype(str) + '_' + training_df['ID'].astype(str)
validation_df['unique_id'] = validation_df['AOI'].astype(str) + '_' + validation_df['ID'].astype(str)
testing_df['unique_id'] = testing_df['AOI'].astype(str) + '_' + testing_df['ID'].astype(str)

# Ensure there are no duplicates in each set
assert not training_df['unique_id'].duplicated().any(), "Duplicates found in training data."
assert not validation_df['unique_id'].duplicated().any(), "Duplicates found in validation data."
assert not testing_df['unique_id'].duplicated().any(), "Duplicates found in testing data."

# Ensure there is no overlap between the sets
combined_df = pd.concat([training_df, validation_df, testing_df], ignore_index=True)
duplicates = combined_df[combined_df.duplicated(subset=['unique_id'], keep=False)]
if not duplicates.empty:
    print("Overlap found between the datasets:")
    print(duplicates)
else:
    print("No overlap found between training, validation, and testing datasets.")

# Get file paths for training, validation, and testing
training_names = get_file_paths(training_df)
validation_names = get_file_paths(validation_df)
testing_names = get_file_paths(testing_df)


# Function to normalize images using min-max scaler
def normalize_image(img):
    img_shape = img.shape
    img = img.reshape(-1, img.shape[2])
    img = pd.DataFrame(img)
    img.columns = min_max.columns
    img = pd.concat([min_max, img]).reset_index(drop=True)
    img = pd.DataFrame(scaler.fit_transform(img))
    img = img.iloc[2:]
    img = img.values.reshape(img_shape)
    img[np.isnan(img)] = -1
    img = np.round(img, 3)
    return img

# Image generator class
class img_gen(tf.keras.utils.Sequence):
    def __init__(self, batch_size, img_size, input_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        import random
        random.shuffle(self.input_img_paths)
        i = idx * self.batch_size
        batch_img_paths = self.input_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size, dtype="uint8")

        for j, path in enumerate(batch_img_paths):
            # Load image
            img = np.round(np.load(path), 3)
            if img.shape[2] == 4:
                img = img[:, :, :-1]
            else:
                img = img[:, :, 6:9]
            img = img.astype(float)
            img = np.round(img, 3)
            img[img == 0] = -999
            img[np.isnan(img)] = -999
            img[img == -999] = np.nan

            # Normalize image
            img = normalize_image(img)
            x[j] = img

            # Load mask
            mask = np.round(np.load(path), 3)[:, :, -1]
            mask = mask.astype(int)
            mask[mask < 0] = 0
            mask[mask > 1] = 0
            mask[~np.isin(mask, [0, 1])] = 0
            mask[np.isnan(mask)] = 0
            y[j] = mask.astype(int)

        return x, y

# Initialize GPUs with TensorFlow
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Set batch size and image size
BATCH_SIZE = 45
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: %d' % strategy.num_replicas_in_sync) 
batch_size = BATCH_SIZE * strategy.num_replicas_in_sync
img_size = (128, 128)
num_classes = 1

# Get image generators
train_gen = img_gen(batch_size, img_size, training_names)
val_gen = img_gen(batch_size, img_size, validation_names)
test_gen = img_gen(batch_size, img_size, testing_names)

# Clear any previous models
tf.keras.backend.clear_session()

# Define optimizer and loss
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss = tf.keras.losses.BinaryFocalCrossentropy(from_logits=False, gamma=2.0, alpha=0.25)

# Define callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=f"/explore/nobackup/people/spotter5/cnn_mapping/Russia/models/combined_good_ndsi_sliding_{fold}_final",
        save_weights_only=False,
        save_best_only=True,
        monitor='val_unet_output_final_activation_iou_score',
        mode='max'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_unet_output_final_activation_iou_score', 
        mode='max',  
        patience=20
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_unet_output_final_activation_loss', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6, 
        verbose=1
    )
]

# Open a strategy scope and compile the model
with strategy.scope():
    model_unet_from_scratch = models.unet_plus_2d(
        (None, None, 3), 
        filter_num=[16,32,64,128], 
        n_labels=num_classes, 
        stack_num_down=2, 
        stack_num_up=2, 
        activation='ReLU', 
        output_activation='Sigmoid', 
        batch_norm=True, 
        pool=False, 
        unpool=False, 
        backbone='EfficientNetB7', 
        weights=None, 
        freeze_backbone=False, 
        freeze_batch_norm=False, 
        deep_supervision=True,
        name='unet'
    )
    model_unet_from_scratch.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[
            sm.metrics.Precision(threshold=0.5),
            sm.metrics.Recall(threshold=0.5),
            sm.metrics.FScore(threshold=0.5), 
            sm.metrics.IOUScore(threshold=0.5),
            'accuracy'
        ]
    )

# Fit the model
history = model_unet_from_scratch.fit(
    train_gen,
    epochs=50,
    callbacks=callbacks,
    validation_data=val_gen,
    verbose=0
)

# Save the model
model_unet_from_scratch.save(f"/explore/nobackup/people/spotter5/cnn_mapping/Russia/models/combined_good_ndsi_sliding_{fold}_final.tf")

# Save training history
history_dict = history.history
result = pd.DataFrame({
    'Precision': history_dict["unet_output_final_activation_precision"],
    'Val_Precision': history_dict['val_unet_output_final_activation_precision'],
    'Recall': history_dict["unet_output_final_activation_recall"],
    'Val_Recall': history_dict['val_unet_output_final_activation_recall'],
    'F1': history_dict["unet_output_final_activation_f1-score"],
    'Val_F1': history_dict['val_unet_output_final_activation_f1-score'],
    'IOU': history_dict["unet_output_final_activation_iou_score"],
    'Val_IOU': history_dict['val_unet_output_final_activation_iou_score'],
    'Loss': history_dict['unet_output_final_activation_loss'],
    'Val_Loss': history_dict['val_unet_output_final_activation_loss'],
    'Accuracy': history_dict['unet_output_final_activation_accuracy'],
    'Val_Accuracy': history_dict['val_unet_output_final_activation_accuracy']
})
result.to_csv(f"/explore/nobackup/people/spotter5/cnn_mapping/Russia/combined_good_ndsi_sliding_{fold}_final.csv")

# Record the end time and print the time taken
end_time = time.time()
time_difference_seconds = end_time - start_time
time_difference_hours = time_difference_seconds / 3600
print(f"Time taken: {time_difference_hours:.2f} hours")
