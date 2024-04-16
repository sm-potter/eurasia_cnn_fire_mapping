import pandas as pd
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.python.lib.io import file_io
from tensorflow.python.keras.optimizer_v2.adam import Adam
import os
import segmentation_models as sm
import matplotlib.pyplot as plt
import numpy as np
#from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate, Conv2DTranspose, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Input, AvgPool2D
from tensorflow.keras.models import Model
from keras_unet_collection import models
import tensorflow_addons as tfa
import geopandas as gpd
import logging
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

min_max_vi = pd.read_csv("/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/l8_sent_collection2_global_min_max_cutoff_proj.csv").reset_index(drop = True)
min_max_vi = min_max_vi[['6', '7', '8']]

class img_gen_vi(tensorflow.keras.utils.Sequence):

    """Helper to iterate over the data (as Numpy arrays).
    Inputs are batch size, the image size, the input paths (x) and target paths (y)
    """

    #will need pre defined variables batch_size, img_size, input_img_paths and target_img_paths
    def __init__(self, batch_size, img_size, input_img_paths):
	    self.batch_size = batch_size
	    self.img_size = img_size
	    self.input_img_paths = input_img_paths
	    self.target_img_paths = input_img_paths

    #number of batches the generator is supposed to produceis the length of the paths divided by the batch siize
    def __len__(self):
	    return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_img_paths = self.input_img_paths[i : i + self.batch_size] #for a given index get the input batch pathways (x)
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size] #for a given index get the input batch pathways (y)
		
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32") #create matrix of zeros which will have the dimension height, wideth, n_bands), 8 is the n_bands
        
  
         #start populating x by enumerating over the input img paths
        for j, path in enumerate(batch_img_paths):

           #load image
            img =  np.round(np.load(path), 3)
            
            if img.shape[2] == 4:
                
                img = img[:, :, :-1]

            else:
                
                img = img[:, :, 6:9]

            # img = img * 1000
            img = img.astype(float)
            img = np.round(img, 3)
            img[img == 0] = -999

            img[np.isnan(img)] = -999


            img[img == -999] = np.nan

            in_shape = img.shape
            
            #turn to dataframe to normalize
            img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
			
            img = pd.DataFrame(img)
			
            img.columns = min_max_vi.columns
			
            img = pd.concat([min_max_vi, img]).reset_index(drop = True)


            #normalize 0 to 1
            img = pd.DataFrame(scaler.fit_transform(img))
			
            img = img.iloc[2:]
#
#             img = img.values.reshape(in_shape)
            img = img.values.reshape(in_shape)

#             replace nan with -1
            img[np.isnan(img)] = -1

#apply standardization
# img = normalize(img, axis=(0,1))

            img = np.round(img, 3)
            #populate x
            x[j] = img#[:, :, 4:] index number is not included, 


        #do tthe same thing for y
        y = np.zeros((self.batch_size,) + self.img_size, dtype="uint8")

        for j, path in enumerate(batch_target_img_paths):

            #load image
            img =  np.round(np.load(path), 3)[:, :, -1]

            img = img.astype(int)

            img[img < 0] = 0
            img[img >1] = 0
            img[~np.isin(img, [0,1])] = 0

            img[np.isnan(img)] = 0
            img = img.astype(int)

            # img =  tf.keras.utils.to_categorical(img, num_classes = 2)
            # y[j] = np.expand_dims(img, 2) 
            y[j] = img
  
       
    #Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
    # y[j] -= 1

        return x, y

    
#batch size and img size
BATCH_SIZE = 45
GPUS = ["GPU:0", "GPU:1", "GPU:2", "GPU:3"]
strategy = tensorflow.distribute.MirroredStrategy() #can add GPUS here to select specific ones
print('Number of devices: %d' % strategy.num_replicas_in_sync) 

batch_size = BATCH_SIZE * strategy.num_replicas_in_sync

#image size
img_size = (128, 128)

#number of classes to predict
num_classes = 1

#nbac mtbs model
model_1 = tensorflow.keras.models.load_model("/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/models/nbac_mtbs_regularize_50_global_norm.tf", 
                                           custom_objects={'precision':sm.metrics.Precision(threshold=0.5), 
                                                           'recall':sm.metrics.Recall(threshold = 0.5),
                                                            'f1-score': sm.metrics.FScore(threshold=0.5),
                                                             'iou_score': sm.metrics.IOUScore(threshold=0.5),
                                                              'accuracy': 'accuracy'})

#nbac mtbs model with 85% dnbr threshold
model_2 = tensorflow.keras.models.load_model("/explore/nobackup/people/spotter5/cnn_mapping/nbac_training/models/nbac_mtbs_regularize_50_global_norm_85.tf", 
                                           custom_objects={'precision':sm.metrics.Precision(threshold=0.5), 
                                                           'recall':sm.metrics.Recall(threshold = 0.5),
                                                            'f1-score': sm.metrics.FScore(threshold=0.5),
                                                             'iou_score': sm.metrics.IOUScore(threshold=0.5),
                                                              'accuracy': 'accuracy'})

#combined old dnbr
# model_3 = tensorflow.keras.models.load_model("/explore/nobackup/people/spotter5/cnn_mapping/Russia/models/combined_good.tf", 
#                                            custom_objects={'precision':sm.metrics.Precision(threshold=0.5), 
#                                                            'recall':sm.metrics.Recall(threshold = 0.5),
#                                                             'f1-score': sm.metrics.FScore(threshold=0.5),
#                                                              'iou_score': sm.metrics.IOUScore(threshold=0.5),
#                                                               'accuracy': 'accuracy'})

model_3 = tensorflow.keras.models.load_model("/explore/nobackup/people/spotter5/cnn_mapping/Russia/models/combined_good_old_dnbr.tf", 
                                           custom_objects={'precision':sm.metrics.Precision(threshold=0.5), 
                                                           'recall':sm.metrics.Recall(threshold = 0.5),
                                                            'f1-score': sm.metrics.FScore(threshold=0.5),
                                                             'iou_score': sm.metrics.IOUScore(threshold=0.5),
                                                              'accuracy': 'accuracy'})
#combined 85
model_4 = tensorflow.keras.models.load_model("/explore/nobackup/people/spotter5/cnn_mapping/Russia/models/combined_good_85.tf", 
                                           custom_objects={'precision':sm.metrics.Precision(threshold=0.5), 
                                                           'recall':sm.metrics.Recall(threshold = 0.5),
                                                            'f1-score': sm.metrics.FScore(threshold=0.5),
                                                             'iou_score': sm.metrics.IOUScore(threshold=0.5),
                                                              'accuracy': 'accuracy'})

#russia onl
model_5 = tensorflow.keras.models.load_model("/explore/nobackup/people/spotter5/cnn_mapping/Russia/models/russia_good_no_regularize.tf", 
                                           custom_objects={'precision':sm.metrics.Precision(threshold=0.5), 
                                                           'recall':sm.metrics.Recall(threshold = 0.5),
                                                            'f1-score': sm.metrics.FScore(threshold=0.5),
                                                             'iou_score': sm.metrics.IOUScore(threshold=0.5),
                                                              'accuracy': 'accuracy'})
#russia old dnbr method
model_6 = tensorflow.keras.models.load_model("/explore/nobackup/people/spotter5/cnn_mapping/Russia/models/russia_good_no_regularize_old_dnbr.tf", 
                                           custom_objects={'precision':sm.metrics.Precision(threshold=0.5), 
                                                           'recall':sm.metrics.Recall(threshold = 0.5),
                                                            'f1-score': sm.metrics.FScore(threshold=0.5),
                                                             'iou_score': sm.metrics.IOUScore(threshold=0.5),
                                                              'accuracy': 'accuracy'})

def predict_model(model, generator, name, fid, count):
    
    '''
    model: tensorflow model to predict
    generator: keras generator with the images to predict on
    name: string, model name\
    fid: variable I was looping through
    count: count retained earlier
    '''
    #get the results from the nbac and mtbs model
    model_1_res = model.evaluate_generator(generator, 100)

    iou = model_1_res[-2]
    precision = model_1_res[-5]
    recall = model_1_res[-4]
    f1 = model_1_res[-3]
    accuracy = model_1_res[-1]

    #make new dataframe with scores
    in_df = pd.DataFrame({
        'Model': [name],
        'FID': [fid],
        'Count': [count],
        'IOU': [iou],
        'Precision': [precision],
        'Recall': [recall],
        'F-1': [f1],
        'Accuracy': [accuracy]
                        }, index=[0])  # Explicitly setting index to [0] for a single row

    return in_df


#first these are all the good anna polygon ids, use this for joining later
# good_ids = gpd.read_file('/explore/nobackup/people/spotter5/cnn_mapping/Russia/good_polys_anna.shp')

#for the grids I have two ids, FID which is the fishnet grid cells to loop through, and ID which is teh good anna polygon nids
fish_good = gpd.read_file('/explore/nobackup/people/spotter5/cnn_mapping/Russia/model_iou_spatial/grid.shp')
fish_good['FID'] = fish_good['FID'].astype(int)
#all the fishnet ids to loop through
all_fid = fish_good['FID'].unique().tolist()

#get all the testing full pathways to predict on, will need to filter fish good with this
testing_names = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/Russia/anna_good_testing_files_full_fire.csv')['ID'].tolist()


#now I need to get the chunked files which match the fire ids to make new training, validation and testing times
#path to the chunked files
chunked =  os.listdir('/explore/nobackup/people/spotter5/cnn_mapping/Russia/anna_training_85_subs_0_128')

def filter_chunked(in_names, chunked):
    """
    Filters items in the 'chunked' list based on whether the specified part of
    each item (extracted by splitting the item's string) is in 'training_names'.

    Parameters:
    - training_names: List of integers to filter against.
    - chunked: List of strings, where each string is a filename that contains numbers.

    Returns:
    - List of strings from 'chunked' that match the filtering criteria.
    """
    # Filter the 'chunked' list
    filtered_chunked = [
        name for name in chunked 
        if int(name.split('_')[-1].split('.')[0]) in in_names
    ]
    
    filtered_chunked = ['/explore/nobackup/people/spotter5/cnn_mapping/Russia/anna_training_85_subs_0_128/' + i for i in filtered_chunked]
    return filtered_chunked

def filter_chunked2(in_names, chunked):
    """
    Filters items in the 'chunked' list based on whether the specified part of
    each item (extracted by splitting the item's string) is in 'training_names'.

    Parameters:
    - training_names: List of integers to filter against.
    - chunked: List of strings, where each string is a filename that contains numbers.

    Returns:
    - List of strings from 'chunked' that match the filtering criteria.
    """
    # Filter the 'chunked' list
    filtered_chunked = [
        name for name in chunked 
        if int(name.split('_')[-1].split('.')[0]) in in_names
    ]
    
    # filtered_chunked = ['/explore/nobackup/people/spotter5/cnn_mapping/Russia/anna_training_85_subs_0_128/' + i for i in filtered_chunked]
    return filtered_chunked

testing_names = filter_chunked(testing_names, chunked)



#save all dataframes
final = []

for fid in all_fid:
    
    print(f"Processing {fid}")
    
    #sub shapefile
    sub_grid = fish_good[fish_good['FID'] == fid]
                         
    #get the anna ids in this fid
    anna_in_fid = sub_grid['ID'].unique().tolist()
                        
    count = len(anna_in_fid)
    
    #get full pathway to the anna ids in the fids
    model_test = filter_chunked2(anna_in_fid, testing_names)
    
    #get the batch sie
    if len(model_test) <= 45:
        
        batch_size = len(model_test)
    else:
        batch_size = 45
        
    if len(model_test) > 0:
    
        #create an image generator for this fid and then predict
        models_vi_gen =  img_gen_vi(batch_size, img_size, model_test)

        mtbs_nbac = predict_model(model_1, models_vi_gen, 'MTBS_NBAC', fid, count)
        mtbs_nbac_85 = predict_model(model_2, models_vi_gen, 'MTBS_NBAC_85', fid, count)
        combined = predict_model(model_3, models_vi_gen, 'Combined', fid, count)
        combined_85 = predict_model(model_4, models_vi_gen, 'Combined_85', fid, count)
        russia = predict_model(model_6, models_vi_gen, 'Russia', fid, count)
        russia_85 = predict_model(model_5, models_vi_gen, 'Russia_85', fid, count)

        final.append(mtbs_nbac)
        final.append(mtbs_nbac_85)
        final.append(combined)
        final.append(combined_85)
        final.append(russia)
        final.append(russia_85)


final = pd.concat(final).reset_index(drop=True)

final['FID'] = final['FID'].astype(int)

final.to_csv("/explore/nobackup/people/spotter5/cnn_mapping/Russia/model_iou_spatial/grid_metrics.csv", index = False)


from pyproj import CRS

#merge back to the original shapefile for plotting
grid_metrics = pd.read_csv("/explore/nobackup/people/spotter5/cnn_mapping/Russia/model_iou_spatial/grid_metrics.csv")

#fishnet good
# fish_good = gpd.read_file('/explore/nobackup/people/spotter5/cnn_mapping/Russia/model_iou_spatial/grid.shp')
fish_good = gpd.read_file('/explore/nobackup/people/spotter5/cnn_mapping/Russia/ea_grid_clip/ea_grid_clip.shp')
# Defining the Albers Equal Area projection parameters
# albers_ea_projection = CRS("+proj=aea +lat_0=56 +lon_0=100 +lat_1=50 +lat_2=70 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

# Transforming the GeoDataFrame to the new projection
# fish_good= fish_good.to_crs(albers_ea_projection)

fish_good = fish_good.rename(columns = {'Id': 'FID'})


# # 
# fish_good['FID'] = fish_good['FID'].astype(int)

merged = fish_good.merge(grid_metrics, on = 'FID', how = 'inner')

merged = merged[['FID', 'Model', 'Count', 'IOU', 'Precision', 'Recall', 'F-1', 'Accuracy', 'geometry']]

#round floats to 2 digits
merged.loc[:, merged.select_dtypes(include=['float64']).columns] = merged.select_dtypes(include=['float64']).round(2)

#create a new column which will difference the combined model and the north america model from eurasia, do it so we subtract from russia, so larger values are better for russia
na = merged[merged['Model'] == 'MTBS_NBAC']
na_85 = merged[merged['Model'] == 'MTBS_NBAC_85']
# combined = merged[merged['Model'] == 'Combined']
# combined_85 = merged[merged['Model'] == 'Combined_85']
russ = merged[merged['Model'] == 'Russia']
russ_85 = merged[merged['Model'] == 'Russia_85']


na_russ_diff = russ['IOU'].values - na['IOU'].values
na_russ_85_diff = russ_85['IOU'].values - na_85['IOU'].values
combined_russ_diff = russ['IOU'].values - combined['IOU'].values
combined_russ_85_diff = russ_85['IOU'].values - combined_85['IOU'].values


#for each unique model loop through and save individual files, this is for making maps in arc later easier
models = merged['Model'].unique()

for m in models:
    
    sub = merged[merged['Model'] == m]
    sub['na_russ_IOU_diff'] = na_russ_diff
    sub['na_russ_85_IOU_diff'] = na_russ_85_diff
    sub['combined_russ_IOU_diff'] = combined_russ_diff
    sub['combined_russ_85_IOU_diff'] = combined_russ_85_diff
   
    
    sub.to_file(os.path.join("/explore/nobackup/people/spotter5/cnn_mapping/Russia/model_iou_spatial", f"grid_metrics_{m}.shp"))

# merged.head()

#first these are all the good anna polygon ids, use this for joining later
# good_ids = gpd.read_file('/explore/nobackup/people/spotter5/cnn_mapping/Russia/good_polys_anna.shp')

GPUS = ["GPU:0", "GPU:1", "GPU:2", "GPU:3"]
strategy = tensorflow.distribute.MirroredStrategy() #can add GPUS here to select specific ones
print('Number of devices: %d' % strategy.num_replicas_in_sync) 

#for the grids I have two ids, FID which is the fishnet grid cells to loop through, and ID which is teh good anna polygon nids
fish_good = gpd.read_file('/explore/nobackup/people/spotter5/cnn_mapping/Russia/model_iou_spatial/ecoregions.shp')

#all the fishnet ids to loop through
all_fid = fish_good['ecoregion'].unique()

#get all the testing full pathways to predict on, will need to filter fish good with this
testing_names = pd.read_csv('/explore/nobackup/people/spotter5/cnn_mapping/Russia/anna_good_testing_files_full_fire.csv')['ID'].tolist()


#now I need to get the chunked files which match the fire ids to make new training, validation and testing times
#path to the chunked files
chunked =  os.listdir('/explore/nobackup/people/spotter5/cnn_mapping/Russia/anna_training_85_subs_0_128')

def filter_chunked(in_names, chunked):
    """
    Filters items in the 'chunked' list based on whether the specified part of
    each item (extracted by splitting the item's string) is in 'training_names'.

    Parameters:
    - training_names: List of integers to filter against.
    - chunked: List of strings, where each string is a filename that contains numbers.

    Returns:
    - List of strings from 'chunked' that match the filtering criteria.
    """
    # Filter the 'chunked' list
    filtered_chunked = [
        name for name in chunked 
        if int(name.split('_')[-1].split('.')[0]) in in_names
    ]
    
    filtered_chunked = ['/explore/nobackup/people/spotter5/cnn_mapping/Russia/anna_training_85_subs_0_128/' + i for i in filtered_chunked]
    return filtered_chunked

def filter_chunked2(in_names, chunked):
    """
    Filters items in the 'chunked' list based on whether the specified part of
    each item (extracted by splitting the item's string) is in 'training_names'.

    Parameters:
    - training_names: List of integers to filter against.
    - chunked: List of strings, where each string is a filename that contains numbers.

    Returns:
    - List of strings from 'chunked' that match the filtering criteria.
    """
    # Filter the 'chunked' list
    filtered_chunked = [
        name for name in chunked 
        if int(name.split('_')[-1].split('.')[0]) in in_names
    ]
    
    # filtered_chunked = ['/explore/nobackup/people/spotter5/cnn_mapping/Russia/anna_training_85_subs_0_128/' + i for i in filtered_chunked]
    return filtered_chunked

testing_names = filter_chunked(testing_names, chunked)



# #save all dataframes
final = []

for fid in all_fid:
    
    #sub shapefile
    sub_grid = fish_good[fish_good['ecoregion'] == fid]
                         
    #get the anna ids in this fid
    anna_in_fid = sub_grid['ID'].unique().tolist()
                        
    count = len(anna_in_fid)
    
    #get full pathway to the anna ids in the fids
    model_test = filter_chunked2(anna_in_fid, testing_names)
        
   #get the batch sie
    if len(model_test) <= 45:
        
        batch_size = len(model_test)
    else:
        batch_size = 45
        
    if len(model_test) > 0:
    
        #create an image generator for this fid and then predict
        models_vi_gen =  img_gen_vi(batch_size, img_size, model_test)
        
        mtbs_nbac = predict_model(model_1, models_vi_gen, 'MTBS_NBAC', fid, count)
        mtbs_nbac_85 = predict_model(model_2, models_vi_gen, 'MTBS_NBAC_85', fid, count)
        combined = predict_model(model_3, models_vi_gen, 'Combined', fid, count)
        combined_85 = predict_model(model_4, models_vi_gen, 'Combined_85', fid, count)
        russia = predict_model(model_6, models_vi_gen, 'Russia', fid, count)
        russia_85 = predict_model(model_5, models_vi_gen, 'Russia_85', fid, count)

        final.append(mtbs_nbac)
        final.append(mtbs_nbac_85)
        final.append(combined)
        final.append(combined_85)
        final.append(russia)
        final.append(russia_85)



final = pd.concat(final).reset_index(drop=True)

final.to_csv("/explore/nobackup/people/spotter5/cnn_mapping/Russia/model_iou_spatial/ecoregion_metrics.csv", index = False)

from pyproj import CRS

#merge back to the original shapefile for plotting
grid_metrics = pd.read_csv("/explore/nobackup/people/spotter5/cnn_mapping/Russia/model_iou_spatial/ecoregion_metrics.csv")

#fishnet good
# fish_good = gpd.read_file('/explore/nobackup/people/spotter5/cnn_mapping/Russia/model_iou_spatial/grid.shp')
fish_good = gpd.read_file('/explore/nobackup/people/spotter5/cnn_mapping/Russia/raw_files/EcoRegion_AlbEAadj/EcoRegion_AlbEAadj/EcoRegion_g.shp')
# Defining the Albers Equal Area projection parameters
# albers_ea_projection = CRS("+proj=aea +lat_0=56 +lon_0=100 +lat_1=50 +lat_2=70 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs")

# Transforming the GeoDataFrame to the new projection
# fish_good= fish_good.to_crs(albers_ea_projection)

fish_good = fish_good.rename(columns = {'ecoregion': 'FID'})


# # 
# fish_good['FID'] = fish_good['FID'].astype(int)

merged = fish_good.merge(grid_metrics, on = 'FID', how = 'inner')

merged = merged[['FID', 'Model', 'Count', 'IOU', 'Precision', 'Recall', 'F-1', 'Accuracy', 'geometry']]

#round floats to 2 digits
merged.loc[:, merged.select_dtypes(include=['float64']).columns] = merged.select_dtypes(include=['float64']).round(2)

#create a new column which will difference the combined model and the north america model from eurasia, do it so we subtract from russia, so larger values are better for russia
na = merged[merged['Model'] == 'MTBS_NBAC']
na_85 = merged[merged['Model'] == 'MTBS_NBAC_85']
# combined = merged[merged['Model'] == 'Combined']
# combined_85 = merged[merged['Model'] == 'Combined_85']
russ = merged[merged['Model'] == 'Russia']
russ_85 = merged[merged['Model'] == 'Russia_85']


na_russ_diff = russ['IOU'].values - na['IOU'].values
na_russ_85_diff = russ_85['IOU'].values - na_85['IOU'].values
combined_russ_diff = russ['IOU'].values - combined['IOU'].values
combined_russ_85_diff = russ_85['IOU'].values - combined_85['IOU'].values

#for each unique model loop through and save individual files, this is for making maps in arc later easier
models = merged['Model'].unique()

for m in models:
    
    sub = merged[merged['Model'] == m]
    sub['na_russ_IOU_diff'] = na_russ_diff
    sub['na_russ_85_IOU_diff'] = na_russ_85_diff
    sub['combined_russ_IOU_diff'] = combined_russ_diff
    sub['combined_russ_85_IOU_diff'] = combined_russ_85_diff
    
    sub.to_file(os.path.join("/explore/nobackup/people/spotter5/cnn_mapping/Russia/model_iou_spatial", f"ecoregion_metrics_{m}.shp"))


