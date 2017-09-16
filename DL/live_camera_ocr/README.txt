File/Folder organization : 

Directories : 
1. data : This will contain all the datasets
2. model : This will contain the model and its saved parameters

Files :
1. data_utils.py : This file contains all the helper functions for data loading, manipulation, batch generation, etc

2. live_camera_ocr.py : This file contains the pipeline to build and train a model for image recognition [should be able to train the model from a previous saved state.]

3. DCNN_wrapper.py : This file contains the implementation for the functions used in the pipeline for model creation, training, predictions, etc

