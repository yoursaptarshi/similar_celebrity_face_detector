# import os
# import pickle
#
# #importing all actors name, and storing it in actors varriable
# actors = os.listdir('data')
#
# filenames = []
# #using A FOR LOOP TO RUN THROUGH ALL FILES IN 'actors' VARIABLE and storing each in actor
# for actor in actors:
#     #ANOTHER FOR LOOP TO RUN THROUGH ALL FILES IN 'actor' and store each file as 'file' variable
#     for file in os.listdir(os.path.join('data',actor)):
#         # adding all the file names in filenames array
#         filenames.append(os.path.join('data',actor,file))
#
# #to use them we have to make them binary files
# #to make into binary mode, we have to use pickle in write binary (wb) mode
# #all the file paths are stored in the pkl file
# pickle.dump(filenames,open('filenames.pkl', 'wb'))
#
# #we are commenting the above code, as we have all the paths stored in filenames.pkl

from tensorflow.keras.preprocessing import image
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from tqdm import tqdm
#opening the data stored in filenames.pkl in read binary (rb) mode
filenames=pickle.load(open('filenames.pkl' , 'rb'))

#ResNet-50 is a convolutional neural network that is 50 layers deep, we removed the top layer
model = VGGFace(model='resnet50' , include_top=False , input_shape=(224,224,3) , pooling='avg')

#making a function which will extract features of a given image

#defined a function which need input of image path, and model
def feature_extractor(img_path,model):
    #we are loading a image using the imported image function
    img = image.load_img(img_path,target_size=(224,224))
    #we are making an array of images, by using image_to_array function 
    img_array = image.img_to_array(img)
    #now the image array acts as a single image, so we will use expanable sheet from numpy to make it batch of images
    expanded_img = np.expand_dims(img_array,axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    #now we are feeding all the preprocessed image to our model to get all the features, using flatten to get features in 1d
    result = model.predict(preprocessed_img).flatten()

    return result

features = []

#tqdm will show a progress bar

for file in tqdm(filenames):
    #all the features will be extracted and stored in features array
    features.append(feature_extractor(file,model)) 
#using pickle dump the features in a file called embedding.pkl in write binary (wb) mode
pickle.dump(features,open('embedding.pkl' , 'wb'))