#load img -> detect face and extract its features
#find the cosine distance of current image with all the 8664 features
#recommend that image

from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from mtcnn import MTCNN
from PIL import Image
import os


#loading embedding.pkl file in read binary mode and converting it into an array
feature_list = np.array(pickle.load(open('embedding.pkl' , 'rb')))
#storing all the filenames in a variable from filenames.pkl
filenames = pickle.load(open('filenames.pkl' , 'rb'))

#we are using the same model for extraction of features of the user given image

model = VGGFace(model='resnet50' , include_top=False , input_shape=(224,224,3) , pooling='avg')

detector = MTCNN();

img_src = input("Enter your image path:  ");

sample_img = cv2.imread(img_src)

#detecting face
results = detector.detect_faces(sample_img)

x,y,width,height = results[0]['box']

face = sample_img[y:y+height,x:x+width]

# cv2.imshow('output',face)
# cv2.waitKey(0) #required to show output in opencv

#we are now making an image from the image array (face detected array), to resize it
image = Image.fromarray(face)
#resizing the image and storing it back into image
image = image.resize((224,224))

#make the image back as an array
face_array = np.asarray(image)

#orelse sometimes error shows
face_array = face_array.astype('float32')

expanded_img = np.expand_dims(face_array,axis=0)
#now the expanded img is made into a preprocessed img
preprocessed_img=preprocess_input(expanded_img)
#the model is used with the preprocessed image to get all 2048 features, in 1d -> flatten is used
result = model.predict(preprocessed_img).flatten()

#finding the cosinse similiraty, but cosine similiraty takes only 2d array so we reshaped them
#now the cosine similarity will come in form [[similarity]], so we will need only the [0][0],zero index first element
similarity = []
#using for loop to check similarity form all in feature_list

for i in range(len(feature_list)):
    #all the values of cosine similarity is stored in similarity array by appending
    similarity.append(cosine_similarity(result.reshape(1,-1) , feature_list[i].reshape(1,-1))[0][0])

#Now here is a problem, if we now sort the similarity the the indexes will be changed and we can not recommend the corect name from filenames
#so we will enumurate all the similarity array to get tuples containing the index like ((1,0.344567),(2,0.256758))
#then we will sort with respect to the second item, i.e the cosine similarity so we use key=lambda x:x[1]
#and we sort in descending order, as the largest cosine similarity gives most similar image
#now we take out the [0][0] element which is the index of the most similar image and store it in index_pos

index_pos = sorted(list(enumerate(similarity)),reverse = True,key=lambda x:x[1])[0][0]
temp_img = cv2.imread(filenames[index_pos])

file_name = os.path.basename(filenames[index_pos])
print("you look like:",file_name)

cv2.imshow('output',temp_img)
cv2.waitKey(0)