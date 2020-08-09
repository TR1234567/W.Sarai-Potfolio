#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imutils import paths
import numpy as np
import imutils
import cv2
import os
import matplotlib.pyplot as plt


# In[2]:


rawImages = []
labels = []
DATADIR = 'Path/'
IMG_SIZE = 512

# เป็น function ที่บอกว่า file เรามี normal,abnormal กี่รูป 
def create_array_data():
    #1 = Normal , 0 = Abnormal
    CATEGORIES = ["Abnormal", "Normal"] # มาจาก file เราที่แยกให้แล้วด้วยมือ แต่ถ้าของจริงรูปมันเยอะเราเลยแยกด้วยตาเป็น 2file ไม่ไหว เลยต้องใช้วิธีอื่น
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category) # จาก CATEGORIES ระบุว่าให้แสดง "Normal" เป็น 1, "Abnormal" เป็น 0
        for img in os.listdir(path):    
            img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_COLOR)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            rawImages.append(new_array)
            labels.append(class_num)
create_array_data()


# In[3]:


scaler = StandardScaler()
scaler.fit(rawImage)
scaled_data = scaler.transform(rawImage)
trainFeat= scaler.transform(labels)
pca = PCA(n_components=68)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)


# In[4]:


(trainRI, testRI, trainRL, testRL) = train_test_split(rawImage, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25, random_state=42)


# In[5]:


def image_to_feature_vector(image, size=(256, 256)):
    # resize the image to a fixed size, then flatten the image into
    # a list of raw pixel intensities
    return cv2.resize(image, size).flatten()


# In[6]:


def extract_color_histogram(image, bins=(8, 8, 8)):
    # extract a 3D color histogram from the HSV color space using
    # the supplied number of `bins` per channel
    #clr = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,[0, 180, 0, 256, 0, 256])


    # handle normalizing the histogram if we are using OpenCV 2.4.X

    if imutils.is_cv2():
        hist = cv2.normalize(hist)
        # otherwise, perform "in place" normalization in OpenCV 3 (I
        # personally hate the way this is done
    else:
        cv2.normalize(hist, hist)
    # return the flattened histogram as the feature vector
    return hist.flatten()


# In[7]:


rawImage =[]
for o in rawImages:
    pixels = image_to_feature_vector(o)
    hist = extract_color_histogram(o)
    rawImage.append(pixels)
    features.append(hist)


# In[8]:


print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=2,n_jobs=-1)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))


# In[9]:


pred = model.predict(testFeat)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix( testLabels,pred))


# In[11]:


error_rate = []

# Will take some time
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(trainFeat,trainLabels)
    pred_i = knn.predict(testFeat)
    error_rate.append(np.mean(pred_i != testLabels))


# In[12]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[13]:


# NOW WITH K=2
knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(trainFeat,trainLabels)
pred = knn.predict(testFeat)

print('WITH K=2')
print('\n')
print(confusion_matrix( testLabels,pred))
print('\n')
print(classification_report( testLabels,pred))


# In[ ]:


import pickle 
knnPickle = open('knnpickle_file2', 'wb') 
# source, destination 
pickle.dump(knn, knnPickle)                      
# load the model from disk
loaded_model = pickle.load(open('knnpickle_file2', 'rb'))

