import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import sys
import os
import random
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.svm import SVC 
import seaborn as sns
class_labels_aug = np.load('updated/trainlabels1d_aug.npy')
features_aug = np.load('updated/trainfeatures1d_aug.npy')
test_features_aug = np.load('updated/testfeatures1d_aug.npy')
test_class_labels_aug = np.load('updated/testlabels1d_aug.npy')

svm_model_linear = SVC(kernel = 'linear', C = 1).fit(features_aug, class_labels_aug) 
svm_predictions = svm_model_linear.predict(test_features_aug) 
  
# model accuracy for X_test   
accuracy = svm_model_linear.score(test_features_aug, test_class_labels_aug) 
  
# creating a confusio2n matrix 
cm_Aug = confusion_matrix(test_class_labels_aug, svm_predictions) 
plt.figure(figsize=(9,9))
sns.heatmap(cm_Aug, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(round(accuracy, 2))
plt.title(all_sample_title, size = 15);
plt.savefig('accuracy_withaug_Svm.png', dpi=100,bbox_inches='tight')
plt.show()
print(accuracy)
print(cm_Aug)
