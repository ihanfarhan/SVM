import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm

data1 = pd.read_excel('kon08-0.xlsx',sheet_name='integrasi')
x1 = data1['X1']
x2 = data1['X2']
x3 = data1['X3']
x4 = data1['X4']
x5 = data1['X5']
x_training = np.array(list(zip(x1,x2,x3,x4,x5)))
y_training = data1['Z-Score']

nama_kelas = ['Bankcrupt','Gray Zone','Non Bankcrupt']

#TRAINING DATA
i_non = y_training[y_training > 2.99].index
i_between = y_training [(y_training >= 1.81) & (y_training <= 2.99)].index 
i_min = y_training[y_training < 1.81].index

plt.figure()
plt.scatter(x_training[i_min,0],x_training[i_min,1],c='r',s=100)
plt.scatter(x_training[i_between,0],x_training[i_between,1],c='g',s=100)
plt.scatter(x_training[i_non,0],x_training[i_non,1],c='b',s=100)
plt.legend(nama_kelas)
plt.title('Training Data')

#SVM
svc = svm.SVC(kernel='linear').fit(x_training,y_training)
weight = svc.coef_
intercept = svc.intercept_
a = -weight[0,0] / weight[0,1]
b = -intercept[0]/weight[0,1]
print('y = '+str(a)+' * x + '+str(b))

x_coba = np.arange(2,3.8,0.1)
y_coba = np.multiply(a,x_coba)+b

x_asing = np.array([[3,6]])
print svc.predict(x_asing)

plt.plot(x_coba,y_coba)

print svc.score(x_training,y_training)