from numpy.linalg import norm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical


#%%
#function for dependency between m th and n th sensor

def dep_cor(rsq_mat,sensor,m,n):
  
  series = pd.Series(sensor)
  cumsum = list(series.cumsum())
  cumsum=[0]+cumsum
  ind1=list(range(cumsum[m-1],cumsum[m]))
  ind2=list(range(cumsum[n-1],cumsum[n]))
  print(ind1)
  cor=rsq_mat.iloc[ind1,ind2]
  return(min(cor.apply(max,1)))


#%%
def func_modified(regularizer_rate_0,regularizer_rate_1,num_layers_0, epochs, batch_size, num_classes, sensor_sizes, dep, xvals, yvals, reduction):

  from numpy.linalg import norm
  import numpy as np
  import pandas as pd
  import tensorflow as tf
  from sklearn.metrics import accuracy_score
  from tensorflow.keras.layers import Dense, Activation
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
  from sklearn.model_selection import train_test_split
  import statistics
  xvals_train, xvals_test,yvals_train, yvals_test = train_test_split(xvals,yvals,random_state=None, test_size=0.2,  shuffle=True)
                                                                     
  starter_learning_rate = 0.001
  num_features=sum(sensor_sizes)
  nrow=len(yvals_train)
  num_output=num_classes

  input_X = tf.placeholder('float32',shape =(None,num_features),name="input_X")
  input_y = tf.placeholder('float32',shape = (None,num_classes),name='input_Y')

  s=tf.compat.v1.InteractiveSession()
  ## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
  weights_0 = tf.Variable(tf.random.normal([num_features,num_layers_0], stddev=(1/tf.sqrt(float(num_features)))))
  bias_0 = tf.Variable(tf.random.normal([num_layers_0]))
  weights_1 = tf.Variable(tf.random.normal([num_layers_0,num_output], stddev=(1/tf.sqrt(float(num_layers_0)))))
  bias_1 = tf.Variable(tf.random.normal([num_output]))

  ## Initializing weigths and biases
  hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights_0)+bias_0)
  predicted_y = tf.sigmoid(tf.matmul(hidden_output_0,weights_1) + bias_1)

  ##calculate penalty terms
  series = pd.Series(sensor_sizes)
  cumsum = series.cumsum()
  penalty=(tf.reduce_sum(tf.square(weights_0[0:sensor_sizes[0]])))**0.5/sensor_sizes[0]
  for i in range(len(sensor_sizes)-1):
    penalty=penalty+((tf.reduce_sum(tf.square(weights_0[cumsum[i]:cumsum[i+1]])))**0.5)/sensor_sizes[i+1]

  cumsum =[0]+ list(series.cumsum())
  redund=0
  r_mat=np.array(xvals_train.corr())
  rsq_mat=[[elem*elem for elem in inner] for inner in r_mat]
  rsq_mat=pd.DataFrame(rsq_mat)
  for i in range(len(sensor_sizes)):
    for j in range(len(sensor_sizes)):
      if j!=i:
        redund=redund+dep(rsq_mat,sensor_sizes,i+1,j+1)*((tf.reduce_sum(tf.square(weights_0[cumsum[j]:cumsum[j+1]])))*(tf.reduce_sum(tf.square(weights_0[cumsum[i]:cumsum[i+1]])))**0.5)/(sensor_sizes[i]*sensor_sizes[j])

  if len(sensor_sizes)>1:
    redund=redund/(len(sensor_sizes)*(len(sensor_sizes)-1))

  loss = tf.reduce_mean(tf.square(predicted_y-tf.convert_to_tensor(yvals_train, dtype=tf.float32))) + regularizer_rate_0*redund/num_layers_0**2 + regularizer_rate_1*penalty/num_layers_0 


  ## Variable learning rate
  learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
  ## Adam optimzer for finding the right weight
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=[weights_0,weights_1,
                                                                         bias_0,bias_1])    
  ## Metrics definition
  correct_prediction = tf.equal(tf.argmax(yvals_train,1), tf.argmax(predicted_y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  training_accuracy = []
  training_loss = []

  #s.run(tf.initialize_all_variables)
  s.run(tf.compat.v1.global_variables_initializer())
  for epoch in range(epochs):    
    arr = np.arange(nrow)
    np.random.shuffle(arr)
    for index in range(0,nrow,batch_size):
        s.run(optimizer, {input_X: xvals_train,
                          input_y: yvals_train})
        
    training_accuracy.append(s.run(accuracy, feed_dict= {input_X:xvals_train, 
                                                         input_y: yvals_train}))
    training_loss.append(s.run(loss, {input_X: xvals_train, 
                                      input_y: yvals_train}))
  print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}".format(epoch,
                                                                    training_loss[epoch],
                                                                    training_accuracy[epoch]
                                                                   ))
    
  
  y_pred = np.array(s.run(predicted_y, feed_dict={input_X: xvals_test}))
  y_pred= np.where(y_pred == y_pred.max(axis=1)[:, np.newaxis], 1, 0)
  testacc = accuracy_score(yvals_test, y_pred)
 
  print("\nTest Accuracy: {0:f}\n".format(testacc))

  w0=weights_0.eval()
  w=[]
  #w.append(norm(w0[0:sensor_sizes[0]],2))
  for i in range(len(sensor_sizes)):
    w.append(norm(w0[cumsum[i]:cumsum[i+1]],2))
  print(w)
  #Feature selection
  if reduction==True:
    v=[i for i,x in enumerate(w) if x > 0.1*max(w)]
    selected=[]
    for i in v:
      selected.append(xvals.iloc[:,range(cumsum[i],cumsum[i+1])])
    
    xvals_reduced=pd.concat(selected,ignore_index=True, axis=1)
    
    acc=[]
    sensor_sizes_red=[sensor_sizes[i] for i in v]
    s.close()
    for i in range(10):
     xvals_train, xvals_test,yvals_train, yvals_test = train_test_split(xvals_reduced,yvals,random_state=None, test_size=0.2,  shuffle=True)
     x=func_modified_landsat(0,0,num_layers_0, epochs, batch_size, num_classes, sensor_sizes_red,dep_cor,xvals_train, yvals_train,xvals_test, yvals_test,reduction=False)
     acc.append(x[0])
    
    return([sum(acc)/10,statistics.stdev(acc),len(sensor_sizes_red),v])
  else:
    s.close()
    return([testacc,len(sensor_sizes)])



#%%
def func_modified_landsat(regularizer_rate_0,regularizer_rate_1,num_layers_0, epochs, batch_size, num_classes, sensor_sizes, dep, xvals_train, yvals_train,xvals_test, yvals_test, reduction):

  from numpy.linalg import norm
  import numpy as np
  import pandas as pd
  import tensorflow as tf
  from sklearn.metrics import accuracy_score
  from tensorflow.keras.layers import Dense, Activation
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
  from sklearn.model_selection import train_test_split

  #xvals_train, xvals_test,yvals_train, yvals_test = train_test_split(xvals,yvals,random_state=None, test_size=0.2,  shuffle=True)
                                                                     
  starter_learning_rate = 0.001
  num_features=sum(sensor_sizes)
  nrow=len(yvals_train)
  num_output=num_classes

  input_X = tf.placeholder('float32',shape =(None,num_features),name="input_X")
  input_y = tf.placeholder('float32',shape = (None,num_classes),name='input_Y')

  s=tf.compat.v1.InteractiveSession()
  ## Weights initialized by random normal function with std_dev = 1/sqrt(number of input features)
  weights_0 = tf.Variable(tf.random.normal([num_features,num_layers_0], stddev=(1/tf.sqrt(float(num_features)))))
  bias_0 = tf.Variable(tf.random.normal([num_layers_0]))
  weights_1 = tf.Variable(tf.random.normal([num_layers_0,num_output], stddev=(1/tf.sqrt(float(num_layers_0)))))
  bias_1 = tf.Variable(tf.random.normal([num_output]))

  ## Initializing weigths and biases
  hidden_output_0 = tf.nn.relu(tf.matmul(input_X,weights_0)+bias_0)
  predicted_y = tf.sigmoid(tf.matmul(hidden_output_0,weights_1) + bias_1)

  ##calculate penalty terms
  series = pd.Series(sensor_sizes)
  cumsum = series.cumsum()
  
  cumsum =[0]+ list(series.cumsum())
  
  
  loss = tf.reduce_mean(tf.square(predicted_y-tf.convert_to_tensor(yvals_train, dtype=tf.float32))) 

  ## Variable learning rate
  learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
  ## Adam optimzer for finding the right weight
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=[weights_0,weights_1,
                                                                         bias_0,bias_1])    
  ## Metrics definition
  correct_prediction = tf.equal(tf.argmax(yvals_train,1), tf.argmax(predicted_y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  training_accuracy = []
  training_loss = []

  #s.run(tf.initialize_all_variables)
  s.run(tf.compat.v1.global_variables_initializer())
  for epoch in range(epochs):    
    arr = np.arange(nrow)
    np.random.shuffle(arr)
    for index in range(0,nrow,batch_size):
        s.run(optimizer, {input_X: xvals_train,
                          input_y: yvals_train})
        
    training_accuracy.append(s.run(accuracy, feed_dict= {input_X:xvals_train, 
                                                         input_y: yvals_train}))
    training_loss.append(s.run(loss, {input_X: xvals_train, 
                                      input_y: yvals_train}))
    print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}".format(epoch,
                                                                    training_loss[epoch],
                                                                    training_accuracy[epoch]
                                                                   ))
    
  
    
  y_pred = np.rint(s.run(predicted_y, feed_dict={input_X: xvals_test}))

  testacc = accuracy_score(yvals_test, y_pred)
 
  print("\nTest Accuracy: {0:f}\n".format(testacc))

  w0=weights_0.eval()
  w=[]
  #w.append(norm(w0[0:sensor_sizes[0]],2))
  for i in range(len(sensor_sizes)):
    w.append(norm(w0[cumsum[i]:cumsum[i+1]],2))
  print(w)
  #Feature selection
  if reduction==True:
    v=[i for i,x in enumerate(w) if x > 0.1*max(w)]
    selected=[]
    for i in v:
      selected.append(xvals_train.iloc[:,range(cumsum[i],cumsum[i+1])])
    
    xvals_train_red=pd.concat(selected,ignore_index=True, axis=1)
    
    selected=[]
    for i in v:
      selected.append(xvals_test.iloc[:,range(cumsum[i],cumsum[i+1])])
    
    xvals_test_red=pd.concat(selected,ignore_index=True, axis=1)
    
  
    acc=0
    c=0
    sensor_sizes_red=[sensor_sizes[i] for i in v]
    for i in range(10):
     x=func_modified_landsat(regularizer_rate_0,regularizer_rate_1,num_layers_0, epochs, batch_size, num_classes, sensor_sizes_red,dep_cor,xvals_train_red, yvals_train,xvals_test_red, yvals_test,reduction=False)
     if x[0] > 0.4:
         acc=acc+x[0]
         c=c+1
    s.close()
    return([acc/c,len(sensor_sizes_red),v])
  else:
    s.close()
    return([testacc,len(sensor_sizes)])



#%%

#IRIS-1
iris=pd.read_csv('/Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/Iris.csv')
#iris=pd.read_csv('C:/Users/CILAB2/Downloads/ayti-datasets/iris.csv')
yvals=to_categorical(np.asarray(iris['Species'].factorize()[0]))

shuffled = iris.sample(frac=1)
folds = np.array_split(shuffled, 10) 
num_layers_0=np.arange(4, 21, 2)
acc=[]


for j in num_layers_0:
    a=0
    for i in range(10):
        tst=folds.pop()
        trn=pd.concat(folds)
        x=func_modified_landsat(0,0,j, 500, 10, 3, [2,2],dep_cor,trn.iloc[:,1:5],yvals[trn['Id']-1],tst.iloc[:,1:5],yvals[tst['Id']-1] ,reduction=False)
        a=a+x[0]
        folds.insert(0,tst)
    acc.append(a/10)
    
nodes=num_layers_0[acc.index(max(acc))]

xvals = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].astype(np.float32)

#writer = pd.ExcelWriter('output_1_iris1_new_tuning_10_times.xlsx')

for i in [20, 50]:
  for j in [0, 2, 5]:
      result = []
      for k in range(10):
          print(i,j,k)
          x = func_modified(i, j, nodes, 500, 10, 3, [2, 2], dep_cor, xvals, yvals, True)
          result.append([i, j, x[0], x[1], x[2], x[3]])
      result_iris1 = pd.DataFrame(result)
      result_iris1.columns = ["Lambda", "Mu", "Test Accuracy",
                              "Sd", "Number of sensors selected", "Selected sensors"]
      writer = pd.ExcelWriter('output_1_iris1_new_tuning_'+ str(i) + '_' + str(j)+'_10_times.xlsx')
      # write dataframe to excel
      result_iris1.to_excel(writer)
      # save the excel
      writer.save()
      
#%%
#IRIS-2

f1=iris['SepalLengthCm']
f2=iris['SepalWidthCm']
f3=iris['PetalLengthCm']
f4=iris['PetalWidthCm']
e1=f1+np.random.normal(loc=0.0, scale=0.05, size=150)
e2=f3+np.random.normal(loc=0.0, scale=0.05, size=150)
e3=f4+np.random.normal(loc=0.0, scale=0.05, size=150)
xvals=pd.concat([f1,f2,e1,f3,f4,e2,e3], axis=1, ignore_index=True).astype(np.float32)

for i in [ 20, 50]:
  for j in [0, 2, 5]:
      result = []
      for k in range(10):
          print(i,j,k)
          x = func_modified(i, j, nodes, 500, 10, 3, [2,3,2], dep_cor, xvals, yvals, True)
          result.append([i, j, x[0], x[1], x[2], x[3]])
      result_iris1 = pd.DataFrame(result)
      result_iris1.columns = ["Lambda", "Mu", "Test Accuracy",
                              "Sd", "Number of sensors selected", "Selected sensors"]
      writer = pd.ExcelWriter('output_1_iris2_new_tuning_'+ str(i) + '_' + str(j)+'_10_times.xlsx')
      # write dataframe to excel
      result_iris1.to_excel(writer)
      # save the excel
      writer.save()

#%%
#Gas-sensor

data=pd.read_csv('//Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/GasSensor(cleaned in R).csv',header=None)
#data=pd.read_csv('C:/Users/CILAB2/Downloads/ayti-datasets/GasSensor(cleaned in R).csv')
data.drop(0,axis=1,inplace=True)
data.drop(0,axis=0,inplace=True)
data.dropna() 
data[1]=data[1].replace(['1'],1)
data[1]=data[1].replace(['2'],2)
data[1]=data[1].replace(['3'],3)
data[1]=data[1].replace(['4'],4)
data[1]=data[1].replace(['5'],5)
data[1]=data[1].replace(['6'],6)

yvals=data[1]
data=data.astype(np.float32)
yvals=to_categorical(np.asarray(yvals.factorize()[0]))
from scipy.stats import zscore
for i in range(128):
  data.iloc[:,i+1]=zscore(data.iloc[:,i+1])

data['Id']=list(range(0,13910))

shuffled = data.sample(frac=1)
folds = np.array_split(shuffled, 10) 
num_layers_0=np.arange(4, 21, 2)
acc=[]

from itertools import repeat
for j in num_layers_0:
    a=0
    for i in range(10):
        tst=folds.pop()
        trn=pd.concat(folds)
        x=func_modified_landsat(0,0,j, 500, 2000, 6,
                                list(repeat(8,16)),dep_cor,
                                trn.iloc[:,1:129],yvals[trn['Id']],
                                tst.iloc[:,1:129],yvals[tst['Id']],
                                reduction=False)
        a=a+x[0]
        folds.insert(0,tst)
    acc.append(a/10)
    
nodes=num_layers_0[acc.index(max(acc))]

for i in [50]:
  for j in [ 2, 5]:
      result = []
      for k in range(10):
          print(i,j,k)
          x = func_modified(i,j,nodes, 500, 2000, 6,
                                  list(repeat(8,16)),dep_cor,
                                  data.iloc[:,1:129],
                                  yvals,
                                  reduction=True)
          result.append([i, j, x[0], x[1], x[2], x[3]])
      result_gs = pd.DataFrame(result)
      result_gs.columns = ["Lambda", "Mu", "Test Accuracy",
                              "Sd", "Number of sensors selected", "Selected sensors"]
      writer = pd.ExcelWriter('output_1_gs_new_tuning_'+ str(i) + '_' + str(j)+'_10_times.xlsx')
      # write dataframe to excel
      result_gs.to_excel(writer)
      # save the excel
      writer.save()
      
#%%

#LRS
lrs=pd.read_csv('/Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/lrs_cleaned.csv')
lrs.drop('Unnamed: 0',axis=1,inplace=True)

yvals = lrs.iloc[:,1]
yvals=(yvals-yvals % 10)/10
yvals=to_categorical(np.asarray(yvals.factorize()[0]))

from scipy.stats import zscore
for i in range(93):
  lrs.iloc[:,i+10]=zscore(lrs.iloc[:,i+10])
xvals = lrs.iloc[:,10:103]

lrs['Id']=list(range(0,len(yvals)))

shuffled = lrs.sample(frac=1)
folds = np.array_split(shuffled, 10) 
num_layers_0=np.arange(4, 21, 2)
acc=[]

from itertools import repeat
for j in num_layers_0:
    a=0
    for i in range(10):
        tst=folds.pop()
        trn=pd.concat(folds)
        x=func_modified_landsat(0,0,j, 500, 100,10,[44,49],dep_cor,
                                trn.iloc[:,10:103],yvals[trn['Id']],
                                tst.iloc[:,10:103],yvals[tst['Id']],
                                reduction=False)
        a=a+x[0]
        folds.insert(0,tst)
    acc.append(a/10)
    
nodes=num_layers_0[acc.index(max(acc))]

for i in [50]:
  for j in [0, 2, 5]:
      result = []
      for k in range(10):
          print(i,j,k)
          x = func_modified(i,j,nodes, 500,50,10,[44,49],dep_cor,
                                  xvals,
                                  yvals,
                                  reduction=True)
          result.append([i, j, x[0], x[1], x[2], x[3]])
      result_lrs = pd.DataFrame(result)
      result_lrs.columns = ["Lambda", "Mu", "Test Accuracy",
                              "Sd", "Number of sensors selected", "Selected sensors"]
      writer = pd.ExcelWriter('output_1_lrs_new_tuning_'+ str(i) + '_' + str(j)+'_10_times.xlsx')
      # write dataframe to excel
      result_lrs.to_excel(writer)
      # save the excel
      writer.save()

#%%
#RSData-1

rs=pd.read_csv('/Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/rs_8cl.csv')
#rs=pd.read_csv('C:/Users/CILAB2/Downloads/ayti-datasets/sensor-selection-datasets/rs_8cl.csv')

from scipy.stats import zscore
for i in range(7):
  rs.iloc[:,i+1]=zscore(rs.iloc[:,i+1])

s=[]
for i in range(8):
    s.append(rs.loc[rs['V8']==i+1].sample(frac=1).iloc[0:200,])
    
trn_data=pd.concat(s)

trn_data['Id']=list(range(0,1600))

shuffled = lrs.sample(frac=1)
folds = np.array_split(shuffled, 10) 
num_layers_0=np.arange(4, 21, 2)
acc=[]

from itertools import repeat
for j in num_layers_0:
    a=0
    for i in range(10):
        tst=folds.pop()
        trn=pd.concat(folds)
        x=func_modified_landsat(0,0,j, 500, 100,8,list(repeat(8,16)),dep_cor,
                                trn.iloc[:,1:8],yvals[trn['Id']],
                                tst.iloc[:,1:8],yvals[tst['Id']],
                                reduction=False)
        a=a+x[0]
        folds.insert(0,tst)
    acc.append(a/10)
    
nodes=num_layers_0[acc.index(max(acc))]
    
xvals = rs.iloc[:,1:8]
yvals = rs.iloc[:,8]
yvals=to_categorical(np.asarray(yvals.factorize()[0]))

#%%

import scipy.io

bci=scipy.io.loadmat('/Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/BCICIV_1_mat/BCICIV_calib_ds1a.mat')
ev=scipy.io.loadmat('/Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/BCICIV_1_mat/BCICIV_eval_ds1a.mat')
