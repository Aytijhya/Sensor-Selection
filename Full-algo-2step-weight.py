
from numpy.linalg import norm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

#%%
def func_modified2(regularizer_rate_0,regularizer_rate_1,num_layers_0, epochs, batch_size, num_classes, sensor_sizes, dep, xvals, yvals, reduction):

  from numpy.linalg import norm
  import keras
  from keras.layers import Activation, Dense
  import numpy as np
  import pandas as pd
  import tensorflow as tf
  from sklearn.metrics import accuracy_score
  from tensorflow.keras.layers import Dense, Activation
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
  from sklearn.model_selection import train_test_split
  

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
  w0=weights_0.eval()
  print(w0)
  x=func_mlp(num_layers_0, epochs, batch_size, num_classes, sensor_sizes, dep, xvals_train, yvals_train)
  weights_0=tf.convert_to_tensor(x[0])
  weights_1=tf.convert_to_tensor(x[1])
  bias_0=tf.convert_to_tensor(x[2])
  bias_1=tf.convert_to_tensor(x[3])
  #w0=weights_0.eval()
  #print(w0)
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
    #print("Epoch:{0}, Train loss: {1:.2f} Train acc: {2:.3f}".format(epoch,training_loss[epoch],training_accuracy[epoch]))
   
  ## Evaluation of test data
  #for i in range(len(xvals_test)):
    #print('Actual:', yvals_test[i], 'Predicted:', np.rint(s.run(predicted_y, feed_dict={input_X: [xvals_test.iloc[i]]})))

  y_pred = np.rint(s.run(predicted_y, feed_dict={input_X: xvals_test}))

  testacc = accuracy_score(yvals_test, y_pred)
 
  print("\nTest Accuracy: {0:f}\n".format(testacc))

  w0=weights_0.eval()
  w=[]
  
  print(w0)
  #w.append(norm(w0[0:sensor_sizes[0]],2))
  for i in range(len(sensor_sizes)):
    w.append(norm(w0[cumsum[i]:cumsum[i+1]],2))
  #Feature selection
  if reduction==True:
    v=[i for i,x in enumerate(w) if x > 0.1*max(w)]
    selected=[]
    for i in v:
      selected.append(xvals.iloc[:,range(cumsum[i],cumsum[i+1])])
  
    xvals_reduced=pd.concat(selected,ignore_index=True, axis=1)
    #print(xvals_reduced)
    acc=0
    sensor_sizes_red=[sensor_sizes[i] for i in v]
    for i in range(10):
      acc=acc+func_modified2(regularizer_rate_0,regularizer_rate_1,num_layers_0, epochs, batch_size, num_classes, sensor_sizes_red,dep_cor,xvals_reduced,yvals,reduction=False)[0]
    return([acc/10,len(sensor_sizes_red),v])
  else:
    return([testacc,len(sensor_sizes)])
#%%
def func_mlp(num_layers_0, epochs, batch_size, num_classes, sensor_sizes, dep, xvals, yvals):

  from numpy.linalg import norm
  import numpy as np
  import pandas as pd
 
  import tensorflow as tf
  from sklearn.metrics import roc_auc_score, accuracy_score, precision_score
  from tensorflow.keras.layers import Dense, Activation
  from tensorflow.keras.utils import to_categorical
  import tensorflow.compat.v1 as tf
  tf.disable_v2_behavior()
  from sklearn.model_selection import train_test_split
  

  xvals_train, xvals_test,yvals_train, yvals_test = train_test_split(xvals,yvals,
                                   random_state=104, 
                                   test_size=0.2, 
                                   shuffle=True)
  

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

  ##calculate penalty term
  series = pd.Series(sensor_sizes)
  cumsum = series.cumsum()
  
  cumsum =[0]+ list(series.cumsum())
  

  loss = tf.reduce_mean(tf.square(predicted_y-tf.convert_to_tensor(yvals_train, dtype=tf.float32))) 


  ## Variable learning rate
  learning_rate = tf.train.exponential_decay(starter_learning_rate, 0, 5, 0.85, staircase=True)
  ## Adam optimzer for finding the right weight
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=[weights_0,weights_1,
                                                                         bias_0,bias_1])    
  
  
  #s.run(tf.initialize_all_variables)
  s.run(tf.compat.v1.global_variables_initializer())
  for epoch in range(epochs):    
    arr = np.arange(nrow)
    np.random.shuffle(arr)
    for index in range(0,nrow,batch_size):
        s.run(optimizer, {input_X: xvals_train,
                          input_y: yvals_train})
        

  w0=weights_0.eval()
  w=[]
  #w.append(norm(w0[0:sensor_sizes[0]],2))
  for i in range(len(sensor_sizes)):
    w.append(norm(w0[cumsum[i]:cumsum[i+1]],2))
  print(w)
  return([weights_0.eval(),weights_1.eval(), bias_0.eval(),bias_1.eval()])
#%%
#function for dependency between m th and n th sensor

def dep_cor(rsq_mat,sensor,m,n):
  
  series = pd.Series(sensor)
  cumsum = list(series.cumsum())
  cumsum=[0]+cumsum
  ind1=list(range(cumsum[m-1],cumsum[m]))
  ind2=list(range(cumsum[n-1],cumsum[n]))
  cor=rsq_mat.iloc[ind1,ind2]
  return(min(cor.apply(max,1)))

#%%
iris=pd.read_csv('/Users/aytijhyasaha/Documents/datasets/datasets/Iris.csv')
xvals = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].astype(np.float32)
yvals = iris['Species']
yvals=to_categorical(np.asarray(yvals.factorize()[0]))
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for the number of nodes in the hidden layer
param_grid = {'hidden_layer_sizes': np.arange(1, 20, 2)}

# Create a MLPClassifier object
mlp = MLPClassifier(max_iter=1000)

# Create a GridSearchCV object
grid_search = GridSearchCV(mlp, param_grid, cv=10, scoring='accuracy',verbose = 3)

# Fit the GridSearchCV object to the data
grid_search.fit(xvals, yvals)

# Print the best number of nodes in the hidden layer
print("Best number of nodes in hidden layer: ", grid_search.best_params_)

result2=[]
for i in [0,2,5]:
  for j in [0,2,5]:
    x=func_modified2(i,j,grid_search.best_params_['hidden_layer_sizes'],400,10,3,[2,2],dep_cor,xvals,yvals,True)
    result2.append([i,j,x[0],x[1],x[2]])

result2_iris1=pd.DataFrame(result2)
result2_iris1.columns =["Lambda","Mu", "Test Accuracy", "Number of sensors selected","Selected sensors"]
writer2 = pd.ExcelWriter('output_2_iris1.xlsx')
# write dataframe to excel
result2_iris1.to_excel(writer2)
# save the excel
writer2.save()

#%%
f1=iris['SepalLengthCm']
f2=iris['SepalWidthCm']
f3=iris['PetalLengthCm']
f4=iris['PetalWidthCm']
e1=f1+np.random.normal(loc=0.0, scale=0.05, size=150)
e2=f3+np.random.normal(loc=0.0, scale=0.05, size=150)
e3=f4+np.random.normal(loc=0.0, scale=0.05, size=150)
xvals=pd.concat([f1,f2,e1,f3,f4,e2,e3], axis=1, ignore_index=True).astype(np.float32)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for the number of nodes in the hidden layer
param_grid = {'hidden_layer_sizes': np.arange(1, 20, 2)}

# Create a MLPClassifier object
mlp = MLPClassifier(max_iter=1000)

# Create a GridSearchCV object
grid_search = GridSearchCV(mlp, param_grid, cv=10, scoring='accuracy',verbose = 3)

# Fit the GridSearchCV object to the data
grid_search.fit(xvals, yvals)

# Print the best number of nodes in the hidden layer
print("Best number of nodes in hidden layer: ", grid_search.best_params_)

result2=[]
for i in [0,2,5]:
  for j in [0,2,5]:
    x=func_modified2(i,j,grid_search.best_params_['hidden_layer_sizes'],500,10,3,[2,3,2],dep_cor,xvals,yvals,True)
    result2.append([i,j,x[0],x[1],x[2]])

result2_iris2=pd.DataFrame(result2)
result2_iris2.columns =["Lambda","Mu", "Test Accuracy", "Number of sensors selected","Selected sensors"]


writer2 = pd.ExcelWriter('output_2_iris2.xlsx')
# write dataframe to excel
result2_iris2.to_excel(writer2)
# save the excel
writer2.save()

#%%
data=pd.read_csv('//Users/aytijhyasaha/Documents/datasets/datasets/GasSensor(cleaned in R).csv',header=None)
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
yvals=to_categorical(np.asarray(yvals.factorize()[0]))

xvals=data.iloc[:,1:129].astype(np.float32)
from scipy.stats import zscore
for i in range(128):
  xvals[i+2]=zscore(xvals[i+2])
xvals
from itertools import repeat
# Define the parameter grid for the number of nodes in the hidden layer
param_grid = {'hidden_layer_sizes': np.arange(1, 20, 2)}

# Create a MLPClassifier object
mlp = MLPClassifier(max_iter=1000)

# Create a GridSearchCV object
grid_search = GridSearchCV(mlp, param_grid, cv=10, scoring='accuracy',verbose = 3)

# Fit the GridSearchCV object to the data
grid_search.fit(xvals, yvals)

# Print the best number of nodes in the hidden layer
print("Best number of nodes in hidden layer: ", grid_search.best_params_)

result2=[]
for i in [0,2,5]:
  for j in [0,2,5]:
    x=func_modified2(i,j,19,500,100,6,list(repeat(8,16)),dep_cor,xvals,yvals,True)
    result2.append([i,j,x[0],x[1],x[2]])

result2_gs=pd.DataFrame(result2)
result2_gs.columns =["Lambda","Mu", "Test Accuracy", "Number of sensors selected","Selected sensors"]

writer2 = pd.ExcelWriter('output_2_gs.xlsx')
# write dataframe to excel
result2_gs.to_excel(writer2)
# save the excel
writer2.save()

