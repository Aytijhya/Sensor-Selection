from numpy.linalg import norm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

#%%



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
data_trn=pd.read_csv('/Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/Landsat_trn_modified.csv')
data_tst=pd.read_csv('/Users/aytijhyasaha/Documents/datasets/sensor-selection-datasets/Landsat_tst_modified.csv')
data_trn.drop('Unnamed: 0',axis=1,inplace=True)
data_tst.drop('Unnamed: 0',axis=1,inplace=True)
xvals_train=data_trn.iloc[:,0:44]
xvals_test=data_tst.iloc[:,0:44]
yvals_test=data_tst.iloc[:,44]
yvals_train=data_trn.iloc[:,44]

yvals_train=to_categorical(np.asarray(yvals_train.factorize()[0]))
yvals_test=to_categorical(np.asarray(yvals_test.factorize()[0]))

from scipy.stats import zscore
for i in range(36):
  xvals_test.iloc[:,i]=zscore(xvals_test.iloc[:,i])
  xvals_train.iloc[:,i]=zscore(xvals_train.iloc[:,i])


from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from itertools import repeat

# Define the parameter grid for the number of nodes in the hidden layer
param_grid = {'hidden_layer_sizes': np.arange(2, 21, 2)}

# Create a MLPClassifier object
mlp = MLPClassifier(max_iter=1000)

# Create a GridSearchCV object
grid_search = GridSearchCV(mlp, param_grid, cv=10, scoring='accuracy',verbose = 3)

# Fit the GridSearchCV object to the data
grid_search.fit(xvals_train, yvals_train)

# Print the best number of nodes in the hidden layer
print("Best number of nodes in hidden layer: ", grid_search.best_params_)

result=[]
for i in [0,20,50]:
  for j in [0,2,5]:
    x=func_modified_landsat(i,j,grid_search.best_params_['hidden_layer_sizes'],
                               700,200,6,[11,11,11,11],dep_cor,xvals_train,yvals_train,xvals_test,
                               yvals_test,True)
    result.append([i,j,x[0],x[1],x[2]])
    result_landsat=pd.DataFrame(result)
    result_landsat.columns =["Lambda","Mu", "Test Accuracy", "Number of sensors selected","Selected sensors"]
    writer = pd.ExcelWriter('output_1_landsat_new_tuning.xlsx')
    # write dataframe to excel
    result_landsat.to_excel(writer)
    # save the excel
    writer.save()


