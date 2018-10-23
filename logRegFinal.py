
# coding: utf-8

# In[125]:

import numpy as np
import load_dataset
import timeit


# In[126]:

y_train , x_train= load_dataset.read("training","MNIST")
y_test, x_test= load_dataset.read("testing","MNIST")
x_train = x_train.reshape([60000,28*28])
x_test = x_test.reshape([10000,28*28])
number_of_classes = 10


# In[127]:

w = np.zeros((number_of_classes-1,x_train.shape[1]))


# In[128]:

def p (l , w , x):
    sum = 0.0
    for j in range(number_of_classes-1):
        sum += float(np.exp(np.dot(w[j],x)))
    #print sum
    if l == number_of_classes-1:
        return 1.0/(1.0 + sum)
    else:
        return float(np.exp(np.dot(w[l],x)))/(1.0 + sum)


# In[129]:

def probablity (label , x):
    ans =[]
    for i in range(x.shape[0]):
        sum = 0.0
        for j in range(number_of_classes-1):
            sum += np.exp(np.dot(w[j],x[i]))
        
        if label == number_of_classes-1:
            ans.append(1.0/(1.0 + sum))
        else:
            ans.append( float(np.exp(np.dot(w[label],x[i])))/(1.0 + sum))
    
    return np.array(ans)


# In[ ]:




# In[130]:

iterationAccuracy =[]


# In[131]:

learning_rate = 0.00001


# In[132]:

x_scaled = x_train/255
x_test_scaled = x_test/255


# In[ ]:




# In[133]:

#probs = probablity(2,x_scaled)


# In[134]:

#def select (x)


# In[135]:

def evaluation(iteration):
    ct =0
    cf =0 
    predict = np.zeros(y_test.shape)
    for i in range(x_test_scaled.shape[0]):
        temp = -1
        for l in range(number_of_classes):
            tt = p(l,w,x_test_scaled[i])
            if temp < tt:
                temp = tt
                predict[i] = l
        if predict[i] == y_test[i]:
            ct+=1
        else:
            cf += 1
            #print str(predict[i])+"  "+str(y_test[i])
    efficiency = float(ct)/float((ct+cf))
    print ("for "+str(iteration)+" iteration accuracy is "+str(efficiency))
    iterationAccuracy.append((iteration,efficiency))
    return efficiency


# In[136]:

for iteration in range(100):
    
    #tic = timeit.default_timer()
    for label in range(number_of_classes-1):
        theta = np.where(y_train == label,1,0)
        errors = theta - probablity(label,x_scaled)
        
        for i in range(x_scaled.shape[1]):
            sum = x_scaled[:,i] * errors
            w[label][i] += learning_rate * sum.sum()
    evaluation(iteration)  
    
    
    #if iteration==90 or iteration==94 or iteration==96 or iteration==98  :
    #   print("learning rate change")
    #   learning_rate*=0.5
      
    #toc = timeit.default_timer()
    
    #print (w)


# In[137]:

import matplotlib.pyplot as plt


# In[138]:

plt.ylabel("accuracy")
plt.xlabel("number of iterations")
x = []
y = []
for i in range(len(iterationAccuracy)):
    x.append(iterationAccuracy[i][0]+1)
    y.append(iterationAccuracy[i][1])
    
plt.title("logestic regression Learning Curve")
plt.plot(x,y)
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:



