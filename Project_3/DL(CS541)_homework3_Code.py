import math
import numpy as np
from sklearn.model_selection import train_test_split
from cmath import inf

def Fce(X,Y,w,b,alpha):
    Z=np.dot(X.T,w)+b        
    exp_Z=np.exp(Z)   
    exp_Z_mean=np.reshape(np.sum(exp_Z,axis=1),(-1,1))
    Yhat=exp_Z/exp_Z_mean   
    logYhat=np.log(Yhat)
    loss=-np.sum(Y*logYhat)/X.shape[1]
    Fce=loss
    return [Fce,Yhat]

def softmax(z):
    
    # z--> linear part.
    
    # subtracting the max of z for numerical stability.
    exp = np.exp(z - np.max(z))
    
    # Calculating softmax for all examples.
    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])
        
    return exp

def grad_w(X, w, bias, y, alpha):
    #here gredient with respect to w
    z=X.T@w + bias 
    y_hat=softmax(z)
    diff_with_bias = y_hat-y
    error = (np.dot(X,diff_with_bias))/X.shape[1]
    L2reg = (alpha*w)/X.shape[1]
    grad_Fcew= error + L2reg
    return grad_Fcew

def grad_b(X, w, bias, y):
    #here gredient with respect to b
    z=X.T@w + bias 
    y_hat=softmax(z)
    diff_with_bias = y_hat-y
    grad_Fceb = np.mean(diff_with_bias)
    return grad_Fceb

def model_flow(num_epoch, epsilon, alpha, mini_batch_size, X_train, Y_train, w, bias):

    for i in range(num_epoch):
        #print("Epoch no:",i,"out of",num_epochs)
        batches = int((len(X_train.T)/mini_batch_size))
        
        # batches=2
        start = 0
        end = mini_batch_size
        for j in range(batches):         
            ip_mini_batch= X_train[:,start:end]
            label_mini_batch = Y_train[start:end]
          
            #grad_w,grad_b = gredient_w_b(IP_mini_batch,w,b,OP_mini_batch,alpha)
            #dw, db = gradients(IP_mini_batch, w, bias, OP_mini_batch, alpha)
            dw = grad_w(ip_mini_batch, w, bias, label_mini_batch, alpha)
            db = grad_b(ip_mini_batch, w, bias, label_mini_batch)

            new_w = w - (np.dot(epsilon, dw))
            new_bias = bias - (np.dot(epsilon, db))
            
            start = end
            end = end + mini_batch_size
            #w = w_values
            w = new_w
            #print(w)
            #b = b_values #
            bias = new_bias
            #print(bias)
    
        #Fce_each_epoch,_=Fce(X_train,Y_train,w,b,alpha)
        fCE_each_epoch,_ = Fce(X_train,Y_train,w,bias,alpha)
        #bb=(alpha/2)*(np.sum(np.dot(w.T,w)))
        
        reg_term = (alpha/2)*(np.sum(np.dot(w.T,w)))
        #print(reg_term)
        
        #Fce_each_epoch+=bb
        fCE_each_epoch+=reg_term
        #print("Fce_each_epoch: ",Fce_each_epoch)
        print("Fce_each_epoch: "+str(i),fCE_each_epoch)
      
    #return [Fce_each_epoch,w,b]
    return [fCE_each_epoch,w,bias]

def softmax_regression():
  #Load data
  X_tr = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 28*28))
  ytr = np.load("fashion_mnist_train_labels.npy")
  X_te = np.reshape(np.load("fashion_mnist_test_images.npy"), (-1, 28*28))
  yte = np.load("fashion_mnist_test_labels.npy")

  #Divide training data into training and validation set (80%-20%)
  X_train, X_val, y_train, y_val = train_test_split(X_tr, ytr, test_size=0.2, shuffle=True)

  #print(X_train.shape)
  #print(X_val.shape)
  #print(y_train.shape)
  #print(y_val.shape)

  #Transpose for dimension agreement
  X_train = X_train.T
  X_val = X_val.T
  X_test = X_te.T

  #One hot encoding
  Y_train=np.zeros((len(y_train), 10))
  Y_train[np.arange(len(y_train)), y_train]=1
    
  Y_valid=np.zeros((len(y_val), 10))
  Y_valid[np.arange(len(y_val)), y_val]=1
    
  Y_test=np.zeros((len(yte), 10))
  Y_test[np.arange(len(yte)), yte]=1

  #assigning random weights and bais
  w=np.random.randint(0,1,int(X_train.shape[0]))
  w=np.atleast_2d(w).T

  bias=np.random.random(10)

  #Defining hyperparameters list
  #num_epoch = [100,250,500,1000]
  #epsilon = [0.0001,0.00025,0.0005,0.00075]
  #alpha=[0.5,2,5,10]
  # mini_batch_size=[16, 32, 64, 128]

  num_epoch=[1,2,3,4]
  epsilon=[0.000003,0.000004,0.000005,0.000006]
  alpha=[2,3,4,5]
  mini_batch_size=[600,400,200,100]

  #Assigning Fce_min=inf
  fCE_min=inf

  #Running nested loops for hyperparameters
  for m in num_epoch:
        print("epochs",m)
        for n in epsilon:
            for o in alpha:
                for p in mini_batch_size:
                    #mse_, w, b = model_flow(m,n,o,p,X_train,Y_train,w,bias)
                    #mse_valid,_ = Fmse(X_valid,Y_valid,w,b,o)
                    CEL_, w, bias = model_flow(m,n,o,p,X_train,Y_train,w,bias)
                    Fce_valid,_ = Fce(X_val,Y_valid,w,bias,o)

                    print("Fmse_valid",Fce_valid)
            
                    if Fce_valid<fCE_min:
                        print("Lowest Fmse for epochs (m)",m," Learning rate",n,"alpha", o," mini_batch_size",p)
                        Min_Fce=Fce_valid
                        print("Min_FCE",Min_Fce)
                        Hyper_para=[m,n,o,p]
                        Fce_min=Fce_valid

  best_epoch = Hyper_para[0]
  best_lr = Hyper_para[1]
  best_alpha = Hyper_para[2]
  best_mini = Hyper_para[3]

  Fce_train,weights,bias= model_flow(best_epoch,best_lr,best_alpha,best_mini,X_train,Y_train,w,bias)
  Fce_test,Yhat = Fce(X_test,Y_test,weights,bias,best_alpha)

  Yhat=np.argmax(Yhat,1)
  accuracy = 100*np.sum(yte == Yhat)/X_te.shape[0]
  print("Minimum Fce on validation set",Min_Fce)
  print("Fce on test set is: ",Fce_test)
  print("Best Hyperparameters are: epochs (m)",m," Learning rate",n,"alpha", o," mini_batch_size",p)
  print("Accuracy percentage",accuracy)
  return ...

softmax_regression()