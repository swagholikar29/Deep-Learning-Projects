import numpy as np
from cmath import inf

def linear_regression (X, y_tr):
    X_X_T = np.matmul(X, X.T) #XX.T = X.TX According AB = B.T.A.T
    X_y = np.matmul(X, y_tr)
    return np.linalg.solve(X_X_T, X_y)

def mean_square_error(y_hat, y):
        sq_error = np.square(y_hat-y)
        mse = (np.mean(sq_error))/2
        return mse

def grad_b(X, w, bias, y):
    grad_Fmseb = np.mean((np.dot(X.T,w)+bias-y))
    return grad_Fmseb

def grad_w(X, w, bias, y, alpha):
    #here gredient with respect to w 
    diff_with_bias = (np.dot(X.T,w))+bias-y
    error = (np.dot(X,diff_with_bias))/X.shape[1]
    L2reg = (alpha*w)/X.shape[1]
    grad_Fmsew= error + L2reg
    return grad_Fmsew

def Fmse(X, w, bias, y):
    err_sq = np.square(np.dot(X.T, w) + bias - y)
    Fmse = (np.mean(err_sq)/(2))
    return Fmse

def SGD_flow(num_epoch, epsilon, alpha, mini_batch_size, X_train, Y_train, w, bias):
    for i in range(num_epoch):
        batches=int((len(X_train.T)/mini_batch_size))
        # batches=2
        start = 0
        end = mini_batch_size
        for j in range(batches):         
            
            ip_mini_batch= X_train[:,start:end]
            label_mini_batch = Y_train[start:end]
        
            dw = grad_w(ip_mini_batch, w, bias, label_mini_batch, alpha)
            db = grad_b(ip_mini_batch, w, bias, label_mini_batch)

            new_w = w - (np.dot(epsilon, dw))
            new_bias = bias - (np.dot(epsilon, db))
        
            start = end
            end = end + mini_batch_size
            w = new_w
            bias = new_bias
    
        Fmse_epoch = Fmse(X_train, w, bias, Y_train)
      
    return [Fmse_epoch, w, bias]

def train_age_regressor():
    # Load data
    X_tr = np.reshape(np.load('age_regression_Xte.npy'), (-1, 48*48))
    ytr = np.load('age_regression_ytr.npy')
    X_te = np.reshape(np.load('age_regression_Xte.npy'), (-1, 48*48))
    yte = np.load('age_regression_yte.npy')

    X = X_tr.T

    #Division of Dataset 80% Training Dataset and 20% validation dataset 
    indx = np.random.permutation(X.shape[1])
    # print("X shape [1] ",X.shape[1])

    X_train = X[:,indx[:int(X.shape[1]*0.8)]]
    X_valid = X[:,indx[int(X.shape[1]*0.8):]]
    
    # print("X_train.shape,X_valid.shape",X_train.shape,X_valid.shape)

    Y_train = ytr[indx[:int(X.shape[1]*0.8)]]
    Y_train = np.atleast_2d(Y_train).T
    Y_valid = ytr[indx[int(X.shape[1]*0.8):]]
    Y_valid = np.atleast_2d(Y_valid).T
    print(Y_train.shape)
    print(Y_valid.shape)

    #Random values of 
    w = np.random.randint(-10,10,int(X.shape[0]))
    w = np.atleast_2d(w).T
    # b=np.random.randn(1,1)
    bias = 0
    
    # print("shape of w and b",w.shape)

    num_epoch = [100,250,500,1000]
    epsilon = [0.0001,0.0005,0.0008,0.0006]
    alpha=[0.5,2,5,10]
    mini_batch_size=[25, 50, 75, 100]

    Fmse_min=inf

    for m in num_epoch:
        print("epochs",m)
        for n in epsilon:
            for o in alpha:
                for p in mini_batch_size:
                    #Fmse=Stochastic_gradient_descent(epochs,learning_rate,alpha,mini_batch_size,X_train,Y_train,w,b)
                    mse_, w, b = SGD_flow(m,n,o,p,X_train,Y_train,w,bias)
                    mse_valid = Fmse(X_valid,w,b,Y_valid)
                    if mse_valid<Fmse_min:
                        print("Lowest Fmse for epochs (m)",m," Learning rate",n,"alpha", o," mini_batch_size",p)
                        Min_FMSE = mse_valid
                        print("Min_FMSE",Min_FMSE)
                        hp_set = [m,n,o,p]
                        min_mse = mse_valid
    
    final_num_epoch = hp_set[0]
    final_epsilon = hp_set[1]
    final_alpha = hp_set[2]
    final_mini_bs = hp_set[3]

    #Random values of 
    w = np.random.randint(-10,10,int(X.shape[0]))
    w = np.atleast_2d(w).T

    # bias = np.random.randn(1,1)
    b = 0
    Fmse_train, weights, b= SGD_flow(final_num_epoch, final_epsilon, final_alpha, final_mini_bs, X_train, Y_train, w, bias)
    Fmse_test = Fmse(X_te.T, weights, b, yte)
    print("fMSE on Testing dataset is: ",Fmse_test)
    return ...

train_age_regressor()