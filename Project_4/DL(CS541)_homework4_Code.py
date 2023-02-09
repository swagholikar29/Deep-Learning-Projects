import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from cmath import inf
import scipy.optimize

# For this assignment, assume that every hidden layer has the same number of neurons.
NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = 10
NUM_OUTPUT = 10

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z), axis=0, keepdims=True)

def relu(forward):
    relu = forward.copy()
    relu[relu < 0] = 0
    return relu

def relu_diff(backward):
    diff = backward.copy()
    diff[diff <= 0] = 0
    diff[diff > 0] = 1
    return diff

# Unpack a list of weights and biases into their individual np.arrays.
def unpack(weightsAndBiases, hidden_num, hidden_layers):
    Ws = []
    start = 0
    end = NUM_INPUT * hidden_num
    W = weightsAndBiases[start:end]
    Ws.append(W)

    for i in range(hidden_layers - 1):
        start = end
        end = end + hidden_num * hidden_num
        W = weightsAndBiases[start:end]
        Ws.append(W)

    start = end
    end = end + hidden_num * NUM_OUTPUT
    W = weightsAndBiases[start:end]
    Ws.append(W)

    Ws[0] = Ws[0].reshape(hidden_num, NUM_INPUT)
    for i in range(1, hidden_layers):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(hidden_num, hidden_num)
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, hidden_num)

    # Bias terms
    bs = []
    start = end
    end = end + hidden_num
    b = weightsAndBiases[start:end]
    bs.append(b)

    for i in range(hidden_layers - 1):
        start = end
        end = end + hidden_num
        b = weightsAndBiases[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weightsAndBiases[start:end]
    bs.append(b)

    return Ws, bs

def forward_prop(X, Y, weightsAndBiases, NUM_HIDDEN, hidden_layers):
    Hs = []
    Zs = []

    Ws, bs = unpack(weightsAndBiases, NUM_HIDDEN, hidden_layers)
    h = X   #for first layer
 
    for i in range(hidden_layers):
        b = bs[i].reshape(-1, 1)
        Z = np.dot(Ws[i], h) + b
        Zs.append(Z)
        h = relu(Z)
        Hs.append(h)

    bl= bs[-1].reshape(-1, 1)
    zz = np.dot(Ws[-1], Hs[-1]) + bl
    Zs.append(zz)

    y_hat = softmax(zz)
    loss = np.sum(np.log(y_hat) * Y)
    loss = (-1/Y.shape[1]) * loss

    # Return loss, pre-activations, post-activations, and predictions
    return loss, Zs, Hs, y_hat

def back_prop(X, Y, weightsAndBiases, NUM_HIDDEN, hidden_layers):
    loss, Zs, Hs, yhat = forward_prop(X, Y, weightsAndBiases, NUM_HIDDEN, hidden_layers)
    dJdWs = []  # Gradients w.r.t. weights
    dJdbs = []  # Gradients w.r.t. biases

    Ws, bs = unpack(weightsAndBiases, NUM_HIDDEN, hidden_layers)
    g = yhat - Y

    for i in range(hidden_layers, -1, -1):
        # For grads of b 
        if i != hidden_layers:               #at last layer G =Yhat-Y
            dhdzs = relu_diff(Zs[i])
            g = dhdzs * g

        djdb = np.sum(g, axis=1) / Y.shape[1]
        dJdbs.append(djdb)

        # For grads of W 

        if i == 0:
            fst_layer=np.dot(g, X.T) / Y.shape[1]  #at first layer we multiply with input values X 
            dJdWs.append(fst_layer)
            
        else:
            dJdWs.append(np.dot(g, Hs[i - 1].T) / Y.shape[1])    #G term multipled with privious term h thats why i-1  
     
        g = np.dot(Ws[i].T, g)  #updated G for next layer

    dJdbs.reverse() #Generated list is from last layer to first layer 
    dJdWs.reverse() #reverse to makes it first to end layer
    # Concatenate gradients
    return np.hstack([dJdW.flatten() for dJdW in dJdWs] + [dJdb.flatten() for dJdb in dJdbs])

def accuracy(y_hat, y):
    y_hat = y_hat.T
    y = y.T

    Y_hat = np.argmax(y_hat,1)
    Y = np.argmax(y,1)
    accuracy = 100*np.sum(Y == Y_hat)/y.shape[0]

    return accuracy

def Update_W_B(W, B, grad_w, grad_b, epsilon, alpha, Y_train):
    for i in range(len(W)):
        W[i] = W[i]-(epsilon*grad_w[i])+(alpha * W[i] /Y_train.shape[1])
        B[i] = B[i]-(epsilon*grad_b[i])
    return W, B

def train(X_train, Y_train, weightsAndBiases, hidden_num, hidden_layers, epsilon,alpha):
    trajectory = []

    bp = back_prop(X_train, Y_train, weightsAndBiases, hidden_num, hidden_layers)
    grad_w, grad_b = unpack(bp, hidden_num, hidden_layers)
    W, B = unpack(weightsAndBiases, hidden_num, hidden_layers)
    W, B = Update_W_B(W, B,grad_w, grad_b,epsilon,alpha,trainY)

    weightsAndBiases = np.hstack([w.flatten() for w in W] + [b.flatten() for b in B])
    trajectory.append(weightsAndBiases)

    return weightsAndBiases, trajectory

def SGD_flow(X_train, Y_train, epochs, batch_size, weightsAndBiases, hidden_num, hidden_layers, learning_rate, alpha, X_valid, Y_valid):
    print("epochs",epochs)
    TRAJECT = []
    for epoch in range(epochs):
        print("epoch",epoch)
                           
        N_batches=int((len(X_train.T)/batch_size))
     
        init = 0
        end = batch_size

        for i in range(N_batches):
            mini_batch = X_train[:,init:end]          
            y_mini_batch = Y_train[:,init:end]
            
            weightsAndBiases, trajectory = train(mini_batch, y_mini_batch, weightsAndBiases, hidden_num, hidden_layers, learning_rate, alpha)
            
            init=end
            end=end+batch_size

            if i % 10 == 0:   # sampled every 50 batches to get how the weights evolve 
                    TRAJECT.extend(trajectory)  #stored all trej on mini batched to TRAJECT

        loss, yy, zz, y_hat = forward_prop(X_valid, Y_valid, weightsAndBiases, hidden_num, hidden_layers)
        acc = accuracy(y_hat, Y_valid)
        print("Loss on epoch: ", loss,"Accuracy: ",acc)
        # print(TRAJECT)
    
    return weightsAndBiases, TRAJECT

def findBestHyperparameters(trainX, trainY, testX, testY):

    #hidden_layers_list=[3,4,5]
    #hidden_numbers_list = [30,40,50]
    #mini_batch_size_list=[16,32,64]
    #epsilon_list=[0.001,0.05,0.01,0.1]
    #epochs_list=[40,60,80]
    #alpha_list=[0.00001,0.0001,0.00002]


    hidden_layers_list=[3]
    hidden_numbers_list = [50]
    mini_batch_size_list=[32]
    epsilon_list=[0.01]
    epochs_list=[60]
    alpha_list=[0.00001]
 

    change_order_idx = np.random.permutation(trainX.shape[1])
    trainX = trainX[:, change_order_idx]
    trainY = trainY[:, change_order_idx]

    indx_values = np.random.permutation(trainX.shape[1])
    train_X=trainX[:,indx_values[:int(trainX.shape[1]*0.8)]]
    valid_X=trainX[:,indx_values[int(trainX.shape[1]*0.8):]]
    train_Y=trainY[:,indx_values[:int(trainX.shape[1]*0.8)]]
    valid_Y=trainY[:,indx_values[int(trainX.shape[1]*0.8):]]

    # print("train_x, train_y, valid_x, valid_y ",train_X.shape, train_Y.shape,valid_X.shape, valid_Y.shape )

    L_min=inf
    acc_max=10

    for hidden_layers in hidden_layers_list:
        for hidden_num in hidden_numbers_list:
            for epochs in epochs_list:
                for batch_size in mini_batch_size_list:
                    for learning_rate in epsilon_list:
                        for alpha in alpha_list:

                            print("H_Layers=", hidden_layers, "hidden_num=", hidden_num,"Batch_size=", batch_size )
                            print("learning rate=",learning_rate,"epochs=", epochs,"alpha=", alpha)
                            
                            weightsAndBiases = initWeightsAndBiases(hidden_num, hidden_layers)

                            weightsAndBiases, TRAJECT=SGD_flow(train_X,train_Y,epochs,batch_size,weightsAndBiases, hidden_num, hidden_layers, learning_rate,alpha,valid_X,valid_Y)
        
                            loss, yyy, zzz, yhat = forward_prop(valid_X, valid_Y, weightsAndBiases, hidden_num, hidden_layers)
                          
                            if loss < L_min:
                                L_min = loss
                                bestHyperparameters = [hidden_layers,hidden_num,epochs,batch_size,learning_rate,alpha]

                            acc = accuracy(yhat, valid_Y)

                            if acc> acc_max:
                                Best_accuracy=acc
                                acc_max=acc

    best_hidden_layers = bestHyperparameters[0]
    best_hidden_num = bestHyperparameters[1]
    best_epochs = bestHyperparameters[2]
    best_batch_size = bestHyperparameters[3]
    best_learning_rate = bestHyperparameters[4]
    best_alpha = bestHyperparameters[5]

    weightsAndBiases = initWeightsAndBiases(best_hidden_num, best_hidden_layers)
    weightsAndBiases, trajectory=SGD_flow(train_X,train_Y,best_epochs,best_batch_size,weightsAndBiases, best_hidden_num, best_hidden_layers, best_learning_rate,best_alpha,testX, testY)
    loss, yyy, zzz, yhat = forward_prop(testX, testY, weightsAndBiases,  best_hidden_num, best_hidden_layers)

    print("\nBest Hyper Parameters:  \nBest_Hidden_Layers:",best_hidden_layers,"\nBest_hidden_num: ",best_hidden_num,"\nBest_epochs: ",best_epochs,"\nBest_batch_size: ",best_batch_size)
    print("Best_learning_rate: ",best_learning_rate,"\nBest_alpha: ",best_alpha)
    print("Best_accuracy on validation data :",Best_accuracy)

    print("\nloss value on best hyperparameters: ", loss)
    acc = accuracy(yhat, testY)
    print("\nAccuracy on Test data: ",acc)
    print("\n")
    
    return bestHyperparameters, TRAJECT, hidden_layers, hidden_num

def initWeightsAndBiases(hidden_num, hidden_layers) :
    Ws = []
    bs = []

    np.random.seed(0)
    W = 2 * (np.random.random(size=(hidden_num, NUM_INPUT)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
    Ws.append(W)
    b = 0.01 * np.ones(hidden_num)
    bs.append(b)

    for i in range(hidden_layers- 1):
        W = 2 * (np.random.random(size=(hidden_num,hidden_num)) / hidden_num ** 0.5) - 1. / hidden_num ** 0.5
        Ws.append(W)
        b = 0.01 * np.ones(hidden_num)
        bs.append(b)

    W = 2 * (np.random.random(size=(NUM_OUTPUT,hidden_num)) / hidden_num ** 0.5) - 1. / hidden_num ** 0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return np.hstack([W.flatten() for W in Ws] + [b.flatten() for b in bs])

def plotSGDPath (trainX, trainY, trajectory, hidden_layers, hidden_num):
    # TODO: change this toy plot to show a 2-d projection of the weight space
    # along with the associated loss (cross-entropy), plus a superimposed 
    # trajectory across the landscape that was traversed using SGD. Use
    # sklearn.decomposition.PCA's fit_transform and inverse_transform methods.

    def toyFunction (x1, x2):
        a=[x1,x2]
        WtAndBia= pca.inverse_transform(a)  # to get it back original
        loss, zs, hs, yhat=forward_prop(trainX, trainY, WtAndBia, hidden_num, hidden_layers)
        return loss

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    pca = PCA(n_components=2)  # 2 components
    data=pca.fit_transform(trajectory)  #reduce dims

    axis1 = np.arange(-40, 40, 2)  # Just an example
    axis2 = np.arange(-40, 40, 2) # Just an example
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))
    for i in range(len(axis1)):
        for j in range(len(axis2)):
            Zaxis[i,j] = toyFunction(Xaxis[i,j], Yaxis[i,j])
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  

    # Now superimpose a scatter plot showing the weights during SGD.
    Xaxis =data[:, 0]   # Just an example
    Yaxis =data[:, 1]   # Just an example
    Zaxis = np.zeros(len(Xaxis))
    for i in range(len(Xaxis)):
        Zaxis[i] = toyFunction(Xaxis[i], Yaxis[i])
    ax.scatter(Xaxis, Yaxis, Zaxis, color='r')

    plt.show()

if __name__ == "__main__":
    # TODO: Load data and split into train, validation, test sets
    X_tr = np.reshape(np.load("fashion_mnist_train_images.npy"), (-1, 28*28))/255
    trainX=X_tr.T
    ytr = np.load("fashion_mnist_train_labels.npy")
    train_Y=ytr
    X_te = np.reshape(np.load("fashion_mnist_test_images.npy"), (-1, 28*28))/255
    testX=X_te.T
    yte = np.load("fashion_mnist_test_labels.npy")
    test_Y=yte

    #onehot encoding
    trainY=np.zeros((train_Y.size,train_Y.max()+1))
    testY=np.zeros((test_Y.size,test_Y.max()+1))
    trainY[np.arange(train_Y.size),train_Y]=1
    testY[np.arange(test_Y.size),test_Y]=1
    trainY=trainY.T
    testY=testY.T
    # print("trainX, trainY, testX, testY",trainX.shape, trainY.shape, testX.shape, testY.shape)
    

    weightsAndBiases = initWeightsAndBiases(NUM_HIDDEN,NUM_HIDDEN_LAYERS)


    print("===========scipy.optimize.check_grad==========")
    print(scipy.optimize.check_grad(
        lambda wab: forward_prop(np.atleast_2d(trainX[:, 0:5]), np.atleast_2d(trainY[:, 0:5]), wab, NUM_HIDDEN, NUM_HIDDEN_LAYERS)[0], \
        lambda wab: back_prop(np.atleast_2d(trainX[:, 0:5]), np.atleast_2d(trainY[:, 0:5]), wab, NUM_HIDDEN, NUM_HIDDEN_LAYERS), \
        weightsAndBiases))
    print("==============================================\n")


    # weightsAndBiases, trajectory = train(trainX, trainY, weightsAndBiases, testX, testY)
    bestHyperparameters,trajectory, hidden_layers, hidden_num = findBestHyperparameters(trainX, trainY, testX, testY)


    change_order_idx = np.random.permutation(trainX.shape[1])
    trainX = trainX[:, change_order_idx]
    trainY = trainY[:, change_order_idx]

    # print("trajectory.shapet",trajectory)
    # # print("trajectory.shapet",trajectory[0].shape)
    # print("trajectory.shapet",type(trajectory[0]))
    # print("trajectory.shapet",len(trajectory[0]))
    # print("trainX.shape,trainY.shape",trainX.shape,trainY.shape)
  
    plotSGDPath(trainX[:, 0:5000], trainY[:, 0:5000], trajectory, hidden_layers, hidden_num)

