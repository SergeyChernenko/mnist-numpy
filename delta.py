from keras.datasets import mnist
import numpy as np
import cv2 as cv

(x_train,y_train),(x_test,y_test) = mnist.load_data()

class MNIST:
    def __init__(self,inputs,outputs):
        self.w=np.random.normal(0,0.1,(outputs, inputs))
        
    def train(self,inputs_list, outputs_list):
        inputs=np.array(inputs_list,ndmin=2).T
        outputs=np.array(outputs_list,ndmin=2).T
        out_results=1/(1+np.exp(-(np.dot(self.w,inputs))))
        out_errors=(outputs-out_results)
        delta=self.lr*np.dot(out_errors,inputs.T)
        self.w+=delta
        
    def test(self,inputs_list):
        inputs=np.array(inputs_list,ndmin=2).T
        out_results=1/(1+np.exp(-(np.dot(self.w,inputs))))
        return out_results
    
    def set_learning_rate(self,rate):
        self.lr=rate

mn = MNIST(784,10)

def epoch_train(lr):
    mn.set_learning_rate(lr)
    for n in range(int(x_train.shape[0])):
        target=np.zeros(10)
        target[y_train[n]]=1
        ret,binar = cv.threshold(x_train[n],127,255,cv.THRESH_BINARY)
        inputs=np.array(binar/255).reshape(784)
        mn.train(inputs,target)
        
def epoch_test():
    precision=0
    for n in range(int(x_test.shape[0])):
        ret,binar = cv.threshold(x_test[n],127,255,cv.THRESH_BINARY)
        inputs=np.array(binar/255).reshape(784)
        ans=mn.test(inputs)
        if ans.argmax() == y_test[n]:
            precision+=1
    return precision/(n)  
    
lr=0.2
epoch=20
for i in range(epoch):
    lr=lr/1.3
    epoch_train(lr)
    print("Epoch №",i+1,", Accuracy on the test array ",epoch_test())
    

