from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
(x_train,y_train),(x_test,y_test) = mnist.load_data()
class MNIST:
    def __init__(self,inputs,hidden,outputs):
        self.w_ih=np.random.normal(0,0.1,(hidden, inputs))
        self.w_ho=np.random.normal(0,0.1,(outputs, hidden))
    def train(self,inputs_list, outputs_list):
        hid_results=1/(1+np.exp(-(np.dot(self.w_ih,np.array(inputs_list,ndmin=2).T))))
        out_results=1/(1+np.exp(-(np.dot(self.w_ho,hid_results))))
        self.w_ho+=0.1*np.dot((np.array(outputs_list,ndmin=2).T-out_results)*out_results*(1.0-out_results),hid_results.T)
        hid_errors=np.dot(self.w_ho.T,(np.array(outputs_list,ndmin=2).T-out_results))
        self.w_ih+=0.1*np.dot(hid_errors*hid_results*(1.0-hid_results),np.array(inputs_list,ndmin=2))
    def test(self,inputs_list):
        return 1/(1+np.exp(-(np.dot(self.w_ho,1/(1+np.exp(-(np.dot(self.w_ih,np.array(inputs_list,ndmin=2).T))))))))
mn = MNIST(784,100,10)
for i in range(10):
    [mn.train(np.array(x_train[n]/255).reshape(784),np_utils.to_categorical(y_train[n],10)) for n in range(int(x_train.shape[0]))]
    print("Epoch №",i+1,", Test - ",sum([1 for n in range(int(x_test.shape[0])) if (mn.test(np.array(x_test[n]/255).reshape(784))).argmax() == y_test[n]])/10000)
    
