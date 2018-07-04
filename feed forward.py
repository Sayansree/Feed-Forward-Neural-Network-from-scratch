import numpy as np 
import time
import math 
""" implement most simple neural networks the feed forward neural net from scrach and train it to do binary xor operation
devloper-->sayansree paria"""
class FeedForwawdNetwork:
    def __init__(self):
        """dfault constructor"""
        pass
    
    def fit(self,input,output):
        """to input our training dataset"""
        input=np.append(input,np.ones((input.shape[0],1)),axis=1)           #add 1's for holding threshold
        self.output=output                                                  #store our output labels
        self.input=input                                                    #store our training dataset
    
    def sig_activate(self,x):
        """sigmoid activation function """
        return 1/(1+np.exp(-x))                                             #return sigmoid probanility

    def signum_gradient(seld,output):
        """returns the partial derivative of output wrt input in a sigmoid activation function"""
        return output*(1-output)                                            #returns gradient through sigmoid while propagating backwards
    
    def train(self,hidden_layer_size=10,iteration=10000,rate=1,interval=5000,threshold_error=1e-8):
        """train your dataset with k neurons in hidden layer """ 
        rnd=-int(math.log10(threshold_error))
        synaptic_weight_1=2*np.random.random((self.input.shape[1],hidden_layer_size))-1         #first layer hidden layer weights
        synaptic_weight_2=2*np.random.random((hidden_layer_size,self.output.shape[1]))-1        #second layer output layer weights

        start_time = time.time()                                                
        prev_error=100.0
        prev_time=start_time

        for j in range(iteration+1):
            layer0=self.input                                                   #input layer is our tensor dataset
            layer1=self.sig_activate(np.dot(layer0,synaptic_weight_1))          #pass tensor through hiddenlayer
            layer2=self.sig_activate(np.dot(layer1,synaptic_weight_2))          #pass tensor through output layer

            if j%interval==0:                                                   #print progrss in some interval
                error=self.variance_error_percent(layer2,self.output)           #calculate error in percentage variance
                t=time.time()                                                   #get current time
                decent_rate=(prev_error-error)/(t-prev_time)           #error decent rate per second
                dt=t-start_time                                                 #time elapsed
                s,m,h=round(dt%60,3),dt//60%60,dt//3600                         #convert into hours minutes seconds
                prev_error=error                                                #store error for next reference 
                prev_time=t                                                     #store time for next reference 
                # print all progress stats
                print("iteration:{0} , error:{1}% , time elapsed:{3}h {4}m {5}s , rate:{6} , \t\ttraining:{2}%".format(j,round(error,rnd),round(j*100/iteration,2),h,m,s,round(decent_rate,rnd+2)))
                if(error<threshold_error):
                    print("error minimised below threshold")
                    break
                

            #now after calculating expected outputs
            # we backpropagate through all layers apply Stochistic gradient decnet algorithm to all synaptic weights

            layer2_error= self.output - layer2                                  #error in output layer                            
            error_gardient2=layer2_error*self.signum_gradient(layer2)           #gradient through outputlayer(note:chain rule)

            layer1_error=np.dot(error_gardient2,synaptic_weight_2.T)            #error hidden layer
            error_gardient1=layer1_error*self.signum_gradient(layer1)           #gradient through hidden layer(note: chain rule)

            synaptic_weight_2 += rate * np.dot(layer1.T,error_gardient2)        #update output layer weights
            synaptic_weight_1 += rate * np.dot(layer0.T,error_gardient1)        #update hidden layer weights
        
        self.synaptic_weight_1=synaptic_weight_1                       #after training we save the output layer weights
        self.synaptic_weight_2=synaptic_weight_2                       #after training we save the hidden layer weights
    
    def predict(self,Input,Output=None):
        """ to predict output of new data or determine error in a given data when output label is defined"""
        Input=np.append(Input,np.ones((Input.shape[0],1)),axis=1)           #add 1's to hold threshold values
        layer0=Input                                                        #input layer is data
        layer1=self.sig_activate(np.dot(layer0,self.synaptic_weight_1))     #pass tensor through hidden layer
        layer2=self.sig_activate(np.dot(layer1,self.synaptic_weight_2))     #pass tensor through output layer
        if Output is None:                                                  #if output labels aren't defined                                                                             
            return layer2                                                   #return the output prediction
        return self.variance_error_percent(layer2,Output)                   #else we return error percentage (variance)

    def variance_error_percent(self,calculated_output,desired_output):
        """returns the variance of error in terms of percentage"""
        performance = (calculated_output-desired_output)**2                 #squares of errors
        sum = np.sum(np.sum(performance,axis=1),axis=0)                     #sum of all errors
        return 100*sum/(performance.shape[0]*performance.shape[1])          #return percentage error

#some input data, boolean combinations


X=np.array([[0,0,0,0],
            [0,0,0,1],
            [0,0,1,0],
            [0,0,1,1],
            [0,1,0,0],
            [0,1,0,1],
            [0,1,1,0],
            [0,1,1,1],
            [1,0,0,0],
            [1,0,0,1],
            [1,0,1,0],
            [1,0,1,1],
            [1,1,0,0],
            [1,1,0,1],
            [1,1,1,0],
            [1,1,1,1]])
#xor operation rowwise
XOR=np.array(np.dot(X,np.ones((X.shape[1],1)))%2)

#test our model wheather it learns xor operations

feed_forward_ann=FeedForwawdNetwork()           #initiallise instance of feed forward neural network
feed_forward_ann.fit(X,XOR)                     #pass our training data to model
feed_forward_ann.train(hidden_layer_size=6,iteration=700000,rate=4.5,threshold_error=1e-6)           #train for 70,000 iteration and state hidden layer contains 6 neurons learning rate 4.5
output=feed_forward_ann.predict(X,XOR)          #test performance on our training data itself
print("finally error is {0}%".format(output))   #print our final error
print(feed_forward_ann.predict(X))
