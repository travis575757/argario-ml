import numpy as np
import sys,pygame
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf


class Trainer(object):
    
    def __init__(self):
        self.raw = [[0, 0], [1, 1], [-1, -1], [1, -1], [-1, 1], [1, 0], [-1, 0], [0, 1], [0, -1]]
        self.outputs = [8, 15, 3, 9, 5, 12, 4, 10, 6]
        self.is_trained = False
    
    def train(self):
        pass
    
    def predict(self,data):
        pass
        
    def clean_raw_data(self,data,labels): #cleans the data, currently removes points where the object is not moving or label = [0,0] takes raw data ex non scalar inputs
        res_data = []
        res_labels = []
        for i in range(len(labels)): #get rid of empty labels
            if labels[i] != [0,0] and max([max([abs(y) for y in x]) for x in data[i]]) != 0:
                res_data.append(data[i])
                res_labels.append(labels[i])
                
        print("data cleaned was ",len(data)," now ",len(res_data))
        return zip(res_data,res_labels)
                
        
    def vectorize_images(self,data): #should be fixed eventually but takes a image data set and converts to vectors instead of matrix
        result = []
        for image in data: #reshape and vectorize images
            image = np.array(image)

            #i dont know why this works, seems bugged so currectly does not support 2 different dimensions
            shape = image.shape
            x = shape[0]
            y = x
            dim = x,y
            length = dim[0] * dim[1]
            image = image.reshape(length)
            result.append(list(image))
        return result
    
    def scalar_labels(self,labels): #convert the labels to integers
        result = []
        for label in labels:
            #this might be confusing, all its doing is some math to make converting the vector into a scalar easier
            #basically their are 9 values so they are converted to a 4 bit representation
            lr = label[0] + 2
            ud = label[1] + 4
            #for reference [[0, 0], [1, 1], [-1, -1], [1, -1], [-1, 1], [1, 0], [-1, 0], [0, 1], [0, -1]] = [8, 15, 3, 9, 5, 12, 4, 10, 6]
            result.append(lr * ud)
        return result
    
    def reverse_labels(self,_labels): #convert the integer representations back into vectors
        #reverses the process of scalar_labels
        raw = [[0, 0], [1, 1], [-1, -1], [1, -1], [-1, 1], [1, 0], [-1, 0], [0, 1], [0, -1]]
        output = [8, 15, 3, 9, 5, 12, 4, 10, 6]
        result = []
        for label in _labels:
            for i in range(len(output)):
                if output[i] == label:
                    result.append(raw[i])
        return result
    
    def hot_vec(self,labels):
        result = [[0] * 9] * len(labels)
        for i in range(len(labels)):
            out = [0] * 9
            for j in range(len(self.raw)):
                if self.raw[j] == labels[i]:
                    out[j] = 1 #place 1 into the jth spot 1-9 
                    result[i] = out #add to the result list
        return result
    
    def reverse_hot(self,labels):
        result = []
        for i in range(len(labels)):
            index = np.argmax(labels[i])
            result.append(self.raw[index])
        return result
    
    
class SVMTrainer(Trainer):
    
    def __init__(self):
        Trainer.__init__(self)
        self.c = 1
        self.gamma = 1
        self.svm = SVC()
    
    def get_clf(self):
        return self.svm
    
    def train(self,data,labels):
        data,labels = zip(*self.clean_raw_data(data,labels)) #clean data
        new_data = self.vectorize_images(data)
        new_labels = self.scalar_labels(labels)
        #break down data using sklearn train test split
        train_features,test_features,train_labels,test_labels = train_test_split(new_data,new_labels,test_size=0.3,random_state=42)
        #for label in train_features:
            #print(label.shape)
        self.svm = SVC(C=self.c, kernel='rbf',gamma=self.gamma)
        self.svm.fit(train_features,train_labels)
        #print accuracy
        print(self.svm.score(test_features,test_labels))
        
    def predict(self,data):
        out = self.svm.predict(data)
        return self.reverse_labels(out)

class GridTrainer(Trainer):
        
    def __init__(self):
        Trainer.__init__(self)
        self.c = [0.00001, 0.001, 0.01, 0.1, 1]
        self.gamma = [1, 1e2, 1e3, 1e4, 1e5]
        param_grid = {}
        self.grids = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    
    def get_clf(self):
        return self.grids
    
    def train(self,data,labels):
        data,labels = zip(*self.clean_raw_data(data,labels)) #clean data
        new_data = self.vectorize_images(data)
        new_labels = self.scalar_labels(labels)
        param_grid = {
        'C': self.c,
        'gamma': self.gamma,
        }
        self.grids = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
        clf = self.grids
        train_features,test_features,train_labels,test_labels = train_test_split(new_data,new_labels,test_size=0.3,random_state=42)
        clf = clf.fit(train_features, train_labels)
        print(self.grids.score(test_features,test_labels))
        
    def predict(self,data):
        out = self.grids.predict(data)
        return self.reverse_labels(out)
    
class ForestTrainer(Trainer):
    
    def __init__(self,trees,min_samples):
        Trainer.__init__(self)
        self.trees = trees
        self.forest = RandomForestClassifier(n_estimators=trees,min_samples_split=min_samples)
        
    def get_clf(self):
        return self.forests
        
    def train(self,data,labels):
        data,labels = zip(*self.clean_raw_data(data,labels)) #clean data
        new_data = self.vectorize_images(data)
        new_labels = self.scalar_labels(labels)
        self.forest = RandomForestClassifier(n_estimators=self.trees)
        train_features,test_features,train_labels,test_labels = train_test_split(new_data,new_labels,test_size=0.3,random_state=42)
        self.forest.fit(train_features, train_labels)
        print(self.forest.score(test_features,test_labels))
        
    def predict(self,data):
        #return self.forest.predict(data)
        out = self.forest.predict(data)
        return self.reverse_labels(out)
    
class SigmoidNN(Trainer): #basic sigmoid net with 2 layers, input and output
    
    def __init__(self,nn_in,nn_out,iterations,batch_size):
        Trainer.__init__(self)
        self.iterations = iterations
        self.batch_size = batch_size
        
        self.W_data = []
        self.b_data = []
        
        #create the the basic neural net
        self.W = tf.Variable(tf.zeros([nn_in,nn_out]))
        self.b = tf.Variable(tf.zeros([nn_out]))
        self.x = tf.placeholder(tf.float32, [None,nn_in])
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b) #outputs, apply activation function to weights and biasis
        self.y_ = tf.placeholder(tf.float32, [None,nn_out])#true labels for error
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(0.25).minimize(self.cross_entropy)
        
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        self.sess.as_default()
        
    def get_clf(self):
        pass
        
    def train(self,data,labels):
        data,labels = zip(*self.clean_raw_data(data,labels)) #clean data
        new_data = self.vectorize_images(data)
        new_labels = self.hot_vec(labels) #can handle any data output due to nn structure
        train_features,test_features,train_labels,test_labels = train_test_split(new_data,new_labels,test_size=0.05,random_state=42)

        sess = self.sess #create the tf session
        #sess.as_default()
        for i in range(self.iterations):
            if ((i+1) * self.batch_size) < len(train_features):#check if out of data range
                x = train_features[(i*self.batch_size):((i+1)*self.batch_size)]
                y = train_labels[(i*self.batch_size):((i+1)*self.batch_size)]
                result = sess.run(self.train_step, feed_dict={self.x: x, self.y_: y})
                if i % self.batch_size == 0:
                    print("Processing: ",i * self.batch_size)
                
        #store data for later use
        self.W_data = sess.run(self.W)
        self.b_data = sess.run(self.b)

        print("\nW\n")
        for i in self.W_data:
            print(i)
        print("\nb\n")
        for i in self.b_data:
            print(i)

        correct_preds = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1)) #check if prediction is correct
        acc = tf.reduce_mean(tf.cast(correct_preds, tf.float32)) #get mean which is accruacy
        result_acc = sess.run(acc, feed_dict={self.x: test_features, self.y_: test_labels})
        print("sigmoid nn result: ", result_acc)
        
    def predict(self,data): #assumes data is vectorized
        sess = self.sess
        #tf.global_variables_initializer().run()
        result = sess.run(self.y, feed_dict={self.x: data, self.W: self.W_data, self.b: self.b_data})
        #to keep consistancy must convert to number
        result = self.reverse_hot(result)
        return result
    
class FFNN(Trainer): #3 layer NN, node dimensions for inner size are specified
    
    def __init__(self,nn_in,nn_out,nn_hdim,iterations,batch_size):
        Trainer.__init__(self)
        self.iterations = iterations
        self.batch_size = batch_size
        
        self.W1_data = []
        self.b1_data = []
        self.W2_data = []
        self.b2_data = []
        
        #create the the basic neural net
        self.W1 = tf.Variable(tf.truncated_normal([nn_in,nn_hdim]))
        self.b1 = tf.Variable(tf.truncated_normal([nn_hdim]))
        self.W2 = tf.Variable(tf.truncated_normal([nn_hdim,nn_out]))
        self.b2 = tf.Variable(tf.truncated_normal([nn_out]))
        self.x = tf.placeholder(tf.float32, [None,nn_in])
        self.y1 = tf.nn.softmax(tf.matmul(self.x, self.W1) + self.b1) #outputs, apply activation function to weights and biasis
        self.y2 = tf.nn.softmax(tf.matmul(self.y1, self.W2) + self.b2) #outputs, apply activation function to weights and biasis
        self.y_ = tf.placeholder(tf.float32, [None,nn_out])#true labels for error
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y2), reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(1.0).minimize(self.cross_entropy) #tf.train.AdamOptimizer(0.5).minimize(self.cross_entropy)#
        
        self.sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        self.sess.as_default()
        
    def get_clf(self):
        pass
        
    def train(self,data,labels):
        data,labels = zip(*self.clean_raw_data(data,labels)) #clean data
        new_data = self.vectorize_images(data)
        new_labels = self.hot_vec(labels) #can handle any data output due to nn structure
        train_features,test_features,train_labels,test_labels = train_test_split(new_data,new_labels,test_size=0.05,random_state=42)

        prog = []
        res = []

        sess = self.sess #create the tf session
        #sess.as_default()
        for i in range(self.iterations):
            if ((i+1) * self.batch_size) < len(train_features):#check if out of data range
                x = train_features[(i*self.batch_size):((i+1)*self.batch_size)]
                y = train_labels[(i*self.batch_size):((i+1)*self.batch_size)]
                result,acc = sess.run([self.train_step,self.cross_entropy], feed_dict={self.x: x, self.y_: y})
                prog.append(i)
                res.append(acc)
                if i % self.batch_size == 0:
                    print("Processing: ",i * self.batch_size)
                
        #store data for later use
        self.W1_data = sess.run(self.W1)
        self.b1_data = sess.run(self.b1)
        self.W2_data = sess.run(self.W2)
        self.b2_data = sess.run(self.b2)
        
        import matplotlib.pyplot as plt
        plt.plot(prog,res)
        plt.show()
        plt.savefig("errors.png")

        correct_preds = tf.equal(tf.argmax(self.y2,1), tf.argmax(self.y_,1)) #check if prediction is correct
        acc = tf.reduce_mean(tf.cast(correct_preds, tf.float32)) #get mean which is accruacy
        result_acc = sess.run(acc, feed_dict={self.x: test_features, self.y_: test_labels})
        print("sigmoid nn result: ", result_acc)
        
    def predict(self,data): #assumes data is vectorized
        sess = self.sess
        #tf.global_variables_initializer().run()
        result = sess.run(self.y2, feed_dict={self.x: data, self.W1: self.W1_data, self.b1: self.b1_data, self.W2: self.W2_data, self.b2: self.b2_data})
        #to keep consistancy must convert to number
        result = self.reverse_hot(result)
        return result