import numpy as np
import sys,pygame
import csv
import ast
import re
import os.path

class DataManager(object): #this is where reading and writing data from game is done
    
        def __init__(self):
            self.data = []
            self.labels = []
            self.buffer_size = 200 #bufer of data size, how large data in memory gets before being saved to csv and reset, only for contious write
            self.data_name = './player_dataRG.csv'
            self.labels_name = './player_labelsRG.csv'
            pass
        
        def get_data(self):
            return self.data
        
        def get_labels(self):
            return self.labels
        
        #here data is broken down based on given resolution
        #size = screen dimensions/input regions size, resolution = respixels,world = world to get data from
        def write_data(self,size,resolution,world):
            objects = world.get_objects()
            ply = world.get_player()
            #grab relative positions for objects to player
            image = self.process_image(size,resolution,ply.get_pos(),objects)
            self.data.append(image)
            #image data now process (game state) now processing inputs
            label = ply.get_ctr_state() #must turn the control state into a integer scalar
            result = list(ply.get_ctr_state())
            self.labels.append(result) #append labels for player output
                    
        #here data is pixled and written    
        #for now simply writes to the position in which the center of the box is
        def process_image(self,size,resolution,center,objects):
            res_objects = []
            pos_objs = [objects[i].get_pos() - center for i in range(len(objects))]
            for i in range(len(objects)):
                x_bound = abs(pos_objs[i][0]) < size[0] / 2
                y_bound = abs(pos_objs[i][1]) < size[1] / 2
                if x_bound and y_bound:
                    res_objects.append(objects[i])
            #pixel per width, pixel per height
            ppw = int(size[0] / resolution[0])
            pph = int(size[1] / resolution[1])
            image = np.zeros([resolution[1],resolution[0]]) #matrix of height and width pph and ppw
            for obj in res_objects:
                pos = obj.get_pos() - center + (np.multiply(size,0.5)) #translate relative to player then to screen
                #check if green
                isGreen = True if obj.get_color() == (0,255,0) else False
                color = 1 if isGreen else -1
                #calculate new coordinates relative to new grid
                x = int(pos[0] / ppw)
                y = int(pos[1] / pph)
                image[y,x] = color
                
            return image
            
        def save_data(self):
            with open(self.data_name, 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for image in self.data:
                    writer.writerow(image)

            with open(self.labels_name, 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for label in self.labels:
                    writer.writerow([label])        
            print("entires: ", len(self.labels))
            
        def continous_write(self,size,resolution,world):
            self.write_data(size,resolution,world) #write the data to memory
            # must check that the files exist
            if (len(self.data)) > self.buffer_size and os.path.exists(self.data_name) and os.path.exists(self.labels_name):
                with open(self.data_name, 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    for image in self.data:
                        writer.writerow(image)

                with open(self.labels_name, 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    for label in self.labels:
                        writer.writerow([label])        
                print("entires: ", len(self.labels))
                del self.data[:]
                del self.labels[:]
                print("data has been cleared")
            elif (len(self.data)) > self.buffer_size: #if files dont exist to the normal save
                print("creating new files")
                self.save_data()
                    
        def read_data(self):
            data = []
            labels = []
            if os.path.exists(self.data_name) and os.path.exists(self.labels_name):
                #reading the data is a bit of a pain
                with open(self.data_name, 'r') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';', quotechar='|')
                    for row in reader:
                        img = []
                        for i in range(len(row)): # array of string  \
                            #this is cheap formating for the way the csv was made
                            r = row[i] 
                            r = re.sub('\s', "", r, count=1) #remove the first blank space 
                            r = re.sub('\s+', ',', r) #add commas in the rest
                            img.append(list(ast.literal_eval(r))) #append the row to the image
                        if img != []:
                            data.append(img) #finally add the image to data
                #labels less so since not not 2D
                with open(self.labels_name, 'r') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';', quotechar='|')
                    for row in reader:
                        lab = []
                        for i in range(len(row)): # array of string  \
                            #this is cheap formating for the way the csv was made
                            r = row[i]
                            r = re.sub('\s', "", r, count=1) #remove the first blank space 
                            lab.append(ast.literal_eval(r)) #append the row to the image
                        if lab != []:
                            #note the indexing, ast.literal_eval place the contents of r into a list
                            labels.append(lab[0]) #finally add the image to data
            #labels = np.genfromtxt('./player_labelst.csv', delimiter=';', ) old code
            return zip(data,labels)