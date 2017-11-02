import numpy as np
import sys,pygame

from entites import Point
from data import DataManager
from training_sklearn import Trainer,SVMTrainer,GridTrainer

class World(object):
    
    def __init__(self, objects, player, grid, size): #pass a list of objects for the inital world gen, grid = size of data, size = screen dimensions
        self.objects = objects
        self.player = player
        
        #world generation parameters
        self.max_spawn = 9 #most that can spawn at once
        self.max_objects = 9 #most objects in list
        self.spread = size[0] #how spread out the objects will spawn
        self.delay = 100 #delay for how long to wait before generations
        self.min_spawn_distance = 100 #minimum distance from other objects an object will spawn
        
        #counter incrmenets each world update
        self.counter = 0 
        self.save_interval = 100
        
        self.data_manager = DataManager() #writes and reads data and saves and reads from csv
        self.trainer = SVMTrainer() #trainer
        self.grid = grid #size of images used in data
        self.size = size
        self.world_image = [[0] * self.grid[0] ] * self.grid[1] #intialize a given array with the size of grid for a world snapshot
        self.save_data = True #for toggling saving for csv files
        
    def get_objects(self):
        return self.objects
    
    def get_player(self):
        return self.player
    
    def get_data_manager(self):
        return self.data_manager
    
    def get_image(self):
        image = self.data_manager.process_image(self.size,self.grid,self.player.get_pos(),self.objects)
        return image
    
    def toggle_saving(self): #toggle saving data to csv files
        self.save_data = not self.save_data
        
    def update(self):
        
        #make sure to create objects before calculating the player 
        camera = self.player.get_camera() #get player camera
        camera.render_grid(self.grid)
        self.counter +=1
        if self.counter % self.delay == 0: #for every interval
            self.generate_objects() #generate objects
            self.counter = 0 #reset counter
        
        self.poll_inputs() # poll for interface inputs
        
        self.player.move_step() #move the player
        self.player.fix_render() #render the player
        self.player.update()
        rect = self.player.get_rect() #get the player rect
        for obj in self.objects: #render all objects
            box = obj.get_rect()
            if box.colliderect(rect):#if collision, increment points
                self.objects.remove(obj)
                if obj.get_color() == (255,0,0):
                    self.player.increment_r_score()
                if obj.get_color() == (0,255,0):
                    self.player.increment_g_score()
            else:
                obj.render(camera) #render object relative to player camera
                obj.update()
                    
        if self.counter % self.save_interval == 0 and self.save_data: #means that data is always written when halfway between spawns, means that suddens spawning will not interfer
            self.data_manager.continous_write([480,480],self.grid,self) #write data
    
    def poll_inputs(self): #handles inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit() #quit game 
            self.player.control(event) #handle events on player
            if event.type == pygame.KEYDOWN: #if keydown
                if event.key == pygame.K_r: #save data for r
                    self.data_manager.save_data()
                    print("data has been saved to csv")
                if event.key == pygame.K_f: #read data with f and train
                    zipped = self.data_manager.read_data()
                    data,labels = zip(*zipped)
                    self.trainer.train(data,labels)
                if event.key == pygame.K_v:
                    print("rest world")
                    del self.objects[:]
                    self.player.set_pos([0,0])
                
                    #old test code ignore
                    #test to make sure classifers are getting the same results
                    #trainer = self.player.get_trainer()
                    #data = self.data_manager.get_data()
                    #data = trainer.vectorize_images(data)
                    #clf = trainer.get_classifier()
                    #clf2 = self.trainer.get_classifier()
                    #result1 = clf.predict(data)
                    #result2 = clf2.predict(data)
                    #print("first result: ", result1[0], " result 2 : ", result2[0])
                    #zipped = self.data_manager.read_data()
                    #data,labels = zip(*zipped)
                    #data = trainer.vectorize_images(data)
                    #result3 = clf.predict(data)
                    #result4 = clf.predict(data)
                    #print("true result3 : ",result3," true result 4",result4)
                    
    def generate_objects(self):
        count = [1 if x.get_color()[1] == 255 else 0 for x in self.objects]
        print(self.player.calc_momentum())
        if sum(count) == 0 or count == [] or self.player.calc_momentum() < 3:  #rest world if there are no greens left or if the player momentum is less than 1 meaning its not moving
            self.player.set_pos([0,0])
            self.player.reset_momentum()
            del self.objects[:]
        if len(self.objects) < self.max_objects: #if max size not met
            #rest player based on green count

            size = int(np.random.rand() * self.max_spawn)
            for i in range(size):
                pos = (0.5 - np.random.rand(2)) * self.spread
                pos += self.player.get_pos() # add player positions ot they spawn near the player
                #test if distance between all points is greater than the min, yes overally long line
                can_spawn_array = [np.hypot(pos[0]-self.objects[i].get_pos()[0],pos[1]-self.objects[i].get_pos()[1]) > self.min_spawn_distance for i in range(len(self.objects))]
                can_spawn = all(can_spawn_array[i] == True for i in range(len(can_spawn_array))) #check that all are true
                if can_spawn or len(self.objects) < 1: #object can spawn if it is the min distance from all others or if objects length is 0
                    w = h = 30 * np.random.rand(1) + 1 #+1 added so object is never less than 1 pixel, height and width are the same
                    size = [w,h]
                    res = np.random.rand() #determine the color of the object, red or green
                    color = [255,0,0] 
                    if res > 0.2: #uncomment to make different color
                        color = 255,0,0
                    else:
                        color = 0,255,0
                    ptr = Point(color,pos,size,[0,0]) #create a point object and add to objects
                    self.objects.append(ptr)
                         