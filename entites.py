import numpy as np
import sys,pygame
from visual import Camera
from training_sklearn import Trainer
from data import DataManager
from sklearn.svm import SVC
from collections import deque

class Entity(object):
    
    def __init__(self, image, position, size, step):
        #position = position, step = step per movement
        self.dir_restrictions = [[0, 0], [1, 1], [-1, -1], [1, -1], [-1, 1], [1, 0], [-1, 0], [0, 1], [0, -1]]
        self.image = image
        self.position = position
        self.size = size
        self.step = step
        
        self.momentum_threshold = 1
        self.momentum_vals = deque([[i * self.momentum_threshold ,i * self.momentum_threshold ] for i in range(10)],maxlen=10) #note, not actually momentum, just a series of position values to help express how the object has moved
    def get_pos(self):
        return self.position
    
    def get_box(self):
        return self.size
    
    def get_rect(self):
        left = self.position[0] - self.size[0] / 2
        top = self.position[1] - self.size[1] / 2
        return pygame.Rect(left,top,self.size[0],self.size[1])
    
    def get_image(self):
        return self.image
    
    def set_pos(self,position):
        self.position = position
    
    def move(self,direction): # [1/0/-1,1/0/-1]dir * [dx,dy]step = [x',y']disp
        if all([abs(direction[i]) for i in range(len(direction))]) > 0 : #normalize direction vector
            mag = np.linalg.norm(direction)
            direction = [float(i)/mag for i in direction]
        displacement = (direction[0] * self.step[0],direction[1] * self.step[1]) #get displacement
        self.position[0] += displacement[0]
        self.position[1] += displacement[1]
        
        self.momentum_vals.append(list(self.position)) #add position to momentum
        
    def find_direction(self,dir):
        result = [0,0]
        if sum(dir) != 0:
            mag = np.linalg.norm(dir)
            dir = [float(i)/mag for i in dir] #normalize
            result = [round(i) for i in dir] #round values
        return result
    
    def render(self,screen):
        screen.blit(self.image, self.get_rect())
    
    def render(self,camera):
        camera.render(self)
        
    def update(self):
        pass #placeholder
    
    def get_momentum(self):
        return self.momentum
    
    def reset_momentum(self):
        self.momentum_vals = deque([[i * self.momentum_threshold ,i * self.momentum_threshold ] for i in range(10)],maxlen=10)
    
    def calc_momentum(self):
        base = self.momentum_vals[0]
        x = 0
        y = 0
        for xr,yr in self.momentum_vals:
            x += abs(xr - base[0])
            y += abs(yr - base[1])
        return (x + y)
    
    def __str__(self):
        return "Object is at %s speed %s" % (self.position,self.step)
        
class Player(Entity):    

    def __init__(self, image, position, size, step, camera):
        Entity.__init__(self, image,position,size,step)
        self.ctr_state = [0,0]
        self.camera = camera
        self.g_score = 0
        self.r_score = 0

    def move_step(self):
        self.move(self.ctr_state)
        
    def get_camera(self):
        self.camera.set_pos(self.position)
        return self.camera
    
    def get_ctr_state(self):
        return self.ctr_state
    
    def fix_render(self):
        self.camera.fixed_render(self)
        
    def increment_g_score(self):
        self.g_score += 1
        
    def increment_r_score(self):
        self.r_score += 1
        
    def get_g_score(self):
        return self.g_score
    
    def get_r_score(self):
        return self.r_score
    
    def render_scores(self):
        # initialize font; must be called after 'pygame.init()' to avoid 'Font not Initialized' error
        myfont = pygame.font.SysFont("monospace", 15)

        # render text
        screen = self.camera.get_screen()
        text = 'score is r = {} g = {}'.format(self.r_score, self.g_score)
        label = myfont.render(text, 1, (255,255,255))
        screen.blit(label, (10, 10))  
        
    def control(self,event):
        if event.type == pygame.KEYDOWN: #if key is pressed down
            if event.key == pygame.K_w:
                self.ctr_state[1] = -1
            if event.key == pygame.K_a:
                self.ctr_state[0] = -1
            if event.key == pygame.K_s:
                self.ctr_state[1] = 1
            if event.key == pygame.K_d:
                self.ctr_state[0] = 1
        if event.type == pygame.KEYUP: #if key is lifted up
            if event.key == pygame.K_w:
                self.ctr_state[1] = 0
            if event.key == pygame.K_a:
                self.ctr_state[0] = 0
            if event.key == pygame.K_s:
                self.ctr_state[1] = 0
            if event.key == pygame.K_d:
                self.ctr_state[0] = 0

class Point(Entity):
    
    def __init__(self, color, position, size, step):
        image = pygame.Surface((size[0],size[1]))
        image.fill(color)
        self.color = color
        Entity.__init__(self,image,position,size,step)
        
    def get_color(self):
        return self.color
    
class Machine(Player): #object inherits player but controlls itself after being tought a data set
           
    def __init__(self, image, position, size, step, camera, ptr_objects, img_size, grid, trainer): #machine needs a world in which to input information from
        Player.__init__(self, image,position,size,step, camera)
        self.trainer = trainer #trainer for training the AI with a data set
        self.is_trained = False
        
        self.data_manager = DataManager()
        self.ptr_objects = ptr_objects #c habits, i feel better calling this a point
        self.img_size = img_size
        self.grid = grid
        
    def get_trainer(self):
        return self.trainer
        
    def train(self,data,labels):#stuff is hard coded, shouldbe changed later
        c = [1, 1e2, 1e3, 1e4, 1e5]
        gamma = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1, 1e2]
        self.trainer.train(data,labels)
        self.is_trained = True
        
    def change_state(self,image): #lets the Machine take a matrix input and 
        if self.is_trained: #if trained
            v_image = self.trainer.vectorize_images([image])[0]#transforms into array then references since its vectorize image takes a list of images
            out = self.trainer.predict([v_image])[0] #get the predicted direction, only predicting one direction so one image is passed
            if (all([self.ctr_state[i] == -out[i] for i in range(len(out))])): #this should help get the object unstuck
                self.set_pos(self.get_pos() + np.random.rand(2))
            self.ctr_state = out
            
    def update(self):
        image = self.data_manager.process_image(self.img_size,self.grid,self.get_pos(),self.ptr_objects)
        self.change_state(image)
        
class CPU(Player): #dumb AI that inherits machine, update uses a hard coded algorthim to compute actions

    def __init__(self, image, position, size, step, camera, ptr_objects, img_size):
        Player.__init__(self, image,position,size,step, camera)
        
        self.ptr_objects = ptr_objects #c habits, i feel better calling this a point
        
        self.vision_limit_enabled = True
        self.vision_limit = img_size
        self.update_mode = 1
    
    def update(self): #currently moves to the red object of least distance
        if len(self.ptr_objects) > 0:
            new_dir = self.ptr_objects[0].get_pos() - self.position
            min_dist = np.linalg.norm(new_dir) #get the dist of the first object
            for obj in self.ptr_objects:
                pos = obj.get_pos() - self.position #get relative positionsf
                dist = np.linalg.norm(pos) #get distance
                if dist < min_dist:
                    xr = pos[0]
                    yr = pos[1]
                    is_limited = self.vision_limit_enabled
                    is_bound = (abs(xr) < self.vision_limit[0] / 2) and (abs(yr) < self.vision_limit[1] / 2)
                    if (is_limited and is_bound) or not is_limited: #vision limit makes it so that it so dumb AI cant see stuff smart one can
                        min_dist = dist
                        new_dir = pos #set the direction to pos but dont calculate yet
                
            if self.update_mode == 0:
                self._ud1(new_dir)
            elif self.update_mode == 1:
                self._ud2(new_dir)

    def _ud1(self,new_dir): #updates using a directional method, will draw a unit vector the the direction and snap to nearest direction

        result = self.find_direction(new_dir) #this will normailize the position to get direction vector + compute ctr value
        self.ctr_state = result
    
    def _ud2(self,new_dir):#this method makes x and y directions independent, this should be easier for learning
        mag = np.linalg.norm(new_dir)
        _dir = [float(i)/mag for i in new_dir] #normalize the vector
        x = _dir[0]/abs(_dir[0]) if abs(new_dir[0]) > self.step[0] else 0 #will return 1, 0 or -1
        y = _dir[1]/abs(_dir[1]) if abs(new_dir[1]) > self.step[1] else 0
        self.ctr_state = [x,y]
        
class CPU2(CPU): #dumb AI that moves towards green objects and away from red

    def __init__(self, image, position, size, step, camera, ptr_objects, img_size):
        CPU.__init__(self, image, position, size, step, camera, ptr_objects, img_size)
    
    def update(self): #moves towards green and away from red, this is a result of two vectors that area sum of inverse distances among objects
        if len(self.ptr_objects) > 0:
            objects = [] #create new list of objects based on what the CPU can see
            
            vec_gr = [0,0]
            
            #for x in self.ptr_objects:
                #print(x)
            
            for obj in self.ptr_objects: #pick an intial position to base off of by chosing the first green
                pos = obj.get_pos() - self.position #get relative positions
                dist = np.linalg.norm(pos) #get distance
                is_limited = self.vision_limit_enabled
                is_bound = (abs(pos[0]) < self.vision_limit[0] / 2) and (abs(pos[1]) < self.vision_limit[1] / 2)
                if ((is_limited and is_bound) or not is_limited) and obj.get_color()[1] == 255: #vision limit makes it so that it so dumb AI cant see stuff smart one can
                    vec_gr = obj.get_pos() - self.position
                    break #im sorry

            min_dist = np.linalg.norm(vec_gr) #get the dist of the first object
            for obj in self.ptr_objects:
                pos = obj.get_pos() - self.position #get relative positionsf
                dist = np.linalg.norm(pos) #get distance
                is_limited = self.vision_limit_enabled
                is_bound = (abs(pos[0]) < self.vision_limit[0] / 2) and (abs(pos[1]) < self.vision_limit[1] / 2)
                if (is_limited and is_bound) or not is_limited: #vision limit makes it so that it so dumb AI cant see stuff smart one can
                    objects.append(obj)
                    if dist < min_dist and obj.get_color()[1] == 255:
                        min_dist = dist
                        vec_gr = pos
            
            rweight = 100 #does not conflict with g weight as it i used before normalization

            sum_inv_red = [[0],[0]]
            #sum_inv_gr = [[0],[0]]
            for obj in objects:
                pos = obj.get_pos() - self.position
                mag = np.linalg.norm(pos)
                u = [rweight * x / mag for x in pos]
                vec = [(1/(mag*mag))*x for x in u]
                if obj.get_color()[0] == 255: #if red
                    vec = [-x for x in vec] #reverse direction
                    sum_inv_red[0].append(vec[0])
                    sum_inv_red[1].append(vec[1])
                elif obj.get_color()[1] == 255: #if green
                    #sum_inv_gr[0].append(vec[0])
                    #sum_inv_gr[1].append(vec[1])
                    pass
                    
            vec_red = [sum(sum_inv_red[0]),sum(sum_inv_red[1])] #add up values across x and y, gives direction that moves away from red fastest
            #vec_gr = [sum(sum_inv_gr[0]),sum(sum_inv_gr[1])] #add up values across x and y gives direction toward green fastest

            gweight = 5 #bias for how much green is chosen over red
    
            #normalize both so they are weighted equally
            mag1 = np.linalg.norm(vec_red)
            mag2 = np.linalg.norm(vec_gr)
            vec_red = [x/mag1 for x in vec_red] if mag1 > 0.0001 else [0,0]
            vec_gr = [gweight * x/mag2 for x in vec_gr] if mag2 > 0.0001 else [0,0]
            new_dir = [ x + y for x,y in zip(vec_red,vec_gr)]
            #print(new_dir)
            mag = np.linalg.norm(new_dir)
            new_dir = [x/mag for x in new_dir]

            #print(vec_red," r : g ", vec_gr," dir ",new_dir)
            
            if self.update_mode == 0:
                self._ud1(new_dir)
            elif self.update_mode == 1:
                self._ud2(new_dir)
            