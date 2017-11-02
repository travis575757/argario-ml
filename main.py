import numpy as np
import sys, pygame

size = 480,480
black = 0,0,0
red = 255,0,0
blue = 0,0,255
grid_size = [24,24]

screen = pygame.display.set_mode(size)

from entites import Entity
from entites import Player
from entites import Point
from entites import Machine
from entites import CPU,CPU2
from training_sklearn import Trainer,SVMTrainer,GridTrainer,ForestTrainer,SigmoidNN,FFNN
from visual import Camera
from world import World
  
entity_rect = pygame.Rect(0,0,40,40)
entity_image = pygame.Surface((entity_rect.width,entity_rect.height)) #py surface object with texture
entity_image.fill(red)
mchn_image = pygame.Surface((40,40))
mchn_image.fill(blue)

box_image = pygame.Surface((20,20))
box_image.fill(blue)

objects = []
take_inputs = False
tr = 0 #trainer index
pl = 1 #playre index

trainers = [SVMTrainer(),GridTrainer(),ForestTrainer(16,16),SigmoidNN(grid_size[0] * grid_size[1],9,17000,2),FFNN(grid_size[0] * grid_size[1],9,72,17000,2)]
trainer = trainers[tr] #default svm

if take_inputs:
    tr = int(input("# for trainer out of "))
    trainer = trainers[tr]

camera = Camera(screen,[0,0],1,size)

players = [Player(entity_image,[0,0],[40,40],[0.3,0.3],camera),Machine(mchn_image,[0,0],[40,40],[0.3,0.3],camera,objects,size,grid_size,trainer),CPU2(mchn_image,[0,0],[40,40],[0.2,0.2],camera,objects,size)]
ply = players[pl] #default CPU

if take_inputs:
    pl = int(input("# for player out of "))
    ply = players[pl]

world = World(objects,ply,grid_size,size) 
world.generate_objects()

if pl == 1: #if machine, train, the machine
    mgr = world.get_data_manager()
    zipped = mgr.read_data()
    data,labels = zip(*zipped)
    ply.train(data,labels)
if pl != 2 and pl != 0: #if not an CPU, then disable saving data
    world.toggle_saving()
    
def main():
    pygame.init()
    while 1:
                            
        screen.fill(black) #fill with black, should eventually add this stuff to a screen object
        
        world.update() #update simulation

        pygame.display.flip() #flip frame buffer

if __name__ == '__main__':
    main()

