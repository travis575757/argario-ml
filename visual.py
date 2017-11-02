import numpy as np
import sys,pygame

class Camera(object):
    
    def __init__(self, screen, position, zoom, size):
        self.screen = screen
        self.position = position #x and y position
        self.zoom = zoom #vertical vector
        self.size = size
        
    def set_zoom(self,zoom):
        self.zoom = zoom
        
    def set_pos(self,position):
        self.position = position
        
    def set_size(self,size):
        self.size = size
            
    def get_screen(self):
        return self.screen
            
    def get_pos(self):
        return self.position
    
    def get_zoom(self):
        return self.zoom
    
    def get_size(self):
        return self.size
    
    def render(self,entity): #in future versions this should use a view matrix, takes a given entity and renders based on camera parameters
        rect = entity.get_rect()
        pos = [rect.left - self.position[0], rect.top - self.position[1]] #get relative position
        box = [rect.width,rect.height] #get new matrix for dimensions
        self._render(entity,pos,box)
        
    def fixed_render(self,entity):
        box = entity.get_box() #get new matrix for dimensions
        pos = [ - box[0] / 2 , - box[1] / 2]
        self._render(entity,pos,box)
        
    def _render(self,entity,pos,box): #entity is entity to render pos is the position on screen to render RELATIVE to camera, box is the size of the size to render to
        zoomBox = np.multiply(self.zoom,box).tolist()
        zoomPos = np.multiply(self.zoom,pos).tolist()
        result_rect = pygame.Rect(zoomPos,zoomBox)
        image = pygame.transform.rotozoom(entity.get_image(),0,self.zoom) #this is very slow as it creates a new image but it works
        result_rect.left += (self.size[0] / 2)
        result_rect.top += (self.size[1] / 2)
        self.screen.blit(image,result_rect)
        
    def render_debug(self):
        pygame.draw.line(self.screen, (255,255,0), (self.size[0]/2,0), (self.size[0]/2,self.size[1]))
        pygame.draw.line(self.screen, (255,255,0), (0,self.size[1]/2), (self.size[0],self.size[1]/2))
        
    def render_grid(self,grid): #useful for visualiziing the matrix #width, height
        ppw = int(self.size[0] / grid[0])
        pph = int(self.size[1] / grid[1])
        #render grids independtly as they may be different sizes
        for i in range(grid[0]):
            pygame.draw.line(self.screen, (255,255,0), (i*ppw,0), (i*ppw,self.size[1]))
        for i in range(grid[1]):
            pygame.draw.line(self.screen, (255,255,0), (0,i*pph), (self.size[0],i*ppw))