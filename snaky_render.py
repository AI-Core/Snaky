from gym.envs.classic_control import rendering
import pyglet
import math

def make_oval(width=10, height=10, res=30, filled=True):
    points = []
    for i in range(res):
        ang = 2*math.pi*i / res
        points.append((math.cos(ang)*width/2, math.sin(ang)*height/2))
    if filled:
        return rendering.FilledPolygon(points)
    else:
        return rendering.PolyLine(points, True)
        
class TextLabel(rendering.Geom):
    def __init__(self, **kwargs):
        super().__init__()
        self.label = pyglet.text.Label(**kwargs)
    def render1(self):
        self.label.draw()

class Image(rendering.Geom):
    def __init__(self, path, loc):
        super().__init__()
        image = pyglet.image.load(path)
        self.sprite = pyglet.sprite.Sprite(image)
        self.sprite.x, self.sprite.y = loc['x'], loc['y']
    def render1(self):
        self.sprite.draw()

