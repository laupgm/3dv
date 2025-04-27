###ASSUMES THE WOUND IS VERTICAL IN THE IMAGE

from math import *

def length(x1,y1,x2,y2): return sqrt(((x1-x2)**2)+((y1-y2)**2))

def distance(xy1,xy2): return abs(xy1-xy2)

def stitch_orientation(x1,y1,x2,y2): return 360*atan2(y2-y1,x2-x1)/2/pi 

def wound_orientation(x1,y1,x2,y2): return (360*atan2(y2-y1,x2-x1)/2/pi)

def avg(v):
    s=0
    for i in range(len(v)):
        s+=v[i]
    return s/(len(v))
