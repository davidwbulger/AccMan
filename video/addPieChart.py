# Add the pie chart to the thanks title card

import numpy as np
from PIL import Image, ImageDraw, ImageFont

image = Image.open("media/titleCard19.jpg")
draw = ImageDraw.Draw(image)

radius = 300
centre = np.array([960, 560])
arc = 7*np.pi/8*np.array([-1,1])

theta = np.linspace(*arc, 200)
xy = np.array([np.cos(theta), np.sin(theta)]).T
xy = centre + radius * np.vstack((xy, [0,0]))
xy = [tuple(row) for row in xy]
draw.polygon(xy,fill='Yellow')

draw.line([tuple(centre-2/3*radius*np.array([1,0])), 
  tuple(centre+radius*np.array([-4/3,-1/2]))], fill="Red", width=3)
draw.line([tuple(centre+1/3*radius*np.array([1,0])), 
  tuple(centre+radius*np.array([4/3,-1/2]))], fill="Red", width=3)

font = ImageFont.truetype("../game/PressStart2P-Regular.ttf", 32)
draw.text(centre+radius*np.array([-1.35,-1/2]),
  'Time spent\nmodifying\nPAC-MAN\ncode', font=font, fill="Red", anchor='rm',
  align='center')
draw.text(centre+radius*np.array([1.35,-1/2]),
  'Time spent\nproducing\nthis video', font=font, fill="Red", anchor='lm',
  align='center')

image.save('piche.jpg')
###############################################################################
