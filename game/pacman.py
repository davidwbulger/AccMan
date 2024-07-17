import pygame
from pygame.locals import *
from vector import Vector2
from constants import *
from entity import Entity
from sprites import PacmanSprites

controlStyle = 'VIM'

class Pacman(Entity):
  def __init__(self, node, micListener):
    Entity.__init__(self, node )
    self.name = PACMAN  
    self.color = YELLOW
    self.direction = LEFT
    self.setBetweenNodes(LEFT)
    self.alive = True
    self.sprites = PacmanSprites(self)
    self.micListener = micListener
    match controlStyle:
      case 'CIRCLEOFSEMITONES':
        self.controlsMapper = {
          AU_REST:STOP, AU_C:UP, AU_Eb:RIGHT, AU_Gb:DOWN, AU_Bbb:LEFT}
      case 'CIRCLEOFFIFTHS':
        self.controlsMapper = {
          AU_REST:STOP, AU_C:UP, AU_Eb:LEFT, AU_Gb:DOWN, AU_Bbb:RIGHT}
      case 'VIM':
        self.controlsMapper = {
          AU_REST:STOP, AU_C:LEFT, AU_Eb:DOWN, AU_Gb:UP, AU_Bbb:RIGHT}

  def reset(self):
    Entity.reset(self)
    self.direction = LEFT
    self.setBetweenNodes(LEFT)
    self.alive = True
    self.image = self.sprites.getStartImage()
    self.sprites.reset()

  def die(self):
    self.alive = False
    self.direction = STOP

  def update(self, dt):	
    self.sprites.update(dt)
    self.position += self.directions[self.direction]*self.speed*dt
    direction = self.getValidKey()
    if self.overshotTarget():
      self.node = self.target
      if self.node.neighbors[PORTAL] is not None:
        self.node = self.node.neighbors[PORTAL]
      self.target = self.getNewTarget(direction)
      if self.target is not self.node:
        self.direction = direction
      else:
        self.target = self.getNewTarget(self.direction)

      if self.target is self.node:
        self.direction = STOP
      self.setPosition()
    else: 
      if self.oppositeDirection(direction):
        self.reverseDirection()

  def getValidKey(self):
    return self.controlsMapper[self.micListener.heard()]

  def eatPellets(self, pelletList):
    for pellet in pelletList:
      if self.collideCheck(pellet):
        return pellet
    return None  
  
  def collideGhost(self, ghost):
    return self.collideCheck(ghost)

  def collideCheck(self, other):
    d = self.position - other.position
    dSquared = d.magnitudeSquared()
    rSquared = (self.collideRadius + other.collideRadius)**2
    if dSquared <= rSquared:
      return True
    return False
