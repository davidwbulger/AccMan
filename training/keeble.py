# Listen for accordion notes, classify pitches, depict against keyboard image.

import matplotlib.image as mimg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pickle
import pyaudio
import torch
from torch import nn

# PARAMETERS
N = 34  #  number of sample notes; will eventually be 34
chunkLen = 2048  #  chunk length in samples
sampFreq = 44100
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1

## THE NEURAL NETWORK MODEL
class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.nnet = nn.Linear(chunkLen//2, N)

  def forward(self, spectra):
    return self.nnet(spectra)

model = NeuralNetwork()
model.load_state_dict(torch.load('../game/pitch.ptm'))
model.eval()

p = pyaudio.PyAudio()  # Create an interface to PortAudio
stream = p.open(format=sample_format, channels=channels, rate=sampFreq,
  frames_per_buffer=chunkLen, input=True)

(fig,ax) = plt.subplots(2,1,height_ratios=(2,1))
plt.subplots_adjust(hspace=0)
xdata = np.arange(43,77)
ydata = 0*xdata
barcol = ax[0].bar(xdata,ydata,edgecolor='black',
  color=['black' if np.mod(k,12)in(1,3,6,8,10)else 'white' for k in xdata])
img = mimg.imread('keyboard.jpeg')
ax[1].imshow(img)
ax[0].xaxis.set_visible(False)
ax[1].axis('off')

def init():
    ax[0].set_xlim(min(xdata)-4, max(xdata)+3.6)
    ax[0].set_ylim(0, 1)

def update(frame):
  samplesToSkip = stream.get_read_available() - chunkLen
  if samplesToSkip < 0:
    return
  if samplesToSkip > 0:
    _ = stream.read(samplesToSkip, exception_on_overflow=False)
  aud = np.frombuffer(stream.read(chunkLen, exception_on_overflow=False),
    dtype=np.int16)
  if np.max(np.abs(aud)) < 1037:  #  15dB noise floor
    pred = np.zeros(N)
  else:
    spectrum = np.abs(np.fft.rfft(aud)[1:])
    with torch.no_grad():
      pred = torch.nn.functional.softmax(
	0.0000005*model(torch.tensor(spectrum,dtype=torch.float32)), dim=0)
  for (yv,bar) in zip(pred,barcol):
    bar.set_height(yv)

ani = FuncAnimation(fig, update, frames=[0], repeat=True,
  init_func=init, interval=1)
plt.show()
