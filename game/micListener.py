from constants import *
import numpy as np
import pyaudio
import torch
from torch import nn

# PARAMETERS:
chunkLen=2048

## THE NEURAL NETWORK MODEL
class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.nnet = nn.Linear(chunkLen//2, 4)  #  4 being |{C,Eb,Gb,Bbb}|

  def forward(self, spectra):
    return self.nnet(spectra)

class MicListener():
  def __init__(self):
    self.model = NeuralNetwork()
    self.model.load_state_dict(torch.load('pitch.ptm'))
    self.model.eval()
    self.p = pyaudio.PyAudio()
    self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=44100,
      frames_per_buffer=chunkLen, input=True)

  def heard(self):
    aud = np.frombuffer(self.stream.read(chunkLen), dtype=np.int16)
    if np.mean(np.abs(aud)) < 328:  #  20dB below max
      return AU_REST
    spectrum = np.abs(np.fft.rfft(aud)[1:])
    with torch.no_grad():
      pred = torch.nn.functional.softmax(
	self.model(torch.tensor(spectrum,dtype=torch.float32)), dim=0)
    return np.argmax(pred).item()  #  = either AU_C, AU_Eb, AU_Gb or AU_Bbb
