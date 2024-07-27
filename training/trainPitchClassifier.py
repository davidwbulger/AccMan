# Train the pitch classifier to be used in AccMan.

import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

startTime = time.time()
rng = np.random.default_rng()

## NOTE FOR PYTORCH VERSION:
# The pytorch parts refer to the classification tutorial
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# and my own 'circle learning' demo leranCircle.py.

# PARAMETERS
sampfile = 'sample.pickle'
chl = 2048  #  chunk length in samples
subset = [17,20,23,26]  #  if not None, the list of notes to train for.

with open(sampfile, 'rb') as fid:
  samples = pickle.load(fid)
if subset is not None:
  samples = samples[subset,:]
(N,L) = samples.shape  #  number of notes, length of each sample note

batch_size = N  #  one chunk per note in each batch
batch_mult = 64  #  chunks from each note in each batch
batch_count = 2**20//batch_mult
learningRate = 30/batch_mult  #  but will be reduced according to the schedule
gamma = 1-2/batch_count  #  learning rate exponential decrease
lossFn = nn.CrossEntropyLoss()


# For training, at each epoch, we will choose one random chunk uniformly
# distributed within each note. Obviously this plan fails if any note is
# shorter than a chunk, so let's test that:
if L < chl:
  raise ValueError('The notes are too short. Either rerecord the note samples,'
    + ' or use shorter chunks.')

classes = torch.eye(N).repeat(batch_mult,1)
def trainingBatch():
  # Randomly select batch_mult chunks from the middle of each audio sample:
  starts = rng.integers(0,L-chl,size=N*batch_mult,endpoint=True)

  # Calculate their FFT amplitudes:
  spectra = np.abs(np.fft.rfft(np.array([samples[n%N,s:s+chl]
    for (n,s) in enumerate(starts)]), axis=1)[:,1:])
  return(torch.tensor(spectra,dtype=torch.float32),classes)

## THE NEURAL NETWORK MODEL
class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.nnet = nn.Linear(chl//2, N)

  def forward(self, spectra):
    return self.nnet(spectra)

def train(model, lossFn, optimiser):
  optimiser.zero_grad()
  model.train()
  (spectra,c) = (d.to(device) for d in trainingBatch())
  pred = model(spectra)
  loss = lossFn(pred,c)
  loss.backward()
  optimiser.step()
  scheduler.step()
  return loss.item()

# Get cpu, gpu or mps device for training. (Mine was just cpu, sadly.)
device = ("cuda" if torch.cuda.is_available()
  else "mps" if torch.backends.mps.is_available()
  else "cpu")
print(f"Using {device} device")

model = NeuralNetwork().to(device)
optimiser = torch.optim.SGD(model.parameters(), lr=learningRate)
scheduler = ExponentialLR(optimiser, gamma=gamma)
print(model) 

running_loss = 0.0
for t in range(batch_count):
  loss = train(model, lossFn, optimiser)
  running_loss = 0.99*running_loss + 0.01*loss
  if t % 50 == 0:
    print(f'Epoch {t}: recent average loss = {running_loss}.')

torch.save(model.state_dict(), 'pitch.ptm')
print(f'Elapsed time {int(time.time()-startTime)} seconds.')
