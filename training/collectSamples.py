# This program's job is to record one audio sample, for training, for each key
# on the accordion's right-hand side.

import numpy as np
import pickle
import sounddevice as sd
import time

# Parameters
sf = 44100  #  sample frequency in hertz
T = 40  #  length of each sample in seconds
tt = 2  #  transition time for switching notes
N = 34  #  number of keys (note classes) to collect samples of
firstNote= 'G'
filename = 'sample.pickle'

tsf = int(T*sf)
rec = np.zeros((N,tsf))

# Names of the notes:
notes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']*9
while notes[0] != firstNote:
  notes = notes[1:]
notes = notes[:N]
print(notes)

print('Record samples of notes, to be used in training the pitch classifier.')
print('Start with your lowest notes, & go up in semitones.')
print('For each note, play when instructed; try to include onsets and offsets')
print(f'but no silences in the note\'s {T} second sample.')
print('Are ya ready??\n\n')

for (n,note) in enumerate(notes):
  print(f'Start playing {note}')
  time.sleep(tt)
  print(f'Keep playing {note} -- RECORDING')
  rec[n] = sd.rec(tsf,samplerate=sf,channels=1,blocking=True).flatten()
print('You can stop playing now.')

with open(filename, 'wb') as fid:
  pickle.dump(rec, fid)
print(f'The collected audio samples have been saved to {filename}.')
