###############################################################################
# Editor for the AccMan video.

import numpy as np
import subprocess

sf = 24  #  Scale factor for the whole video. Dimensions are 16sf x 9sf, so
         #  sf=120 will be 1080p. Smaller values are used for rapid dev.
sf = 120

"""
We're going to build an ffmpeg command using a spine-with-overlays method.
We'll have a list of tuples, each with:
  a 'vertebra'
  an offset (i.e., an amount to trim off the beginning of the vertebra)
  a duration (specifying, perhaps modif by xtn, the start time for next vert)
  an object describing the transition from the previous vert (ignored for [0])
  a list of overlay objects describing media/ to superimpose.

Assume input type according to file extension. It seems to be fine to simply
repeat inputs in place of splitting them; it might be simpler, so I think I'll
just do that.

As we process the script tuple, we will build two lists:
  the input specification list
  the video filter edge list
The audio situation is simpler, because all the audio is on the 'spine.' (There
is ONE scene with an audio overlay, but I've processed it separately as a
special case in prepro.py.) Other than a video/audio split, we'll assume a
linear filter graph.
"""

def isVideo(fname):
  return fname.split('.')[-1] in {'mp4','mov'}

def isGif(fname):
  return fname.split('.')[-1] == 'gif'

class Mixer():
  def __init__(self):
    ## DEFINE THE COMMAND ARGUMENT VARIABLES:
    self.INS = []   #  the input specification list
    self.FE = []   #  the filter edge list
    self.nif = 0   #  number (so far) of input files
    self.nvn = 0   #  number of video nodes (excluding the file inputs)

  def newNode(self):
    retv = f'vn{self.nvn}'
    self.nvn+=1
    return retv  #  name (sans []) of the newly allocated node

  def loadFile(self, vfname, duration=0):
    if isVideo(vfname):
      ispec = f'-i {vfname}'
    elif isGif(vfname):
      # ispec = f'-ignore_loop 0 -i {vfname}'
      ispec = f'-i {vfname}'
    else:
      if duration>0:
        ispec = f'-loop 1 -t {duration} -i {vfname}'
      else:
        ispec = f'-loop 1 -i {vfname}'
    self.INS.append(ispec)
    inum = self.nif
    self.nif += 1
    return inum

  def processScript(self, script):
    # Work out the start time of each 'vertebra':
    vertDurations = [dur for (vfname, trim, dur, trans, FX) in script[:-1]]
    vertOverlaps = [trans[1] for (vfname, trim, dur, trans, FX) in script[1:]]
    starts = np.cumsum([0,*(d-o for (d,o) in zip(vertDurations,vertOverlaps))])
    eschaton = sum(dur for (a,b,dur,c,d) in script) - sum(
      trans[1] for (a,b,c,trans,d) in script[1:])

    # Load the vertebral input files:
    for (vfname, trim, duration, trans, FX) in script:
      self.loadFile(vfname, duration)  #  vertebrae are numbered sequentially

    # The audio mix is relatively simple, so do that first. Note, we're making
    # some assumptions here:
    #   only vertebra have audio content,
    #   only mp4 or mov can have audio content,
    #   there's no overlap in audio.
    sonix = [(k, trim, dur, delay) for (k,((vfname,trim,dur,x,FX),delay)) in
      enumerate(zip(script,starts)) if isVideo(vfname)]
    for (k,trim,dur,delay) in sonix:
      fe = (f'[{k}:a]atrim=start={trim}:duration={dur},' +
        f'asetpts=PTS-STARTPTS,adelay={int(1000*delay)}:all=true[a{k}]')
      self.FE.append(fe)
    self.FE.append(''.join((f'[a{k}]' for (k,t,du,de) in sonix)) +
      f'amix=inputs={len(sonix)}:normalize=0[apsn]')
    self.FE.append('[apsn]speechnorm=e=9.375:r=0.00003:l=1[a]')

    # Now preprocess each vertebral input:
    for (k,(vfname,trim,dur,trans,FX)) in enumerate(script):
      fe = f'[{k}:v]scale={16*sf}:{9*sf},'
      if isVideo(vfname):
        fe += f'trim=start={trim}:duration={dur},setpts=PTS-STARTPTS,'
      fe += f'settb=AVTB,fps=30[pv{k}]'
      self.FE.append(fe)

    # Process the FX, mapping each pv node to a corresponding fv node:
    for (k,(vfname,trim,dur,trans,FX)) in enumerate(script):
      curNode = f'pv{k}'
      for fx in FX:
        # curNode = fx[0](self,curNode,starts[k]+fx[1],*fx[2:])
        curNode = fx[0](self,curNode,fx[1],*fx[2:])
      self.FE.append(f'[{curNode}]null[fv{k}]')

    # Process the transitions:
    for (k,(vfname,trim,dur,trans,FX)) in enumerate(script[1:]):
      trans[0](self, k, starts[k+1], *trans[1:])

    # Build the ffmpeg command:
    fcom = ' '.join(['ffmpeg -y', *self.INS,
      '-filter_complex', '"'+';'.join(self.FE)+'"',
      f'-map [tv{len(script)-1}] -map [a] -t {eschaton} AccMan.mp4'])

    # Output:
    subprocess.run(fcom, shell=True)
    print('Command was:\n')
    [print(f'[{k}]   {ins}') for (k,ins) in enumerate(self.INS)]
    print()
    [print(fe) for fe in self.FE]

# THE TRANSITION EFFECTS:
# These work by modifying the mixer object's 'command argument variables.'

def wiper(mixer,  #  the Mixer object
  k,              #  vertebra number, for node labelling
  startTime,      #  start time of wipe, relative to whole mix
  dur,            #  duration of the wipe itself (figleaf is visible longer)
  figLeaf):       #  the image or gif that scrolls along to hide the wipe
  # The crossfade:
  fe = f'[{'t' if k else 'f'}v{k}][fv{k+1}]xfade=transition=wiperight:'
  fe += f'duration={dur}:offset={startTime:.3f},format=yuv420p[wi{k}a]'
  mixer.FE.append(fe)

  # Importing the fig leaf:
  fl = mixer.loadFile(figLeaf, -1) # dur*2)
  fe = f'[{fl}:v]scale={sf*10.67}:{sf*10.67}'
  if isGif(figLeaf):
    fe += f',fps=30,loop=loop=-1:size={2*30*dur}:start=0'
  fe += f'[wi{k}b]'
  mixer.FE.append(fe)

  # Hiding the wipe boundary with the fig leaf:
  fe = f'[wi{k}a][wi{k}b]overlay=shortest=1:'
  fe += f'x=(t-{startTime+0.3185})*{16*sf/dur:.3f}:y={int(-0.09*9*sf)}:enable='
  fe +=f"'between(t,{startTime-0.5*dur:.3f},{startTime+1.5*dur:.3f})'[tv{k+1}]"
  mixer.FE.append(fe)

def fade(mixer, k, startTime, dur):
  fe = f'[{'t' if k else 'f'}v{k}][fv{k+1}]xfade='
  fe += f'duration={dur}:offset={startTime:.3f},format=yuv420p[tv{k+1}]'
  mixer.FE.append(fe)

# THE "FX" (i.e., NONTRANSITION EFFECTS) FUNCTIONS:
# Each 'fx' function should have arguments:
#   the mixer object
#   the video node for this vertebra prior to the effect
#   the effect's start time (relative to the vertebra)
#   any other effect-specific required arguments
# and should return a new node with the effect applied. 

def wtfi(mixer, srcNode, startTime):
  # A one-off effect: animates the WTFi logo from startTime.
  dur = 3
  win = mixer.loadFile(f'media/WTFi.png')
  fe = f'[{win}]fade=out:st={dur+startTime-0.5}:d=0.5[wtfi0]'
  mixer.FE.append(fe)
  fe = f'[wtfi0]scale=(t-{startTime})*iw*{2*sf/120/dur}:'
  fe += f'(t-{startTime})*ih*{2*sf/120/dur}:eval=frame[wtfi1]'
  mixer.FE.append(fe)
  destNode = mixer.newNode()
  fe = f'[{srcNode}][wtfi1]overlay='
  fe += f'x=({950*sf/120}+{250*sf/120/dur}*(t-{startTime})):'
  fe += f'y=({950*sf/120}-{850*sf/120/dur}*(t-{startTime})):shortest=1:'
  fe += f"enable='between(t,{startTime+0.06},{startTime+dur})'[{destNode}]"
  mixer.FE.append(fe)
  return destNode

def tcard(mixer, srcNode, startTime, suffix):
  # Shows a titleCard from startTime to the end of the vertebra.
  cin = mixer.loadFile(f'media/titleCard{suffix}.jpg')
  scaleNode = mixer.newNode()
  fe = f'[{cin}]scale={16*sf}:{9*sf},settb=AVTB,fps=30[{scaleNode}]'
  mixer.FE.append(fe)
  destNode = mixer.newNode()
  fe = f'[{srcNode}][{scaleNode}]xfade=duration=0.3:offset={startTime},'
  fe += f'format=yuv420p[{destNode}]'
  mixer.FE.append(fe)
  return destNode

def initialOverlay(mixer, srcNode, endTime, graphic):
  # Shows a graphic (image or video) from the start of the vertebra to endTime.
  grin = mixer.loadFile(f'media/{graphic}')
  scaleNode = mixer.newNode()
  fe = f'[{grin}]scale={16*sf}:{9*sf},settb=AVTB,fps=30[{scaleNode}]'
  mixer.FE.append(fe)
  destNode = mixer.newNode()
  fe = f"[{srcNode}][{scaleNode}]overlay=enable='between(t,0,{endTime})'"
  fe += f'[{destNode}]'
  mixer.FE.append(fe)
  return destNode

def midOverlay(mixer, srcNode, startTime, endTime, graphic):
  # Shows a graphic from startTime to endTime.
  # DELAY THE OVERLAYS!
  grin = mixer.loadFile(f'media/{graphic}')
  scaleNode = mixer.newNode()
  fe = f'[{grin}]scale={16*sf}:{9*sf},settb=AVTB,fps=30,'
  fe += f'setpts=PTS+{startTime}/TB[{scaleNode}]'
  mixer.FE.append(fe)
  destNode = mixer.newNode()
  fe = f'[{srcNode}][{scaleNode}]overlay=enable='
  fe += f"'between(t,{startTime},{endTime})'[{destNode}]"
  mixer.FE.append(fe)
  return destNode

# THE SCRIPT, WHICH CAN REFER TO CONTENT FILES & THE ABOVE FX & TRANSITIONS:
script = (
  ('media/IWorkedOut.mp4', 0.8, 6.4, None, [(wtfi,2)]),
  ('media/titleCard0.jpg', 0, 4, (wiper, 1.2, 'media/blueGhost.png'), []),
  ('media/HeresTheBasic.mp4', 3, 18.5, (wiper, 1.2, 'media/pm.gif'),
    [(tcard, 2, '2'), (tcard, 4.9, '5'), (tcard, 10.3, '9')]),
  ('media/titleCard10.jpg', 0, 5, (wiper, 1.2, 'media/blueGhost.png'),
    [(tcard,2,'11')]),
  ('media/PacManCodeCom.mp4', 0.4, 76.6, (wiper, 1.2, 'media/pm.gif'), []),
  ('media/titleCard12.jpg', 0, 5, (wiper, 1.2, 'media/blueGhost.png'),
    [(tcard,2,'13')]),
  ('media/DramaMix.mp4', 0.4, 30.6, (wiper, 1.2, 'media/pm.gif'),
    [(initialOverlay,6,'pitchDetectionReadings.png')]),
  ('media/NoGPU.mp4', 0.2, 27.6, (wiper, 1.2, 'media/redGhost.png'), []),
  ('media/FFT.mp4', 0.2, 31.2, (wiper, 1.2, 'media/pm.gif'),
    [(midOverlay,6,14,'ShowFFT2D.mp4'),(midOverlay,14,31.8,'ShowFFT3D.mp4'),]),
  ('media/Numbers.mp4', 0.2, 29, (fade,0.3),
    [(midOverlay,19.2,25.2,'MMIIL.mov')]),
  ('media/WhichOctave.mp4', 0.2, 55, (wiper, 1.2, 'media/redGhost.png'),
    [(midOverlay,6,55,'ShowFFTHaSerBb.mp4')]),
  ('media/MasterSwitch.mp4', 1, 19, (fade,0.3), []),
  ('media/FetA.mp4', 1, 60, (fade,0.3),
    [(midOverlay,0,60,'ShowFFTHaSerFA.mp4')]),
  ('media/Heisenberg.mp4', 0.4, 25, (fade,0.3),
    [(midOverlay,12,25,'PositionMomentum.mp4')]),
  ('media/NotSidetracked.mp4', 1.2, 17.5, (fade,0.3), []),
  ('media/GiantHassle.mp4', 0, 26, (wiper,1.2, 'media/pm.gif'), []),
  ('media/NNLooksLike.mp4', 0, 26, (fade,0.3), [(midOverlay,0,30,'NN.mp4')]),
  ('media/AirQuotes.mp4', 0.2, 24.0, (fade,0.3), []),
  ('media/Keeble.mp4', 0.7, 47, (fade,0.3), []),
  ('media/SeventeenthDimension.mp4', 0.1, 16, (fade,0.3), []),
  ('media/RoomInYourHeart.mp4', 6, 38, (fade,0.3),
    [(midOverlay,15,26,'Bicycle.mp4')]),
  ('media/titleCard14.jpg', 0, 5, (wiper, 1.2, 'media/blueGhost.png'),
    [(tcard,2,'15')]),
  ('media/CodeTour.mp4', 0, 114, (wiper, 1.2, 'media/pm.gif'), []),
  ('media/GamePlay2.mp4', 0, 27, (fade,0.3), []),
  ('media/BetterRun.mp4', 149, 51, (fade,0.3), []),
  ('media/titleCard17.jpg', 0, 3, (wiper, 1.2, 'media/blueGhost.png'), []),
  ('media/EnjoyedHearing.mp4', 0.5, 120.5, (wiper, 1.2, 'media/pm.gif'), []),
  ('media/ChooseFourChords.mp4', 0.0, 27.6, (fade,0.3),
    [(midOverlay,0,27.6,'Noodle.mp4')]),
  ('media/ThanksForWatching.mp4', 0.4, 9.4, (fade,0.3), []),
  ('media/titleCard18.jpg', 0, 5, (fade,0.3),
    [(tcard,1.2,'19')]),
)

Mixer().processScript(script)


"""
For each vertebra:
  append an input spec
  append any initial scale/format filters
  append transition &c. filters to form the video spine
    maybe it's easiest to handle the first one separately
  append filters to form the audio spine
For each fx:
  compute start time, call function

I've now got everything working except FX. I had been imagining that now, the
FX would be edited into the video, over the top of the 'spine.' However, no fx
will span a transition (otherwise, it would BE a transition), and it may be
convenient if transitions (wipes &c.) could apply to fx as well as to the
vertebrae underneath.

Therefore, I should redesign it so that the FX are applied to the vertebrae
BEFORE the transitions are applied.

Look at the 'Process the FX' section of processScript, and thus modify the
signatures of the fx functions wtfi and tcard. Each fx function should input
and output the node names. Any ad hoc nodes required by an fx function can be
created serially using newNode. Should rewrite all of this, especially the
comments, but note that we don't seem to need ad hoc audio nodes at all, and
many of even the internal video nodes are numbered according to their own
specific logic. For FX's internal nodes, could do the same, suffixing with
vertNum_fxnum, but using an altered newNum is probably simpler.

"""
