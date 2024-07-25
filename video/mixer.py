###############################################################################
# Editor for the AccMan video.

import numpy as np
import subprocess

sf = 24  #  Scale factor for the whole video. Dimensions are 16sf x 9sf, so
         #  sf=120 will be 1080p. Smaller values are used for rapid dev.

ffmpegCommandParts = [
  'ffmpeg',
  '-i media/IWorkedOut.mp4',
  '-loop 1 -t 4 -i media/titleCard0.jpg',
  '-i media/HeresTheBasic.mp4',
  '-loop 1 -i media/blueGhost.png',
  '-i media/pm.gif',
  '-loop 1 -i media/WTFi.png', 
  '-loop 1 -t 4 -i media/titleCard2.jpg',
  '-filter_complex "' +
  f'[3:v]scale={sf*10.67}:{sf*10.67}[bg0]' +
  f';[4:v]scale={sf*10.67}:{sf*10.67},fps=30,loop=loop=-1:' +
    'size=4*30:start=0[pm0]' +
  f';[0:v]scale={16*sf}:{9*sf},trim=start=6.4:duration=6.6,' +
    'setpts=PTS-STARTPTS,settb=AVTB,fps=30[v0]'+
  ';[0:a]atrim=start=6.4:duration=6.6,asetpts=PTS-STARTPTS[a0]' +
  f';[1:v]scale={16*sf}:{9*sf},settb=AVTB,fps=30[v2]' +
  ';[v0][v2]xfade=transition=wiperight:duration=1.2:offset=5.4,' +
    'format=yuv420p[v3]' +
  f';[v3][bg0]overlay=shortest=1:x=(0.833*t-4.766)*16*{sf}:y=-0.074*9*{sf}:' +
    "enable='between(t,4,8)'[v4]" +
  f';[2:v]scale={16*sf}:{9*sf},trim=start=1.0:duration=24.0,' +
    'setpts=PTS-STARTPTS,settb=AVTB,fps=30[v5]' +
  ';[2:a]atrim=start=1.0:duration=24.0,asetpts=PTS-STARTPTS,' +
    'adelay=8200:all=true[a1]' +
  ';[v4][v5]xfade=transition=wiperight:duration=1.2:offset=8.2,' +
    'format=yuv420p[v7]' +
  f';[v7][pm0]overlay=shortest=1:x=(0.833*t-7.099)*16*{sf}:y=-0.074*9*{sf}:' +
    "enable='between(t,6.8,10.8)'[v8]" +
  f';[5]fade=out:st=3.5:d=0.5[wtfi0]' +
  f';[wtfi0]scale=(t*(8-t)-15)*iw*{sf/120}:(t*(8-t)-15)*ih*{sf/120}:' +
    'eval=frame[wtfi1]' +
  f';[v8][wtfi1]overlay=x=(t*750-1300-517*(t*(8-t)-15))*{sf/120}:' +
    f'y=(950-850*(t*(8-t)-15))*{sf/120}:' +
    "shortest=1:enable='between(t,3.06,4)',split=2[v9][v9a]" +
  ';[v9a]trim=start=14[v9b]' +
  f';[6:v]scale={16*sf}:{9*sf},settb=AVTB,fps=30[v10]' +
  ";[v9][v10]xfade=duration=0.3:offset=11,format=yuv420p[v11]" +
  ";[v11][v9b]xfade=duration=0.3:offset=14,format=yuv420p[v12]" +
  ";[a0][a1]amix=inputs=2:normalize=0[a3]" +
  '"',
  '-map [v12] -map [a3] output.mp4']

ffmpegCommandParts = [
  'ffmpeg -y',
  '-i media/IWorkedOut.mp4',
  '-loop 1 -i media/titleCard0.jpg',
  '-i media/HeresTheBasic.mp4',
  '-loop 1 -i media/blueGhost.png',
  '-i media/pm.gif',
  '-filter_complex "' +
  '[0:a]atrim=start=6.4:duration=6.6,asetpts=PTS-STARTPTS,adelay=0:all=true[a0]' +
  ';[2:a]atrim=start=1.0:duration=24.0,asetpts=PTS-STARTPTS,' +
    'adelay=8200:all=true[a1]' +
  ";[a0][a1]amix=inputs=2:normalize=0[a3]" +
  f';[0:v]scale={16*sf}:{9*sf},trim=start=6.4:duration=6.6,' +
    'setpts=PTS-STARTPTS,settb=AVTB,fps=30[pv0]'+
  f';[1:v]scale={16*sf}:{9*sf},settb=AVTB,fps=30[pv1]' +
  f';[2:v]scale={16*sf}:{9*sf},trim=start=1.0:duration=24.0,' +
    'setpts=PTS-STARTPTS,settb=AVTB,fps=30[pv2]' +
  ';[pv0][pv1]xfade=transition=wiperight:duration=1.2:offset=5.4,' +
    'format=yuv420p[wi0a]' +
  f';[3:v]scale={sf*10.67}:{sf*10.67}[wi0b]' +
  #f';[wi0a][wi0b]overlay=shortest=1:x=(0.833*t-4.766)*16*{sf}:y=-0.074*9*{sf}:' +
  f';[wi0a][wi0b]overlay=shortest=1:x=(t-5.7185)*320.000:y=-15:' +
    "enable='between(t,4,8)'[tv1]" +
  ';[tv1][pv2]xfade=transition=wiperight:duration=1.2:offset=8.2,' +
    'format=yuv420p[wi1a]' +
  f';[4:v]scale={sf*10.67}:{sf*10.67},fps=30,loop=loop=-1:' +
    'size=4*30:start=0[wi1b]' +
  f';[wi1a][wi1b]overlay=shortest=1:x=(0.833*t-7.099)*16*{sf}:y=-0.074*9*{sf}:' +
    "enable='between(t,6.8,10.8)'[tv1]" +
  '"',
  '-map [tv1] -map [a3] outputA.mp4']

commandString = ' '.join(ffmpegCommandParts)
if (stillWorkingOutAudioGraph := False):
  subprocess.run(commandString, shell=True)
  # print(commandString)
  quit()

"""
Now we're going to try to build a similar command using a spine-with-overlays
method. We'll have a list of tuples maybe, each with
  a 'vertebra'
  an offset (i.e., an amount to trim off the beginning of the vertebra)
  a duration (specifying, perhaps modif by xtn, the start time for next vert)
  a list of overlay objects describing media/ to superimpose
  an object describing the transition from the previous vert (ignored for [0]).

Define functions wtfi & wiper, taking remaining args & a vert time offset.

Assume input type according to file extension. It seems to be fine to simply
repeat inputs in place of splitting them; it might be simpler, so I think I'll
just do that.

As we process the script tuple, we will build three lists:
  the input specification list
  the video filter edge list
  the audio filter edge list
Other than a video/audio split, we'll assume a linear filter graph.

I'm a little unclear on how the audio graph should look. Probably I ought to
manually edit the sample ffmpeg command just a little further so it encompasses
the second webcam capture (after the 1st title card).

PLAN:
  Maybe with help from ChatGPT, edit manual ffmpeg command a little further:
    Do we really need to separate the video & audio?
    Go as far as HeresTheBasic with the titlecard overlays
    Include WTFi
    Find out whether we can specify x&y ito t rather than n
  Then continue developing the ffmpeg-command-generation code
  Also at some point I should edit the thanks card to include the pie chart
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

  ## DEFINE THE TRANSITION AND EFFECT FUNCTIONS:
  # These work by modifying the 'command argument variables' defined above.

  # Note, every transition must have new source and duration as first two args.
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
    vertDurations = [dur for (vfname, trim, dur, FX, trans) in script[:-1]]
    vertOverlaps = [trans[2] for (vfname, trim, dur, FX, trans) in script[1:]]
    starts = np.cumsum([0,*(d-o for (d,o) in zip(vertDurations,vertOverlaps))])
    eschaton = sum(dur for (a,b,dur,c,d) in script) - sum(
      trans[2] for (a,b,c,d,trans) in script[1:])

    # Load the vertebral input files:
    for (vfname, trim, duration, FX, trans) in script:
      self.loadFile(vfname, duration)  #  vertebrae are numbered sequentially

    # The audio mix is relatively simple, so do that first. Note, we're making
    # some assumptions here:
    #   only vertebra have audio content,
    #   only mp4 or mov can have audio content,
    #   there's no overlap in audio.
    sonix = [(k, trim, dur, delay) for (k,((vfname,trim,dur,FX,x),delay)) in
      enumerate(zip(script,starts)) if isVideo(vfname)]
    for (k,trim,dur,delay) in sonix:
      fe = (f'[{k}:a]atrim=start={trim}:duration={dur},' +
        f'asetpts=PTS-STARTPTS,adelay={int(1000*delay)}:all=true[a{k}]')
      self.FE.append(fe)
    self.FE.append(''.join((f'[a{k}]' for (k,t,du,de) in sonix)) +
      f'amix=inputs={len(sonix)}:normalize=0[apsn]')
    self.FE.append('[apsn]speechnorm=e=9.375:r=0.00003:l=1[a]')

    # Now preprocess each vertebral input:
    for (k,(vfname,trim,dur,FX,trans)) in enumerate(script):
      fe = f'[{k}:v]scale={16*sf}:{9*sf},'
      if isVideo(vfname):
        fe += f'trim=start={trim}:duration={dur},setpts=PTS-STARTPTS,'
      fe += f'settb=AVTB,fps=30[pv{k}]'
      self.FE.append(fe)

    # Process the FX, mapping each pv node to a corresponding fv node:
    for (k,(vfname,trim,dur,FX,trans)) in enumerate(script):
      curNode = f'pv{k}'
      for fx in FX:
        # curNode = fx[0](self,curNode,starts[k]+fx[1],*fx[2:])
        curNode = fx[0](self,curNode,fx[1],*fx[2:])
      self.FE.append(f'[{curNode}]null[fv{k}]')

    # Process the transitions:
    for (k,(vfname,trim,dur,fx,trans)) in enumerate(script[1:]):
      trans[0](self, k, starts[k+1], *trans[1:])

    # Build the ffmpeg command:
    fcom = ' '.join(['ffmpeg -y', *self.INS,
      '-filter_complex', '"'+';'.join(self.FE)+'"',
      f'-map [tv{len(script)-1}] -map [a] -t {eschaton} outputB.mp4'])

    # Output:
    if (justPrintLists := False):
      [print(f'[{k}]   {ins}') for (k,ins) in enumerate(self.INS)]
      print()
      [print(fe) for fe in self.FE]
    elif (justPrintCommand := False):
      print(fcom)
    else:
      subprocess.run(fcom, shell=True)
      print('Command was:\n')
      # print(fcom.replace('"','"\n').replace(';',';\n'))
      [print(f'[{k}]   {ins}') for (k,ins) in enumerate(self.INS)]
      print()
      [print(fe) for fe in self.FE]

# THE TRANSITION EFFECTS:
def wiper(mixer,  #  the Mixer object
  k,              #  vertebra number, for node labelling
  startTime,      #  start time of wipe, relative to whole mix
  figLeaf,        #  the image or gif that scrolls along to hide the wipe
  dur):           #  duration of the wipe itself (figleaf is visible longer)
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

# THE "FX" (i.e., NONTRANSITION EFFECTS) FUNCTIONS:
def wtfi(mixer, srcNode, startTime):
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
  cin = mixer.loadFile(f'media/titleCard{suffix}.jpg')
  scaleNode = mixer.newNode()
  fe = f'[{cin}]scale={16*sf}:{9*sf},settb=AVTB,fps=30[{scaleNode}]'
  mixer.FE.append(fe)
  destNode = mixer.newNode()
  fe = f'[{srcNode}][{scaleNode}]xfade=duration=0.3:offset={startTime},'
  fe += f'format=yuv420p[{destNode}]'
  mixer.FE.append(fe)
  return destNode

# THE SCRIPT, WHICH CAN REFER TO CONTENT FILES & THE ABOVE FX & TRANSITIONS:
script = (
  ('media/IWorkedOut.mp4', 6.4, 6.6, [(wtfi,2)], None),
  ('media/titleCard0.jpg', 0, 4, [], (wiper, 'media/blueGhost.png', 1.2)),
  ('media/HeresTheBasic.mp4', 1, 24,
    [(tcard, 3, '2'), (tcard, 8, '5'), (tcard, 15, '9')],
    (wiper, 'media/pm.gif', 1.2))
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
