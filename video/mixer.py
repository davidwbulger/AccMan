###############################################################################
# Editor for the AccMan video.

import subprocess

ffmpegCommandParts = [
  'ffmpeg',
  '-loop 1 -i graphics/blueGhost.png',
  '-ignore_loop 0 -i graphics/pm.gif',
  '-i captures/IWorkedOut.mp4',
  '-loop 1 -i graphics/WTFi.png', 
  '-loop 1 -t 4 -i graphics/titleCard0.jpg',
  '-loop 1 -i graphics/titleCard2.jpg',
  '-loop 1 -i graphics/titleCard5.jpg',
  '-loop 1 -i graphics/titleCard9.jpg',
  '-loop 1 -i graphics/titleCard10.jpg',
  '-loop 1 -i graphics/titleCard11.jpg',
  '-loop 1 -i graphics/titleCard12.jpg',
  '-i captures/HeresTheBasic.mp4',
  '-filter_complex "' +
  '[0]scale=1280:1280[bg0];' +
  '[1]scale=1280:1280[pm0];' +
  '[2:v]trim=start=6.4:duration=6.6,setpts=PTS-STARTPTS[v0];' +
  '[2:a]atrim=start=6.4:duration=6.6,asetpts=PTS-STARTPTS[a0];' +
  '[v0]settb=AVTB,fps=30[v1];' +
  '[4]settb=AVTB,fps=30[v2];' +
  '[v1][v2]xfade=transition=wiperight:duration=1.2:offset=5.4,' +
    'format=yuv420p[v3];' +
  "[v3][bg0]overlay=shortest=1:x=53.33*n-9200:y=-80:enable='between(t,4,8)'" +
    '[v4];' +
  '[v4][18]xfade=transition=wiperight:duration=1.2:offset=8.2,' +
    'format=yuv420p[v5];' +
  '[v5][pm0]overlay=shortest=1:x=53.33*n-13680:y=-80:' +
    "enable='between(t,6.8,10.8)'[v6];" +
  '"',
  '-map [v4] -map [a0] output.mp4']

commandString = ' '.join(ffmpegCommandParts)
if (stillWorkingOutAudioGraph := True):
  subprocess.run(commandString, shell=True)
  quit()

"""
Now we're going to try to build a similar command using a spine-with-overlays
method. We'll have a list of tuples maybe, each with
  a 'vertebra'
  an offset (i.e., an amount to trim off the beginning of the vertebra)
  a duration (specifying, perhaps modif by xtn, the start time for next vert)
  a list of overlay objects describing graphics to superimpose
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
## DEFINE THE COMMAND ARGUMENT VARIABLES:
fps = 30
cfn = 0  #  current frame number
IS = []   #  the input specification list
VFE = []  #  the video filter edge list
AFE = []  #  the audio filter edge list
cvn = ''  #  current video node
can = ''  #  current audio node
nif = 0  #  number (so far) of input files

## DEFINE THE TRANSITION AND EFFECT FUNCTIONS:

script = (
  ('captures/IWorkedOut.mp4', 6.4, 6.6, [(wtfi, 4)], None),
  ('graphics/titleCard0.jpg', 0, 4, [], (wiper, 'blueGhost.png')),
  ('captures/HeresTheBasic.mp4', 3, 14,
    [(tcard, 2, 3, 2), (tcard, 5, 5, 4), (tcard, 9, 9, 4)], (wiper, 'pm.gif'))
)

for (vfname, offset, duration, FX, trans) in script:
  (ispec, vinum) = loadFile(vfname, duration)
  IS.append(ispec)

def loadFile(vfname, duration):
  ext = vfname.split('.')[-1]
  if ext=='mp4' or ext=='mov':
    ispec = f'-i {vfname}'
  elif ext=='gif':
    ispec = f'-ignore_loop 0 -i {vfname}'
  else:
    ispec = f'-loop 1 -t {duration} -i {vfname}'
  inum = nif
  nif += 1
  return(ispec, inum)
