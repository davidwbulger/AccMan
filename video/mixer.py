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
  '-loop 1 -i graphics/titleCard13.jpg',
  '-loop 1 -i graphics/titleCard14.jpg',
  '-loop 1 -i graphics/titleCard15.jpg',
  '-loop 1 -i graphics/titleCard16.jpg',
  '-loop 1 -i graphics/titleCard17.jpg',
  '-loop 1 -i graphics/titleCard18.jpg',
  '-loop 1 -i graphics/titleCard19.jpg',
  '-i captures/HeresTheBasic.mp4',
  '-filter_complex "' +
  # '[0]scale=1120:1120,split=2[bg0][bg1];' +
  '[0]scale=1280:1280[bg0];' +
  # '[1]scale=1120:1120,split=2[pm0][pm1];' +
  '[1]scale=1280:1280[pm0];' +
  '[2:v]trim=start=6.4:duration=6.6,setpts=PTS-STARTPTS[v0];' +
  '[2:a]atrim=start=6.4:duration=6.6,asetpts=PTS-STARTPTS[a0];' +
  '[v0]settb=AVTB,fps=30[v1];' +
  '[4]settb=AVTB,fps=30[v2];' +
  '[v1][v2]xfade=transition=wiperight:duration=1.2:offset=5.4,' +
    'format=yuv420p[v3];' +
  "[v3][bg0]overlay=shortest=1:x=53.33*n-9200:y=-80:enable='between(t,4,8)'" +
    '[v4]' +
  '"',
  '-map [v4] -map [a0] output.mp4']

commandString = ' '.join(ffmpegCommandParts)
# subprocess.run(commandString, shell=True)

"""
Now we're going to try to build a similar command using a spine-with-overlays
method. We'll have a list of tuples maybe, each with
  a 'vertebra'
  an offset (i.e., an amount to trim off the beginning of the vertebra)
  a duration (specifying, perhaps modif by xtn, the start time for next vert)
  an object describing the transition from the previous vert (ignored for [0])
  a list of overlay objects describing graphics to superimpose.
"""
script = [
  ('captures/IWorkedOut.mp4', 6.4, 6.6, None),
  ()
]
