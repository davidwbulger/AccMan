###############################################################################
# Preprocess one scene for the AccMan video.

import numpy as np
import subprocess

delay2 = 17.5
trim2 = 13

# I will be damned if I can see the difference, but this didn't work:
ffmpegCommandParts = [
  'ffmpeg -y',
  '-i media/AlgoPitchDetection.mp4',
  '-i media/Dramatisation.mp4',
  '-filter_complex',
  f'\'[0:a]asetpts=PTS-STARTPTS[a0]' +
  f';[1:a]atrim=start={trim2},asetpts=PTS-STARTPTS,' +
  f'adelay={int(1000*delay2)}|{int(1000*delay2)}[a1]' +
  ';[a0][a1]amix=inputs=2:weights="0.9 0.12"[a]' +
  ';[0:v]setpts=PTS-STARTPTS,fps=fps=30[v0]'
  f';[1:v]trim=start={trim2-0.15},setpts=PTS-STARTPTS,eq=saturation=0.4,' +
  'drawtext=fontfile=../game/PressStart2P-Regular.ttf:' +
  'text=Dramatisation:fontcolor=white:fontsize=84:box=1:boxcolor=black@0.8:' +
  'boxborderw=5:x=(w-text_w)/2:y=0.84*h[v1]' +
  ';[v1]fps=fps=30[v1a]'
  f';[v0][v1a]xfade=duration=0.3:offset={delay2-0.15},format=yuv420p[v]\'',
  '-map [v] -map [a] media/DramaMix.mp4']

# This was written with ChatGPT's help:
commandString = f"""
ffmpeg -y -i media/AlgoPitchDetection.mp4 -i media/Dramatisation.mp4 \
-filter_complex \
"[0:a]asetpts=PTS-STARTPTS[a0]; \
 [1:a]atrim=start={trim2},asetpts=PTS-STARTPTS,adelay={int(1000*delay2)}|{int(1000*delay2)}[a1]; \
 [a0][a1]amix=inputs=2:weights='0.9 0.12'[a]; \
 [0:v]setpts=PTS-STARTPTS,fps=fps=30[v0]; \
 [1:v]trim=start={trim2-0.15},setpts=PTS-STARTPTS,eq=saturation=0.55,drawtext=fontfile=../game/PressStart2P-Regular.ttf:text='Dramatisation':fontcolor=white:fontsize=84:box=1:boxcolor=black@0.8:boxborderw=5:x=(w-text_w)/2:y=0.88*h[v1]; \
 [v1]fps=fps=30[v1_fixed]; \
 [v0][v1_fixed]xfade=transition=fade:duration=0.3:offset={delay2-0.15},format=yuv420p[v]" \
-map "[v]" -map "[a]" media/DramaMix.mp4
"""

# commandString = ' '.join(ffmpegCommandParts)
subprocess.run(commandString, shell=True)
print(commandString)
