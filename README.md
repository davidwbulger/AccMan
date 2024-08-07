# AccMan
Modify PAC-MAN to be controlled by an acoustic instrument (accordion in my case).

This project is explained in [my YouTube video](https://youtu.be/dHK-RqhQYiM). Briefly, I
* trained a PyTorch model to recognise which accordion note the microphone is hearing,
* modified the code from [pacmancode.com](https://pacmancode.com) to accept accordion controls rather than keyboard controls,
* made a video describing how it works.

This repo includes one top-level folder for each of the above bullet points. Thus
* ./training contains code so you can train your own model (so you can play with your bassoon or whatever you have),
* ./game contains the actual game-play code (including the trained model, which you'll probably need to replace, unless you have an accordion similar to mine),
* ./video is probably only of interest to me, but contains most of the code I wrote to illustrate and edit the YouTube video.
