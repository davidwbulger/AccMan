You're almost certain to need to make a few changes to get this to work.

Files in this folder (in likely order of use):
* `collectSamples.py`: a utility to collect a list of audio samples, one per note on your instrument, and store them to a `pickle` file. Before running it, adjust `N` (the number of notes your instrument produces) and `firstNote` (the lowest note it can make). I assume the instrument is chromatic; otherwise you'll need to make further changes.
* `trainPitchClassifier.py`: uses the audio samples to train the neural network. The likeliest thing you'll want to change is `subset`, depending on the notes you want to detect.
* `keyboard.jpeg` and `keeble.py`: Unlikely to be directly useful, but these to show how I did the animated keyboard, shown in [my YouTube video about this repo](https://youtu.be/dHK-RqhQYiM), before connecting the pitch classifier to the PAC-MAN code.
