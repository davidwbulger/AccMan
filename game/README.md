This folder contains the actual game code. Almost everything here is borrowed from [pacmancode.com](pacmancode.com).
I've included it here for completeness,
but I encourage you visit that site for more details about how the game code works.

To play, simply open a command prompt in this folder and execute the command `python run.py`.
You need Python installed, of course, and you may find that you're missing some modules,
in which case it should tell you which are missing;
just install them (e.g., via `pip`) and try again.
To be honest, I found installing PyAudio a bit of an ordeal (it had some further dependency,
outside of Python) but hopefully you'll have less trouble.

Unless you have my exact accordion, microphone and acoustic environment,
you might experience lower accuracy in controlling PAC-MAN.
In that case, you probably need to replace the file `pitch.ptm`
with a PyTorch model file trained on the sound of *your* instrument.
There's code and advice about that in `../training`.
