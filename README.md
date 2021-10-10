## Intro ##
OpenAI Atari Pong game Group project for Postgraduate module in Machine Learning. 
Videos are included on various training steps to showcase the Ai learning over time. It took 8hours to train the Ai to win the game of pong 100% of the time. 




https://user-images.githubusercontent.com/61083107/136714676-679d4174-687b-4111-8a0d-6d63c61b284d.mp4



Pc Specs:
1060 GTX
i7 7700k 

## HOW TO RUN ##

Its advised to use the zip file provided since all the folders are created inside there.
Make sure requirments are fully installed, and make sure the correct folders are created.

Once complete just run the main.py file and the program should run.

Change global params depending on what you want to do:

RENDER_GAME=True #IF TRUE THEN OPEN AND WATCH GAME PLAY.
LOAD_CHECKPOINT=False # wont train network , just load the saved file.
SAVE_FILE=True # saves file only if load checkpoint is set false
PLOT=True #save a plot if true

SAVE_VIDEO=True # SAVES FILE BUT DOESNT PLAY FILE ON MEDIA PLAYER

NUM_GAMES = 501 #takes about 5-6 hours of training on 1660 GTX GPU.
environment="PongNoFrameskip-v4"



BUGS:
Load checkpoint stopped working properly. It loads but doesnt plot the graph or play the game as intended. It plays like it has not learnt and keeps a negative score.
Saving video works but i cant get it to play on my media player. So i screen recorded the example of the agent fully trained at 500 episodes.
