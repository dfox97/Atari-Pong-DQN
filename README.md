## Intro ##
OpenAI Atari Pong game Group project for Postgraduate module in Machine Learning. 
Videos are included on various training steps to showcase the Ai learning over time. It took 8hours to train the Ai to win the game of pong 100% of the time. 

The Ai agent being trained is the green player while the orange is the default Ai from Atari pong.
Winner is the first to score 20.

![]https://im3.ezgif.com/tmp/ezgif-3-1729856bed6d.gif

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

#Videos of training #

### At Step 5 (pretty much always loses) ###
https://user-images.githubusercontent.com/61083107/136714791-6478ea37-ee5d-47a3-9f74-d7d4541117cb.mp4


### At Step 210 (Starting to learn to play the game) ###
https://user-images.githubusercontent.com/61083107/136714802-bcc140b8-386c-49bf-a050-0d919492bf15.mp4


### At Step 500 (Winning) ###
https://user-images.githubusercontent.com/61083107/136714676-679d4174-687b-4111-8a0d-6d63c61b284d.mp4


## BUGS ## :
Load checkpoint stopped working properly. It loads but doesnt plot the graph or play the game as intended. It plays like it has not learnt and keeps a negative score.
Saving video works but i cant get it to play on my media player. So i screen recorded the example of the agent fully trained at 500 episodes.
