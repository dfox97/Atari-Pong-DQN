"""
Assignment 2 COMP532
Program by: Daniel Fox, Beatrice Carroll, Sophie Hook, Devon Motte

The code is inspired from the deep mind paper :https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

The majority of the code was created with the help of pytorch atari videos : 
https://www.youtube.com/watch?v=wc-FxNENg9U -DQN from scratch LunarLander by  Machine Learning with Phil
https://www.youtube.com/watch?v=pDdP0TFzsoQ&t=820s -CNN expained
https://www.youtube.com/watch?v=rFwQDDbYTm4&t=1865s - DeepMind paper explained

The github to the code we followed to help us complete this assignment is here : https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code 
https://github.com/philtabor - Machine learning with Phil's github link, has multiple deep neural network examples and really helped us understand how deep nn's work.

NOTEs:
The code used here is similar to the version from Machine Learning with Phil's github and has been modified to further help understand how dqn works and how it trains the agent. The code was rewrote slighty and commented out to help show that we understand whats going on. 
The majority of help was from the github link referenced above. We attempted to build a DQNagent from scratch but it was taking to long to debug and with the timeframe we had and with what little experience we have. So we decided to use the code above as it is well written ,easy to follow and it worked.

Main issues we had:
When creating the agent and converting between tensors and numpy arrays we had issues with the array sizes at the time, but eventually found a quick fix after watching other videos which was to use the np.concentrate to make array into a big array.
Another issue we had was when designing the agent it was giving an error when the replay memory was called and I wasnt sure how to fix it at all.
Time consumption was an overall struggle since I was still new to pytorch it was challenging but fun to learn, The issue was training the agent took so long that debugging was difficuilt, as I would run the program for a couple hours then to find an error which i wouldnt know how to fix or spot.
I tried implimenting a Double DQN model but didnt have time to finish that, there is a version of this on machine learning with phil's github. 

Further improvements:
Fixing the video recording
Some minor bugs with the load checkpoit file as if u want to replot from load checkpoint, the graph doesnt plot properly nor does the the env.rendor play as expected.It plays like there is no checkpoint.
Adding other deep learning models like dueling , double q networks.
"""
import gym
import numpy as np
from Agent import DQNAgent
from utils import plot_learning_curve, make_env
from gym import wrappers


RENDER_GAME=True #IF TRUE THEN OPEN AND WATCH GAME PLAY.
LOAD_CHECKPOINT=False # wont train network , just load the saved file.
SAVE_FILE=False # saves file only if load checkpoint is set false
PLOT=False #save a plot if true

SAVE_VIDEO=False # SAVES FILE BUT DOESNT PLAY FILE ON MEDIA PLAYER

NUM_GAMES = 500 #takes about 5-6 hours of training.
environment="PongNoFrameskip-v4"


def main():
    """
        Main function which runs whole program:

        Params for agent:

        gamma=0.99 param needed for calculating the dqn. Discount factor. vary between 0 and 1 , higher gives the future rewards more importance.Gamma is closer to zero, the agent will tend to consider only immediate rewards.

        epsilon=1  start at 1 and will decay to epsilon min
        lr=0.0001 learning rate, in deep mind paper it was set at 0.1
        input_dims=(env.observation_space.shape)#find inputs using the .shape.
        n_actions=env.action_space.n  #use env .n to find num of actions 
        mem_size=50000 #50000 used i have 16 gb of ram, if u have less change to lower value.
        epsilon_min =0.1 #lowest value epsilon will decay to
        batch_size=32 #32 or 64 is typical values after the batch is filled the agent will switch to online
        replace=1000 #replace value is ten times smaller than in the deep mind paper because the in the paper the model trained for days, so this was reduced to save time. So this is set to run for around 400-500 games.Replace interval at about 1000 steps should be efficent for the agent.
        epsilon_decay=1e-5 decrease epsilong small percentage at a time
        chkpt_dir='models/' #load a checkpoint directory folder
        model='DQNAgent' #name of model
        env_name='PongNoFrameskip-v4') #name of env
    """
    
    #create environment.
    env = make_env(environment) 
    best_score = -np.inf #starting point for first episode. No best score so just set at negative infinity it will be updated after the first game of pong.

    #INIT THE AGENT BY CALLING DQN AGENT FUNCTION. #Discussed params in DQNAgent 
    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     nActions=env.action_space.n, mem_size=50000, epsilon_min =0.1,
                     batchSize=32, replace=1000, epsilon_decay=1e-5,
                     chkpt_dir='models/', model='DQNAgent',
                     env_name='PongNoFrameskip-v4')

    #LOADING CHECKPOINT.
    if LOAD_CHECKPOINT: #load checkpoint
        agent.loadModel()
    #**************************************************************************************
    #SAVING THE PLOT TO PLOTS FOLDER AND GIVING IT A NAME RELATING TO EPISODE AND AGENT.
    #**************************************************************************************
    fileName = agent.modelName + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' + str(NUM_GAMES) + 'games'
    plotFigure = 'plots/' + fileName + '.png'

    #**************************************************************************************
    #TRIED MAKING MP4 FILE BUT DOESNT WORK. SAVED GAME BUT CANNOT PLAY FILE.
    #**************************************************************************************

    # -----:>>>>>>>> Record video of agent playing not working
    if SAVE_VIDEO:
        env = wrappers.Monitor(env, "videos/",video_callable=lambda episode_id: True, force=True)
    #**************************************************************************************
    

    #**************************************************************************************
    #                    PLAYING EACH GAME N NUMBER OF TIMES./TRAINING AGENT
    #**************************************************************************************
    
    numSteps = 0
    scores=[]
    epsilonHistory=[] 
    steps_array = []

    for i in range(NUM_GAMES):#loop through games
        done = False#done should be set at false from start of each game
        observation = env.reset() #call to reset environment with each game played
        score = 0 #score should be set at 0 
        
        while not done:
            action = agent.chooseAction(observation) #call choose action and pass in obs
            new_observation, reward, done, info = env.step(action) #call wrapper function to step and get output params back. Take input action
            score += reward #add reward to total score for the game played.

            if RENDER_GAME: #THIS OPENS THE GAME. SO YOU CAN SEE GAME PLAY
                env.render()
            
            if not LOAD_CHECKPOINT: #AGENT Learns if loading checkpoint is set at false checkpoint
                agent.storeTransition(observation, action,reward, new_observation, done) #store each transaction
                agent.learn() #caall learn function
            #at end of step set new state to old.
            observation = new_observation #update the observation function
            numSteps += 1 #increment the number of steps taken.

        scores.append(score) #append the total score from the game to the total scores array
        avg_score = np.mean(scores[-100:])#work out the average score over the previous 100 games.

        steps_array.append(numSteps) #append steps taken


        print("Episode: ", i,"Score: ", score," Average Score: %.1f" % avg_score, "Best Score: %.2f" % best_score,"Epsilon Value: %.2f" % agent.epsilon, "Steps Taken:", numSteps)

        if avg_score > best_score: #if out avg score greater than best score then save a checkpoint.
            if not LOAD_CHECKPOINT:#check is load checkpoint selected.
                if SAVE_FILE:#save file if set true
                    agent.saveModel()
            best_score = avg_score # training or testing we set best score to the avg score.

        epsilonHistory.append(agent.epsilon)#agents epsilon appens to history list.
        
    if PLOT:
        plot_learning_curve(steps_array, scores, epsilonHistory, plotFigure) #Plot the graph and save file if set true.

if __name__ == '__main__':
    main()