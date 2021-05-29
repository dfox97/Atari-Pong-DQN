import collections
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gym
#*******************************************************
#              Plotting Graph
#*******************************************************
def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)
#*******************************************************
#              PREBUILT GYM WRAPPERS FROM OPEN AI
#*******************************************************
class RepeatActionAndMaxFrame(gym.Wrapper):
    
    def __init__(self, env=None, repeat=4, no_ops=0):
        """
        Calling the open ai gym wrappers too add functionality to the environment
        Gym wrapper has functions which relate to step and reset, which renders the environment at each step. while the reset resets the environment.

        Args:
        env=environment
        repeat=how many steps the agent should perform within the environment.
        
        no_ops=Number of operations . which is performed at the very start of the episode  
        

        """
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.no_ops=no_ops
     

    def step(self, action):
        """
        From gym.Wrapper class, function step takes action and steps through environment.
        """
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)#call step function from wrapper and produce outputs for states, rewards,info and done
            t_reward += reward #increment the reward to total reward by plus or minus 1
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info

    def reset(self):
        """
        From gym.Wrapper class, function reset , resets the environment when done is true.
        """
        obs = self.env.reset()
        no_ops = np.random.randint(self.no_ops)+1 if self.no_ops > 0 else 0 #find the n operations we want to make which is a random number between the number of operations 
        for _ in range(no_ops):#iterate over n ops
            _, _, done, _ = self.env.step(0) #determine from step function if done. other params are not needed.
            if done:#if output for done is true, then finished then reset and go again.
                self.env.reset()

        self.frame_buffer = np.zeros_like((2,self.shape))
        self.frame_buffer[0] = obs

        return obs

class PreprocessFrame(gym.ObservationWrapper):
    
    def __init__(self, shape, env=None):
        """
        Uses gym.ObservationWrapper class
        init will take new shape and environment as input.

        self.observation_space = gym.spaces.Box built in function from open ai gym. 

        observation function using open-cv libary
        newFrame=converts does the image processing, converts atari game to greyscale. 
        resizImage= the image to make it easier to render.
        
        we take reshaping and convert to array aswell as swapping the axes since its converted to greyscale 1 x 255 channels.
        rescaled_obs = divide by 255.0 since greyscale

        returns new observation / rescaled_obs

        
        """
        super(PreprocessFrame, self).__init__(env)#return env to superconstructor.
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,shape=self.shape, dtype=np.float32,)#dealing with real-valued quantities

    #IMAGE PROCESS PART.
    def observation(self, obs):#needs to be named observation related to gym.ObservationWrapper wrapper.
        newFrame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY) #converting to greyscale.
        resizeImage = cv2.resize(newFrame, self.shape[1:],interpolation=cv2.INTER_AREA)
        rescaled_obs = np.array(resizeImage, dtype=np.uint8).reshape(self.shape)#reshape resize image.
        rescaled_obs = rescaled_obs / 255.0 #divide by 1 x 255 channel greyscle.

        return rescaled_obs #rturn the new image observation.

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        """
        Uses gym.ObservationWrapper class using env as superinstructor
        Args:
        env=environment
        repeat=number of times to repeat
        
        observation_space=finding the observation low and high of repeat.
        stack=use deque library here tp create collection array of length of repeat.

        reset function= clears stack and resets base environment. and append the observation to our stack repeat times.
        convert stack to numpy array and return it.

        """
        super(StackFrames, self).__init__(env)#superconstructor  
        self.observation_space = gym.spaces.Box(env.observation_space.low.repeat(repeat, axis=0),env.observation_space.high.repeat(repeat, axis=0),dtype=np.float32)#taking two arrays finding real valued quantities.
        self.stack = collections.deque(maxlen=repeat) #declare stack using dque collections with the size of repeat.

    def reset(self):
        self.stack.clear()#clear stack
        observation = self.env.reset()#reset env
        for _ in range(self.stack.maxlen):#append repeat number of times.
            self.stack.append(observation)#appending stack with reset observation 
        #return as numpy array  reshape with observation space low shape
        return np.array(self.stack).reshape(self.observation_space.low.shape) #doesnt matter to use high or low observtion space
         

    def observation(self, observation):
        self.stack.append(observation)#takes observation and appends to stack
        return np.array(self.stack).reshape(self.observation_space.low.shape)#do same as in reset and return np array

def make_env(env_name, shape=(84,84,1), repeat=4,
             no_ops=0):
    #just defaulted shape to 84,84,1 common when researching to use this.
    #repeat 4 times like in papers.

    #calling above functions , passing env each time so we bascilly stack each change on the environment as we go.
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, no_ops)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)

    return env
