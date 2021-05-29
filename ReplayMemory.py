import numpy as np

class ReplayBuffer(object):
    """
    key reason for using replay memory is to break the correlation between consecutive samples. If the network learned only from consecutive samples of experience as they occurred sequentially in the environment, the samples would be highly correlated and would therefore lead to inefficient learning.
    https://deeplizard.com/learn/video/Bcuj2fTH4_4#:~:text=A%20key%20reason%20for%20using,therefore%20lead%20to%20inefficient%20learning.

    Goal of the replay memory is to keep track of the states,actions , rewards , new states and dones the agent has seen.

    uniformly sample memory to enable equal probability of being sampled.
    Shouldnt repeat any memories 

    Implimented using numpy arrays , deques could also be used but it was easier to use np arrrays and pytorch tensors.

    Args:
    mem_size=max memory size

    input_shape=shape of observations from environment

    n_actions=number of actions 

    memCounter=position of the last stored memory

    state_memory=new state memory as np array of zeros with relation to memory size by *input shape(* means deals with different shapes/observations of environment) and dtype of float32#import to use 32 to reduce size 64 uses to much memory.

    new_state_memory=identicle to the state memory. however will store the next state.

    action_memory=array of mem_size  uses int64 works best with pytorch instead of argubly using int32.

    reward_memory=array of mem_size float32 used 

    terminal_memory= array of mem_size unbool to determine true or false.
    """
    def __init__(self, max_size, input_shape, n_actions):
        self.memCounter = 0 #counter
        self.mem_size = max_size#max mem

        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),dtype=np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def storeTransition(self, state, action, reward, next_state, done):
        """
        store memories in first position of unoccupied memory.
        #Modules operator gives us position of the first unoccupied memory and in the even that the  memcounter is greater than the memory size itll go back to the begining and start overwriting memory as told in the deep mind paper.
        """
        index = self.memCounter % self.mem_size #Modules operator gives us position of the first unoccupied memory and in the even that the  memcounter is greater than the memory size itll go back to the begining and start overwriting memory as told in the deep mind paper.
        self.state_memory[index] = state #saving index memory
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.memCounter += 1 #increment

    def sample_buffer(self, batch_size):
        """
        
        Sample the memory buffer uniformly.
        max mem=
        figure out what the position of the last stored memory is in the event that we have filled up the agents memory.
        We want to sample all the way up to the agents memory size in the event that we havent filled up the agents memory size we want to go all the way up to the memory counter 
        Giving either the min memCounter or mem size.

        batch=uniformly sample a batch memory using np random choice function with max index and max mem and shape batch_size and replace is equal false to ensure a number cant be selected again. eliminating the chance of repeating memory samples.

        Then get state action, reward,nextstate and dones out of our memory.
        """
        max_mem = min(self.memCounter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]#select mem array at index batch and store each value.
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_state = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        #after sampling we return all params.
        return states, actions, rewards, next_state, dones
