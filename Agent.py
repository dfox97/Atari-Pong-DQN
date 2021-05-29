import numpy as np
import torch
from DQN import DeepQNetwork #local files functions
from ReplayMemory import ReplayBuffer

class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, nActions, input_dims,
                 mem_size, batchSize=32, epsilon_min =0.01, epsilon_decay=5e-7,replace=1000, model=None, env_name=None, chkpt_dir='models/'):
        """
        AGENT CLASS
        Online and target networks using replay memory, online network gets updated with a gradient descent in this case the model uses RMSProp. The target network handles the calculations of target values.The target network only gets updated periodically with the weights of the online network.

        The replay memory is used for sampling agents history and training the network

        chooseAction=Agent needs function for epsilon greedy action selection
        copy weights of online network to target network and 
        decrement_epsilon=decreasing epsilon over time learning from its experience and storing new memories.

        save and load model functions.

        ARGS:
        lr=learning rate
        nActions=number of actions
        input_dims=input dimensions 
        batchSize=32 
        epsilon=epsilon
        epsilon_min=epsilon_min in the deep mind paper it uses 0.1
        epsilon_decay=epsilon_decay #in about every 100thousand steps epsilon will hit minimum. 
        gamma=gamma , discount factor
        model = model name dqn
        env_name = store name of environment
        chkpt_dir=checkpoint directory

        replaceTargetCnt = replace #replace value is ten times smaller than in the deep mind paper because the in the paper the model trained for days, so this was reduced to save time. So this is set to run for around 400-500 games.Replace interval at about 1000 steps should be efficent for our agent.

        learn_step_counter = 0 counter num of times we have called the learn function so we know when to update weights of target network and eval network.

        action_space = store range of nActions 

        memory = ReplayBuffer(mem_size, input_dims, nActions)

        self.q_eval = DeepQNetwork(self.lr, self.nActions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.modelName+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)

        #q next never needs to perform gradent descent or back propagation with a q next network. only the q eval.
        self.q_next = DeepQNetwork(self.lr, self.nActions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.modelName+'_q_next',
                                    chkpt_dir=self.chkpt_dir)
        """ 
        
        self.lr = lr #learning rate
        self.nActions = nActions #number of actions
        self.input_dims = input_dims #input dimensions 
        self.batchSize = batchSize #batch size 32 can also be set at 64
        
        self.epsilon = epsilon #relation to explore/explot
        self.epsilon_min = epsilon_min #lowest value for epsilon
        self.epsilon_decay = epsilon_decay#reduce the change of taking greedy action as time goes on

        self.gamma = gamma #param for calculating dqn ,discount factor 

        self.env_name = env_name #store name of environment to print out
        self.modelName = model #model name = dqn
        
        self.chkpt_dir = chkpt_dir #checkpoint directory

        self.replaceTargetCnt = replace
        self.learnStepCnt = 0 #counter for steps
        
        self.action_space = [i for i in range(nActions)]
        
        self.memory = ReplayBuffer(mem_size, input_dims, nActions)#store memory calling replay buffer from replay memory file.

        #pass q values into the deep q network to determine states.
        self.q_eval = DeepQNetwork(self.lr, self.nActions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.modelName+'_q_eval',
                                    chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(self.lr, self.nActions,
                                    input_dims=self.input_dims,
                                    name=self.env_name+'_'+self.modelName+'_q_next',
                                    chkpt_dir=self.chkpt_dir)

    def chooseAction(self, observation):
        """
        Explore/Explot actions using greedy approach
        if the observation is greater than the epsilon value then explore and perform a random action
        otherwise:
        explot and take a random choice from the action_space.
        """
        if np.random.random() > self.epsilon: #greedy action
            state = torch.tensor([observation],dtype=torch.float).to(self.q_eval.device)#observation is in a list because the cnn neural network expects an input tensor of shape batch size by input dims. We need to add an extra dimnesion so we put it in a list and converting to a tensor.
            actions = self.q_eval.forward(state)
            action = torch.argmax(actions).item()#find max actions
        else:
            action = np.random.choice(self.action_space) #take non greedy action

        return action

    def storeTransition(self, state, action, reward, next_state, done):
        """
        Store transistions in the agents memory by calling the replaymemory file.
        takes state action reward next state and done.
        """
        self.memory.storeTransition(state, action, reward, next_state, done)

    def sampleMemory(self):
        """
        CONVERTING TO PYTROCH TENSORS.
        Store the sample memory for agent and convert to pytorch tensors
        we samples agents memory buffer and store the resulting tensors sarsa and done.
        
        """
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batchSize)
        #converts to torch tensors.
        states = torch.tensor(state).to(self.q_eval.device)
        actions = torch.tensor(action).to(self.q_eval.device)
        rewards = torch.tensor(reward).to(self.q_eval.device)
        next_states = torch.tensor(new_state).to(self.q_eval.device)
        dones = torch.tensor(done).to(self.q_eval.device)
        return states, actions, rewards, next_states, dones

    def replaceNetworkTarget(self):
        """
        replace the target network .
        if it is load the next state dictionary with q eval state dictionary.

        """
        if self.learnStepCnt % self.replaceTargetCnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        """
        decrement the epsilon value over time so it decreases.
        subtract epsilon with decrement and making sure its greater than epsilon minimum.
        """
        self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.epsilon_min else self.epsilon_min

    def saveModel(self):
        """
        saving q eval and q next checkpoints
        """
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def loadModel(self):
        """
        Load q eval and q next checkpoints
        """
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        """
        Learn/training function

        Dilema : when the agent starts to play it will not have filled up the batch size of its memory as it takes the first step or so it wont have filled up the 32 memories. So instead of wanting to sample out memory 32 times.To solve this we wait until the agent has filled up its memory batch size and return if it has not. Other is to let the agent play enoguh games to fill the memory totally randomly without learning at all before entering the phase where the agent start training.The last option is time consuming so we go with the first fix of this problem, by counting the mem counter. 

        When learing:
        q_eval.optimizer.zero_grad= We zero the gradients of our optimizer.
        replaceNetworkTarget()=Replace the target network to remove old paramaters of the target network.

        states, actions, rewards, states_, dones = self.sample_memory() #we sample the memory sarsa and dones. 

        index = np.arange(self.batchSize) #array of 0 to 31
        q_predict = self.q_eval.forward(states)[indices, actions] #feed states through q eval network. giving out action values.
        
        we use index to find the value of the actions the agent actually took in those states. [Index , actions]looks at every row to see what actions the agent actually took with the correct dimensions of batch size. If this isnt used then it would be wrong because dims=>batchSize x nActions which is the incorrect shape. 32x6 shape.


        q_next = self.q_next.forward(next_states).max(dim=1)[0] #take the max along the action dimension and then take the zero element because the action function returns a named tuple so we take the zeroth element [0] for the indecies of the max actions.
        We want to find the max action values in the next states and move the agents estimates of the action towards that estimated max action value

        #now work out the target values
        q_next[dones] = 0.0 #type of mask which switches between 0 and 1 for true.
        q_target_value = rewards + self.gamma*q_next #qtarget calcs if q dones is terminal (0) then q target is just equal rewards.

        loss = self.q_eval.loss(q_target_value, q_predict).to(self.q_eval.device) #from dqn model
        loss.backward() #backpropagation
        self.q_eval.optimizer.step() #call optimizer and step
        self.learn_step_counter += 1#increment counter
        self.decrement_epsilon()#decrement epsilon
        """
        if self.memory.memCounter < self.batchSize:
            return

        #first thing we want to do is zero the gradients of our optimizer when learning 
        self.q_eval.optimizer.zero_grad()

        self.replaceNetworkTarget()#call function

        states, actions, rewards, next_states, dones = self.sampleMemory()#we sample the memory sarsa and dones
        index = np.arange(self.batchSize)

        q_predict = self.q_eval.forward(states)[index, actions] #feed states through q eval network. giving out action values.
        q_next = self.q_next.forward(next_states).max(dim=1)[0] #feed the next states throgh the network

        #now work out the target values
        q_next[dones] = 0.0 #type of mask 
        q_target_value = rewards + self.gamma*q_next #qtarget calcs if q dones is terminal (0) then q target is just equal rewards.

        loss = self.q_eval.loss(q_target_value, q_predict).to(self.q_eval.device) #from dqn model
        loss.backward() #backpropagation
        self.q_eval.optimizer.step() #call optimizer and step
        self.learnStepCnt += 1#increment counter
        self.decrement_epsilon()#decrement epsilon
