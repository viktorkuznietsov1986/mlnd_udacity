from keras import layers, models, optimizers
from keras import backend as K

class Critic():
    """Critic (Value) Model."""
    
    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size
        
        self.build_model()
        
        
    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        
        # input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
        
        # add hidden layers for state pathway
        net_states = layers.Dense(units=32, activation='relu')(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.3)(net_states)
        net_states = layers.Dense(units=64, activation='relu')(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(0.3)(net_states)
                
        # add hidden layers for action pathway
        net_actions = layers.Dense(units=32, activation='relu')(actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.3)(net_actions)
        net_actions = layers.Dense(units=64, activation='relu')(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(0.3)(net_actions)
        
        # combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)
        
        # add final output layer to produce Q values
        Q_values = layers.Dense(units=1, name='q_values')(net)
        
        # create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)
        
        # define optimizer and compile model for training
        optimizer = optimizers.RMSprop()
        self.model.compile(optimizer=optimizer, loss='mse')
        
        # compute action gradients
        action_gradients = K.gradients(Q_values, actions)
        
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()], outputs=action_gradients)