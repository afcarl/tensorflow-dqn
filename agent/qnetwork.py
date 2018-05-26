"""
    Tensorflow implementation of a fully connected network for baseline DQN
"""
import tensorflow as tf

class QNetwork(object):
    """
        The Q Value Network architecture for the DQN agent
        Builds a Tensorflow Graph, session is maintained in agent
    """
    def __init__(self, opt, name='qnetwork'):
        # Network Architecture Hyperparameters
        self.input_size = opt['input_size']
        self.output_size = opt['output_size']
        self.hidden_size = opt['hidden_size']

        # Build Graph, and training operation
        self._build_network(name)
        self.loss = tf.losses.mean_squared_error(labels=self.qvalue_placeholder, predictions=self.qvalues_output)
        self.optimizer = tf.train.AdamOptimizer(opt['learning_rate']) # Adam

        # Define graph running ops
        self.train_op = self.optimizer.minimize(self.loss)

    def _build_network(self, name):
        """
            Build network with name_scope
        """
        with tf.variable_scope(name):
            # Placeholders
            self.state_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.input_size))
            self.action_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,)) 
            self.qvalue_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,))
            
            # [ 4 * 100 * 2 ]
            self.hidden = tf.layers.dense(self.state_placeholder, self.hidden_size, activation=tf.nn.relu)
            self.all_qvalues_output = tf.layers.dense(self.hidden, self.output_size) # (batch_size, output_size)
            
            # Gather with action indices
            mask = tf.one_hot(self.action_placeholder, depth=self.output_size)
            self.qvalues_output = tf.boolean_mask(self.all_qvalues_output, mask)

    def train_one_batch(self, sess, batch_states, batch_actions, batch_target_qvalues):
        # Create feed_dict
        feed_dict = { self.state_placeholder: batch_states, 
                      self.action_placeholder: batch_actions, 
                      self.qvalue_placeholder: batch_target_qvalues }
        # Create fetches
        fetches = [self.train_op, self.loss]
        # Run
        _, batch_loss = sess.run(fetches, feed_dict=feed_dict)
        
        return batch_loss
    
    def predict_one_batch(self, sess, batch_states):
        # Create feed_dict
        feed_dict = { self.state_placeholder: batch_states }
        # Create fetches
        fetches = [ self.all_qvalues_output ]
        # Run session
        batch_qvalues = sess.run(fetches, feed_dict=feed_dict)[0]
        return batch_qvalues
