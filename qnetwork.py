import numpy as np
import tensorflow as tf

class QNetwork(object):
    def __init__(self, input_size, output_size, hidden_size=100, learning_rate=5e-4):
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self._build_network()

        self.loss = tf.losses.mean_squared_error(labels=self.qvalue_placeholder, predictions=self.qvalues_output)
        self.optimizer = tf.train.AdamOptimizer(learning_rate) # Adam

        self.train_op = self.optimizer.minimize(self.loss)

        self.init = tf.global_variables_initializer()

        self.sess = tf.Session()
    
        self.init_weights()

    def _build_network(self):
        
        # Build Network
        self.state_placeholder = tf.placeholder(dtype=tf.float32, shape=(None, self.input_size))
        self.action_placeholder = tf.placeholder(dtype=tf.int16, shape=(None,)) 

        # [ 4 * 100 * 2 ]
        self.hidden = tf.layers.dense(self.state_placeholder, self.hidden_size, activation=tf.nn.relu)
        self.all_qvalues_output = tf.layers.dense(self.hidden, self.output_size) # No activation

        # Gather with action indices
        self.qvalues_output = tf.gather(self.all_qvalues_output, self.action_placeholder)

        # Label placeholder
        self.qvalue_placeholder = tf.placeholder(dtype=tf.int32, shape=(None,))

    def init_weights(self):
        self.sess.run(self.init)

    def train_one_batch(self, batch_states, batch_actions, batch_target_qvalues):
        feed_dict = { self.state_placeholder: batch_states, 
                      self.action_placeholder: batch_actions, 
                      self.qvalue_placeholder: batch_target_qvalues }
        
        fetches = [self.train_op, self.loss]
        _, batch_loss, _ = self.sess.run(fetches, feed_dict=feed_dict)
        return batch_loss
    
    def predict_one_batch(self, batch_states):
        feed_dict = { self.state_placeholder: batch_states }
        fetches = [ self.all_qvalues_output ]
        batch_qvalues = self.sess.run(fetches, feed_dict=feed_dict)[0]
        return batch_qvalues
    

class Model(object):
    def __init__(self, input_size, output_size, hidden_size=100, learning_rate=1e-1):
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self._build_network()

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.output)
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate) # SGD 

        self.train_op = self.optimizer.minimize(self.loss)

        self.init = tf.global_variables_initializer()

        self.sess = tf.Session()
    
    def _build_network(self):
        
        # Build Network
        self.input = tf.placeholder(dtype=tf.float32, shape=(None, self.input_size))

        self.hidden = tf.layers.dense(self.input, self.hidden_size, activation=tf.nn.relu)
        # hidden to output
        self.output = tf.layers.dense(self.hidden, self.output_size) # No activation

        self.predict = tf.argmax(self.output, axis=-1)
        # Label placeholder
        self.label = tf.placeholder(dtype=tf.int32, shape=(None,))

    def init_weights(self):
        self.sess.run(self.init)

    def train_one_batch(self, input, labels):
        feed_dict = { self.input: input, self.label: labels }
        fetches = [ self.train_op, self.loss, self.output ]
        _, batch_loss, _ = self.sess.run(fetches, feed_dict=feed_dict)
        return batch_loss
    
    def predict_one_batch(self, input):
        feed_dict = { self.input: input }
        fetches = [ self.predict ]
        batch_predict = self.sess.run(fetches, feed_dict=feed_dict)[0]
        return batch_predict

if __name__ == "__main__":
    
    # Train MNIST with QNetwork
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    input_size = 784
    output_size = 10
    hidden_size = 100
    learning_rate = 1e-2

    model = Model(input_size, output_size, hidden_size, learning_rate=learning_rate) 

    max_steps = 100000
    eval_per_steps = 1000
    batch_size = 100

    model.init_weights()
    
    for step in range(1, max_steps+1):
        
        batch_input, batch_label = mnist.train.next_batch(batch_size)        
        batch_loss = model.train_one_batch(batch_input, batch_label)
        
        if step % eval_per_steps == 0:
            
            dataset_names = ['train', 'validation', 'test']
            datasets = [ mnist.train, mnist.validation, mnist.test ]

            for name, dataset in zip(dataset_names, datasets):

                total_predict = model.predict_one_batch(dataset.images)
                correct = (total_predict == dataset.labels).sum()
                total = dataset.labels.size

                acc = correct / total
                print("step", step, "dataset", name,"accuracy", acc)