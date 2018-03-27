import tensorflow as tf


class Bi_LSTM(object):

    def __init__(self, lstm_size, batch_len, output_nodes, keep_prob, learning_rate):
        self.inputs_ = tf.placeholder(tf.float32, shape=[batch_len, None, 512], name='lstm_inputs')
        self.targets_ = tf.placeholder(tf.int32, [batch_len], name='lstm_targets')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.learning_rate = learning_rate
        self.output_nodes = output_nodes
        self.lstm_size = lstm_size
        self.batch_len = batch_len
        self.nodes = []
        self.loss = None
        self.output_fw = None
        self.output_bw = None
        self.optimizer = None
        self.y_one_hot = None
        self.predictions = None
        self.final_state_fw = None
        self.final_state_bw = None
        self.initial_state_fw = None
        self.initial_state_bw = None

    def process_node_names(self):
        print("===================================")
        for i in range(len(self.nodes)):
            node_name, node_number = self.nodes[i].split(":")
            self.nodes[i] = node_name

        print(",".join(self.nodes))

    def build_model(self):

        self.initial_state_fw, cell_fw = self.lstm_layers(self.batch_len, self.keep_prob,
                                                          self.lstm_size, number_of_layers=2)

        self.initial_state_bw, cell_bw = self.lstm_layers(self.batch_len, self.keep_prob,
                                                          self.lstm_size, number_of_layers=2)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw,
                                                          cell_bw=cell_bw,
                                                          inputs=self.inputs_,
                                                          initial_state_fw=self.initial_state_fw,
                                                          initial_state_bw=self.initial_state_bw)

        self.output_fw, self.output_bw = outputs
        self.final_state_fw, self.final_state_bw = states

        outputs = tf.concat(outputs, 2)
        self.predictions = tf.contrib.layers.fully_connected(outputs[:, -1], self.output_nodes,
                                                             activation_fn=tf.sigmoid)
        self.build_cost(self.predictions)
        self.build_optimizer()

    def build_cost(self, predictions):
        self.y_one_hot = tf.one_hot(self.targets_, self.output_nodes)
        self.y_one_hot = tf.reshape(self.y_one_hot, predictions.get_shape(), name="lstm_y_one_hot")
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=self.y_one_hot)
        self.loss = tf.reduce_mean(self.loss)
        self.nodes = [self.inputs_.name, self.targets_.name, self.predictions.name, self.y_one_hot.name]

    def build_optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def lstm_layers(self, batch_size, keep_prob, lstm_size, number_of_layers):

        def lstm_cell():
            lstm_nodes = tf.contrib.rnn.BasicLSTMCell(lstm_size)

            drop = tf.contrib.rnn.DropoutWrapper(lstm_nodes, output_keep_prob=keep_prob)
            return drop

        rnn_layers = [lstm_cell() for _ in range(number_of_layers)]
        stacked_lstm = tf.contrib.rnn.MultiRNNCell(rnn_layers)
        initial_state = stacked_lstm.zero_state(batch_size, tf.float32)

        return initial_state, stacked_lstm


if __name__ == '__main__':
  rnn = Bi_LSTM(lstm_size=128, batch_len=1, output_nodes=14, keep_prob=0.85, learning_rate=0.001)
  rnn.build_model()

