import tensorflow as tf
from utils import utility


class ConvAutoEncoder1(object):
    def __init__(self, summary=True):
        self.inputs_ = tf.placeholder(tf.float32, (None, 240, 240, 1), name='inputs')
        self.targets_ = tf.placeholder(tf.float32, (None, 240, 240, 1), name='targets')
        self.nodes = []
        self.summary = summary
        self.decoded = None
        self.loss = None
        self.opt = None
        self.acc = None

    def build_model(self):
        encoded = self.encoder(self.inputs_)
        logits = self.decoder(encoded)
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets_, logits=logits)
        self.loss = tf.reduce_mean(self.loss)
        self.opt = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        self.process_node_names()

    def process_node_names(self):
        print("===================================")
        for i in range(len(self.nodes)):
            node_name, node_number = self.nodes[i].split(":")
            self.nodes[i] = node_name

        print(",".join(self.nodes))

    def encoder(self, inputs_):

        # ===============================
        #             Encoder
        # ===============================
        with tf.variable_scope('encoder'):

            # ===============================
            # conv1 = (?, 240, 240, 16)
            # ===============================
            conv1 = tf.layers.conv2d(inputs_, 16, (3, 3), padding='same', activation=tf.nn.leaky_relu, name="conv1")

            # ===============================
            # inception_1 = (?, 240, 240, 64)
            # ===============================
            inception_1 = utility.apply_inception(conv1, 16, 4)

            # ===============================
            # maxpool1 = (?, 120, 120, 16)
            # ===============================
            maxpool1 = tf.layers.max_pooling2d(inception_1, (2, 2), (2, 2), padding='same', name="maxpool1")

            # ===============================
            # inception_2 = (?, 120, 120, 32)
            # ===============================
            inception_2 = utility.apply_inception(maxpool1, 16, 2)

            # ===============================
            # maxpool2 = (?, 60, 60, 8)
            # ===============================
            maxpool2 = tf.layers.max_pooling2d(inception_2, (2, 2), (2, 2), padding='same', name="maxpool2")

            # ===============================
            # conv3 = (?, 60, 60, 8)
            # ===============================
            inception_3 = utility.apply_inception(maxpool2, 8, 2)

            # ===============================
            # maxpool3 = (?, 30, 30, 8)
            # ===============================
            maxpool3 = tf.layers.max_pooling2d(inception_3, (2, 2), (2, 2), padding='same', name="maxpool3")

            # ===========================================
            # Flattening maxpool3 and encoded = (?, 512)
            # ===========================================
            encoded = tf.layers.dense(tf.contrib.layers.flatten(maxpool3), 512,
                                      activation=tf.nn.leaky_relu, name="encoded")

            self.nodes = [conv1.name, inception_1.name, maxpool1.name, inception_2.name,
                          maxpool2.name, inception_3.name, maxpool3.name, encoded.name]

            if self.summary:
                print(conv1.name, "=", conv1.shape)
                print(inception_1.name, "=", inception_1.shape)
                print(maxpool1.name, "=", maxpool1.shape)
                print(inception_2.name, "=", inception_2.shape)
                print(maxpool2.name, "=", maxpool2.shape)
                print(inception_3.name, "=", inception_3.shape)
                print(maxpool3.name, "=", maxpool3.shape)
                print(encoded.name, "=", encoded.shape)

        return encoded

    def decoder(self, encoded):

        # ======================
        #       Decoder
        # ======================
        with tf.variable_scope('decoder'):

                # ======================
                #  dense1 = (?, 7200)
                # ======================
                dense1 = tf.layers.dense(encoded, 30*30*8, activation=tf.nn.leaky_relu, name="dense1")

                # ========================
                # dense1 = (?, 30, 30, 8)
                # ========================
                reshape1 = tf.reshape(dense1, (-1, 30, 30, 8), name="c")

                # ========================
                #  conv4 = (?, 30, 30, 8)
                # ========================
                conv4 = tf.layers.conv2d(reshape1, 8, (3, 3), padding='same', activation=tf.nn.leaky_relu, name="conv4")

                # ==============================
                #  up_sample1 = (?, 60, 60, 8)
                # ==============================
                up_sample1 = tf.image.resize_nearest_neighbor(conv4, (60, 60), name="upsample1")

                # ==============================
                #  conv5 = (?, 120, 120, 8)
                # ==============================
                conv5 = tf.layers.conv2d(up_sample1, 8, (3, 3), padding='same', activation=tf.nn.leaky_relu, name="conv5")

                # ==============================
                #  upsample2 = (?, 120, 120, 8)
                # ==============================
                up_sample2 = tf.image.resize_nearest_neighbor(conv5, (120, 120), name="upsample2")

                # ==============================
                #  conv6 = (?, 240, 240, 16)
                # ==============================
                conv6 = tf.layers.conv2d(up_sample2, 8, (3, 3), padding='same', activation=tf.nn.leaky_relu, name="conv6")

                # ==============================
                #  up_sample3 = (?, 240, 240, 16)
                # ==============================
                up_sample3 = tf.image.resize_nearest_neighbor(conv6, (240, 240), name="upsample3")

                conv7 = tf.layers.conv2d(up_sample3, 16, (3, 3), padding='same', activation=tf.nn.leaky_relu, name="conv7")

                logits = tf.layers.conv2d(conv7, 1, (3, 3), padding='same', activation=None, name="logits")

                self.decoded = tf.nn.sigmoid(logits, name='decoded')

                self.nodes += [dense1.name, reshape1.name, conv4.name, up_sample1.name, conv5.name, up_sample2.name,
                               conv6.name, up_sample3.name, conv7.name, logits.name, self.decoded.name]

                if self.summary:
                    print(dense1.name, "=", dense1.shape)
                    print(reshape1.name, "=", reshape1.shape)
                    print(conv4.name, "=", conv4.shape)
                    print(up_sample1.name, "=", up_sample1.shape)
                    print(conv5.name, "=", conv5.shape)
                    print(up_sample2.name, "=", up_sample2.shape)
                    print(conv6.name, "=", conv6.shape)
                    print(up_sample3.name, "=", up_sample3.shape)
                    print(conv7.name, "=", conv7.shape)
                    print(logits.name, "=", logits.shape)
                    print(self.decoded.name, "=", self.decoded.shape)

        return logits


if __name__ == "__main__":

    cae = ConvAutoEncoder1()
    cae.build_model()
