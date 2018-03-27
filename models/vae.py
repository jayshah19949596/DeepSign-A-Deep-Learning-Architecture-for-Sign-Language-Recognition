import tensorflow as tf
from utils import utility


class ConVAE(object):
    latent_loss = None
    generation_loss = None
    z_stddev = None
    decoded = None
    z_mean = None
    loss = None
    opt = None
    acc = None

    def __init__(self, summary=True):
        self.inputs_ = tf.placeholder(tf.float32, (None, 240, 240, 1), name='vae_inputs')
        self.targets_ = tf.placeholder(tf.float32, (None, 240, 240, 1), name='vae_targets')
        self.noise_var = tf.placeholder(tf.float32, name='noise_var')
        self.summary = summary
        self.nodes = []
        self.n_z = 512

    def process_node_names(self):
        print("===================================")
        for i in range(len(self.nodes)):
            # print(self.nodes[i])
            # print(self.nodes[i].split(":"))
            node_name, node_number = self.nodes[i].split(":")
            self.nodes[i] = node_name

        print(",".join(self.nodes))

    def build_model(self):

        self.encoder(self.inputs_)

        samples = tf.random_normal([512], 0, 1, dtype=tf.float32)
        sampled_z = self.z_mean + (self.z_stddev * samples)

        self.decoder(sampled_z)
        self.build_loss()

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 5)
        train_op = tf.train.AdamOptimizer(0.001)
        self.opt = train_op.apply_gradients(zip(grads, tvars))

    def build_loss(self):
        generated_flat = tf.contrib.layers.flatten(self.decoded)
        images_ = tf.contrib.layers.flatten(self.inputs_)

        self.generation_loss = -tf.reduce_sum(images_ * tf.log(1e-8 + generated_flat) +
                                              (1 - images_) * tf.log(1e-8 + 1 - generated_flat), 1)

        self.latent_loss = (-0.5*tf.reduce_mean
                            ((tf.reduce_mean(1 + tf.clip_by_value(self.z_stddev, -5.0, 5.0)
                             - tf.square(tf.clip_by_value(self.z_mean, -5.0, 5.0))
                              - tf.exp(tf.clip_by_value(self.z_stddev, -5.0, 5.0)), 1))))

        self.loss = self.generation_loss + self.latent_loss
        print("self.loss.shape", self.loss.shape)

    def encoder(self, inputs_):

        # ===============================
        #             Encoder
        # ===============================
        with tf.variable_scope('encoder'):

            # ===============================
            # conv1 = (?, 240, 240, 16)
            # ===============================
            conv1 = tf.layers.conv2d(inputs_, 16, (3, 3), padding='same', activation=tf.nn.leaky_relu, name="conv1")
            inception_1 = utility.apply_inception(conv1, 16, 16)

            # ===============================
            # maxpool1 = (?, 120, 120, 16)
            # ===============================
            maxpool1 = tf.layers.max_pooling2d(inception_1, (2, 2), (2, 2), padding='same', name="maxpool1")

            # ===============================
            # conv2 = (?, 120, 120, 8)
            # ===============================
            conv2 = tf.layers.conv2d(maxpool1, 8, (3, 3), padding='same', activation=tf.nn.leaky_relu, name="conv2")
            inception_2 = utility.apply_inception(conv2, 8, 8)

            # ===============================
            # maxpool2 = (?, 60, 60, 8)
            # ===============================
            maxpool2 = tf.layers.max_pooling2d(inception_2, (2, 2), (2, 2), padding='same', name="maxpool2")

            # ===============================
            # conv3 = (?, 60, 60, 8)
            # ===============================
            conv3 = tf.layers.conv2d(maxpool2, 8, (3, 3), padding='same', activation=tf.nn.leaky_relu, name="conv3")

            # ===============================
            # maxpool3 = (?, 30, 30, 8)
            # ===============================
            maxpool3 = tf.layers.max_pooling2d(conv3, (2, 2), (2, 2), padding='same', name="maxpool3")

            # ===========================================
            # Flattening maxpool3 and encoded = (?, 1024)
            # ===========================================
            dense_1 = tf.layers.dense(tf.contrib.layers.flatten(maxpool3), 1024, activation=tf.nn.leaky_relu, name="dense_1")

            self.z_mean = tf.layers.dense(dense_1, self.n_z, activation=tf.nn.relu, name="z_mean")
            self.z_stddev = tf.layers.dense(dense_1, self.n_z, activation=tf.nn.relu, name="z_stddev")

            self.nodes = [conv1.name, inception_1.name, maxpool1.name, conv2.name, inception_2.name,
                          maxpool2.name, conv3.name, maxpool3.name, self.z_mean.name, self.z_stddev.name]

            if self.summary:
                print(conv1.name, "=", conv1.shape)
                print(inception_2.name, "=", inception_1.shape)
                print(maxpool1.name, "=", maxpool1.shape)
                print(conv2.name, "=", conv2.shape)
                print(inception_2.name, "=", inception_2.shape)
                print(maxpool2.name, "=", maxpool2.shape)
                print(conv3.name, "=", conv3.shape)
                print(maxpool3.name, "=", maxpool3.shape)
                print(dense_1.name, "=", dense_1.shape)
                print(self.z_mean.name, "=", self.z_mean.shape)
                print(self.z_stddev.name, "=", self.z_stddev.shape)

    def decoder(self, encoded):
        # ======================
        #       Decoder
        # ======================
        with tf.variable_scope('decoder'):
            # ======================================
            #  dense_2 = (?, 512) --> (?, 1024)
            # ======================================
            dense_2 = tf.layers.dense(encoded, 1024, activation=tf.nn.leaky_relu, name="dense_2")

            # ======================================
            #  dense_3 = (?, 1113) --> (?, 7200)
            # ======================================
            dense_3 = tf.layers.dense(dense_2, 30*30*8, activation=tf.nn.leaky_relu, name="dense_3")

            # ========================
            # dense1 = (?, 30, 30, 8)
            # ========================
            reshape1 = tf.reshape(dense_3, (-1, 30, 30, 8), name="reshaped")

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
            #  up_sample2 = (?, 120, 120, 8)
            # ==============================
            up_sample2 = tf.image.resize_nearest_neighbor(conv5, (120, 120), name="upsample2")

            # ==============================
            #  conv6 = (?, 120, 120, 16)
            # ==============================
            conv6 = tf.layers.conv2d(up_sample2, 16, (3, 3), padding='same', activation=tf.nn.leaky_relu, name="conv6")

            # ==============================
            #  up_sample3 = (?, 240, 240, 16)
            # ==============================
            up_sample3 = tf.image.resize_nearest_neighbor(conv6, (240, 240), name="upsample3")

            # ==============================
            #  conv7 = (?, 240, 240, 16)
            # ==============================
            conv7 = tf.layers.conv2d(up_sample3, 16, (3, 3), padding='same', activation=tf.nn.leaky_relu, name="conv7")

            logits = tf.layers.conv2d(conv7, 1, (3, 3), padding='same', activation=tf.nn.sigmoid, name="logits")
            self.decoded = logits

            self.nodes += [dense_2.name, reshape1.name, conv4.name, up_sample1.name,
                           conv5.name, up_sample2.name, conv6.name]

            self.nodes += [up_sample3.name, conv7.name, logits.name]

            if self.summary:
                print("encoded", encoded.shape)
                print(dense_2.name, "=", dense_2.shape)
                print(dense_3.name, "=", dense_3.shape)
                print(reshape1.name, "=", reshape1.shape)
                print(conv4.name, "=", conv4.shape)
                print(up_sample1.name, "=", up_sample1.shape)
                print(conv5.name, "=", conv5.shape)
                print(up_sample2.name, " =", up_sample2.shape)
                print(conv6.name, "=", conv6.shape)
                print(up_sample3.name, "=", up_sample3.shape)
                print(conv7.name, "=", conv7.shape)
                print(logits.name, "=", logits.shape)
                self.process_node_names()


def conv2d_layer(x, filter_w, in_d, out_d):
    conv_w = tf.Variable(tf.truncated_normal(shape=(filter_w, filter_w, in_d, out_d), mean=0.0, stddev=0.1))
    conv_b = tf.Variable(tf.zeros(out_d))
    conv_layer = tf.nn.conv2d(x, conv_w, strides=[1, 2, 2, 1], padding='SAME') + conv_b
    return conv_layer


def de_conv(inputs, out_d):
    conv_trans = tf.layers.conv2d_transpose(inputs, out_d, kernel_size=[3, 3], strides=(2, 2), padding='SAME')
    return conv_trans


def dense_layer(x, in_d, out_d):
    w = tf.Variable(tf.truncated_normal(shape=[in_d, out_d], mean=0.0, stddev=0.1))
    b = tf.Variable(tf.zeros(out_d))
    return tf.matmul(x, w) + b


if __name__ == "__main__":

    vae = ConVAE(summary=True)
    vae.build_model()
