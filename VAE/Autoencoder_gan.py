
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.layers as tcl
import numpy as np
import timeit
import os
import data_utils
import h5py
import time

PATH = '/media/li/e6622bf6-7f16-4fa9-8a1c-f656feb70e68/soe/PycharmProjects/Convolutional-Sequence-to-Sequence-Model-for-Human-Dynamics-master'

class Autoenc_gan():
    """Baseline variantional auto-encoder. Note that we also use adversirial trainning inside the VAE.

    Args:
      encoder (EncoderBase) : the encoder which encode a sample to a hidden variable.
      decoder (DecoderBase) : the decoder which maps a hidden variable to a sample.
      discriminator (DiscriminatorBase) : the discriminator which is used to measure if a selfample belongs to the real sample.
      length_input: the length of the input sequence.
      length_output: the length of the output sequence.
      modelname:name of the trained model.
      gan_loss_weight:the weight the discriminator loss.
      action:to specify whether use all the actions in the dataset.
      sampling: the weight for sampling for the previous output, the other part is from the ground truth.
      window_length:the input length of the short term CEM.
      is_sampling:to specify the phase, training or sampling
      dataset_name: specify which dataset to use, human3.6m or cmu
      weight_decay (float): the weight decay parameter (i.e. weight of l2 regularizer).
      learning_rate (float): learning_rate of optimizer
      learning_rate_decay (float): a parameter from [0, 1]
      learning_rate_decay_steps (int): number of steps for learning rate decay
      dtype (int) : the type of data inside tensorflow, default is tf.float32.
  """
    
    def __init__(self, encoder,
                 decoder,
                 local_discriminator,
                 global_discriminator,
                 whole_sample_dim,
                 length_input,
                 length_output,
                 modelname='CNNAdTrain',
                 gan_loss_weight= 0.01,
                 action="all",
                 sampling=0.95,
                 window_length=20,
                 concat_input_output=True,
                 trainable=True,
                 is_sampling=False,
                 dataset_name='human3.6m',
                 reuse=False,
                 weight_decay=2e-5,
                 learning_rate=2e-4,
                 learning_rate_decay=0.99,
                 learning_rate_decay_steps=10000,
                 iterations=20000,
                 display_every=100,
                 test_every=1000,
                 batch_size=64,
                 dtype=tf.float32,
                 name_scope='VAESkeleton'):
        # encoder, decoder and discriminator
        assert (dtype == encoder.dtype and dtype == decoder.dtype)
        self.encoder = encoder
        self.decoder = decoder
        self.local_discriminator = local_discriminator
        self.global_discriminator = global_discriminator
        self.length_input=length_input
        self.length_output=length_output
        self.action=action
        self.concat_input_output = concat_input_output
        self.gan_loss_weight = gan_loss_weight
        self.modelname = modelname
        self.window_size=window_length
        self.is_sampling=is_sampling
        self.dataset_name=dataset_name
        self.is_training = tf.placeholder(tf.bool)
        
        ### parameters for optimizer
        self.batch_size=batch_size
        self.iterations=iterations
        self.display=display_every
        self.test_every=test_every
        self.lr = learning_rate
        self.Dlr = learning_rate # learning rate for Discriminators
        self.lr_decay = learning_rate_decay
        self.lr_decay_steps = learning_rate_decay_steps
        self.weight_decay = weight_decay

        self.lr_ph = tf.placeholder(dtype)
        ### Network structure parameters
        self.enc_hidden_num = self.encoder.encoded_desc['hidden_num']
        try:
            self.enc_class_num = self.encoder.encoded_desc['class_num']
        except:
            self.enc_class_num = 0

        self.whole_sample_dim = whole_sample_dim
        
        self.encoder_input = tf.placeholder(dtype, encoder.input_dim)
        self.discriminator_input = tf.placeholder(dtype, whole_sample_dim)

        if (self.enc_class_num is not 0):
            self.yhat = tf.placeholder(dtype, [encoder.input_dim[0], self.enc_class_num])
        else:
            self.yhat = None
            #yhat is the one-hot vector of action type
        self.trainable = trainable
        self.reuse = reuse
        self.steps = 0
        self.global_step = tf.Variable(0, trainable=False)
        with tf.variable_scope(name_scope) as vs:
            if (self.reuse):
                vs.reuse_variables()
            self.ConstructNet(sampling,window_length)
            if (trainable):
                self.ConstructOptimizer()
        self.Saver = tf.train.Saver()

    def Train(self, sess, dataset ):
        """
        Need to be modified
        :param sess:
        :param dataTrain:
        :param dataval:
        :param Epoches:
        :param batch_size:
        :return:
        """
        all_lossd = []
        all_lossg = []
        start=timeit.default_timer()
        for i in range(self.iterations):
            encoder_data, discriminator_data, yhat = dataset.get_train_batch(self.batch_size)
            loss_d = self.TrainDStep(sess, encoder_data, yhat, discriminator_data)
            encoder_data, discriminator_data, yhat = dataset.get_train_batch(self.batch_size)
            loss_g = self.TrainGStep(sess, encoder_data, yhat, discriminator_data)
            
            #loss_d, loss_g = self.TrainOneStep(sess, encoder_data, yhat, discriminator_data)
            all_lossd += [loss_d]
            all_lossg += [loss_g]
            self.steps = self.steps + 1
            if (self.steps >= self.lr_decay_steps and (self.steps % self.lr_decay_steps == 0)):
                self.lr = self.lr * self.lr_decay
            if (self.steps%self.display==0):
                time_elasped = timeit.default_timer() - start
                print('Iterations %d loss_d %f, loss_g %f, lr %f, time %f' % (i, np.mean(all_lossd), np.mean(all_lossg), self.lr, time_elasped))
                import sys
                sys.stdout.flush()
                all_lossd=[]
                all_lossg=[]
                for action in dataset.actions:
                    encoder_data, discriminator, yhat = dataset.get_test_batch(action)
                    self.TestSample(sess, encoder_data, discriminator,action)
            if(self.steps % 500 == 0):
                self.InferenceSample(sess, dataset,self.steps)
                #self.Saver.save(sess, self.modelname, global_step=self.steps)
            if(self.steps % 500 == 0):
                self.Saver.save(sess, 'Models/'+self.modelname, global_step=self.steps)


        self.Saver.save(sess, 'Models/'+self.modelname, global_step=self.steps)

    def TestSample(self, sess, encoder_input, decoder_expect_output,action):
        if self.dataset_name=='human3.6m':
            if action=='walking':

                predict_pose, summ = sess.run([self.test_res, self.walking_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'eating':
                predict_pose, summ = sess.run([self.test_res, self.eating_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'smoking':
                predict_pose, summ = sess.run([self.test_res, self.smoking_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'discussion':
                predict_pose, summ = sess.run([self.test_res, self.discussion_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'directions':
                predict_pose, summ = sess.run([self.test_res, self.directions_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'greeting':
                predict_pose, summ = sess.run([self.test_res, self.greeting_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'phoning':
                predict_pose, summ = sess.run([self.test_res, self.phoning_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'posing':
                predict_pose, summ = sess.run([self.test_res, self.posing_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'purchases':
                predict_pose, summ = sess.run([self.test_res, self.purchases_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'sitting':
                predict_pose, summ = sess.run([self.test_res, self.sitting_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'sittingdown':
                predict_pose, summ = sess.run([self.test_res, self.sittingdown_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'takingphoto':
                predict_pose, summ = sess.run([self.test_res, self.takingphoto_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'waiting':
                predict_pose, summ = sess.run([self.test_res, self.waiting_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'walkingdog':
                predict_pose, summ = sess.run([self.test_res, self.walkingdog_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'walkingtogether':
                predict_pose, summ = sess.run([self.test_res, self.walkingtogether_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
        else:
            if action=='basketball':

                predict_pose, summ = sess.run([self.test_res, self.basketball_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'basketball_signal':
                predict_pose, summ = sess.run([self.test_res, self.basketball_signal_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'directing_traffic':
                predict_pose, summ = sess.run([self.test_res, self.directing_traffic_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'jumping':
                predict_pose, summ = sess.run([self.test_res, self.jumping_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'running':
                predict_pose, summ = sess.run([self.test_res, self.running_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'soccer':
                predict_pose, summ = sess.run([self.test_res, self.soccer_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'walking':
                predict_pose, summ = sess.run([self.test_res, self.walking_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)
            if action == 'washwindow':
                predict_pose, summ = sess.run([self.test_res, self.washwindow_error_summ],
                                              feed_dict={self.encoder_input: encoder_input,
                                                         self.discriminator_input: decoder_expect_output,
                                                         self.is_training: False})
                self.log_file_write.add_summary(summ, self.steps)


        
    def InferenceSample(self, sess,dataset,iter):

        one_hot=False
        srnn_gts_expmap = dataset.get_srnn_gts(one_hot, to_euler=False)
        srnn_gts_euler=dataset.get_srnn_gts(one_hot, to_euler=True)

        SAMPLES_FNAME = PATH + "/samples/{}-{}.h5".format(self.modelname,iter)
        # try:
        #     os.remove(SAMPLES_FNAME)
        # except OSError:
        #     pass
        step_time=[]
        Average = np.zeros(25)
        for action in dataset.actions:

            start_time = timeit.default_timer()
            encoder_input, decoder_expect_output, _ = dataset.get_test_batch(action)


            predict_pose = sess.run(self.test_res,
                           feed_dict={self.encoder_input: encoder_input,
                                      self.discriminator_input: decoder_expect_output, 
                                      self.is_training: False})
            time=timeit.default_timer()-start_time
            step_time.append(time)
            ######################################
            # calculate the euler prediction error here.
            ######################################
            ActionError = dataset.compute_test_error(action, predict_pose, srnn_gts_expmap, srnn_gts_euler, one_hot,
                                                     SAMPLES_FNAME)
            # ActionError is a matrix which ActionError[ms] = error of the action
            Average = Average + ActionError

        Average = Average / 8
        ### print the average errors
        print()
        print("{0: <16} |".format("milliseconds"), end="")
        for ms in [80, 160, 320, 400, 560, 640, 720, 840, 1000]:
            print(" {0:5d} |".format(ms), end="")
        print()
        print("{0: <16} |".format("Average"), end="")
        for ms in [1, 3, 7, 9, 13, 15, 17, 20, 24]:
            print(" {0:.3f} |".format(Average[ms]), end="")
        print()
        ############################
        print(np.mean(step_time))

    def TrainDStep(self, sess, encoder_input, yhat, discriminator_input):
        d_updates, loss_d,  loss_d_summ = sess.run([self.dupdates, self.loss_d, self.loss_d_summ],
                                                   feed_dict={self.encoder_input: encoder_input,
                                                              self.yhat: yhat,
                                                              self.discriminator_input: discriminator_input,
                                                              self.lr_ph: self.Dlr,
                                                              self.is_training: True})
        self.log_file_write.add_summary(loss_d_summ, self.steps)
        return loss_d
    def TrainGStep(self, sess, encoder_input, yhat, discriminator_input):
        g_update, loss_g, loss_l2_summ, loss_g_summ = sess.run([self.gupdates,
                                                  self.loss_Auto,
                                                  self.loss_l2_summ,
                                                  self.loss_g_summ],
                                                 feed_dict={self.encoder_input: encoder_input,
                                                            self.yhat: yhat,
                                                            self.discriminator_input: discriminator_input,
                                                            self.lr_ph: self.lr,
                                                            self.is_training: True})
        self.log_file_write.add_summary(loss_g_summ, self.steps)
        self.log_file_write.add_summary(loss_l2_summ, self.steps)
        return loss_g
        
    def TrainOneStep(self, sess, encoder_input, yhat, discriminator_input):
            #loss_d = 0
        
        return loss_d, loss_g

    def ConstructOptimizer(self):
        self.global_d_vars = [var for var in tf.trainable_variables() if self.global_discriminator.name_scope in var.name]
        self.local_d_vars = [var for var in tf.trainable_variables() if self.local_discriminator.name_scope in var.name]
        self.d_vars = self.global_d_vars + self.local_d_vars
        self.g_vars = [var for var in tf.trainable_variables() if self.decoder.name_scope in var.name] + [var for var in tf.trainable_variables() if self.encoder.name_scope in var.name]
        self.d_optimim = tf.train.AdamOptimizer(self.lr_ph)
        self.g_optimim = tf.train.AdamOptimizer(self.lr_ph)

        # max_gradient_norm = 1
        
        dgradients = self.d_optimim.compute_gradients(self.loss_d_reg, self.d_vars)
        ggradients = self.g_optimim.compute_gradients(self.loss_Auto_reg, self.g_vars)

        
        # clipped_dgradients = [ (tf.clip_by_value(grad, -max_gradient_norm, max_gradient_norm), var) for grad, var in dgradients]
        # clipped_ggradients = [ (tf.clip_by_value(grad, -max_gradient_norm, max_gradient_norm), var) for grad, var in ggradients]

        self.dupdates = self.d_optimim.apply_gradients(dgradients,
                                                       global_step=self.global_step)
        self.gupdates = self.g_optimim.apply_gradients(ggradients,
                                                       global_step=self.global_step)
        
        
    def ConstructNet(self, sampling,window_length):
        """Construct the network structure (i.e. computational graph) for tensorflow.
        """

        hidden = self.encoder.forward(self.encoder_input, self.trainable, self.is_training)
        loss = tf.constant(0.0)
        if (self.enc_class_num is not 0):

            hlayer, hlogits = tf.split(hidden,
                                       [self.enc_hidden_num, self.enc_class_num],
                                       axis=1)
            loss += tf.losses.softmax_cross_entropy(self.yhat,
                                                    hlogits) # classificy loss
            hcategory = tf.nn.softmax(hlogits)  # classificy loss
        else:
            hlayer = hidden
            hcategory = None

        predicted_res = []
        windowLength = window_length
        seqStart = self.length_input - windowLength

        # the training part
        if (not self.is_sampling):
            # This two line select 20 frames from 0 - current frames. It is my encoder
            select = np.linspace(0,(windowLength + seqStart-1), 20, dtype=int, endpoint=True)
            S_index = tf.convert_to_tensor(select)
            my_dec_in0 = tf.gather(self.discriminator_input[:, :, :, :], S_index, axis=1)
            temp = tf.split(self.discriminator_input[:, :50, :, :], 50, axis=1)
            ##########################################################################

            dec_in0 = self.discriminator_input[:, seqStart:(windowLength + seqStart), :, :]
            for it in range(self.length_output):    # sampling from both predicted frame and groundtruth frame during training
                if(sampling == 0):
                    dec_in0 = self.discriminator_input[:, (seqStart+it):(windowLength + it +seqStart), :, :]
                if(it == 0):
                    layer_reuse = False
                else:
                    layer_reuse = True
                dec_out = self.decoder.forward(hlayer, dec_in0, hlogits,
                                               reuse=layer_reuse,
                                               trainable=self.trainable,
                                               is_training=self.is_training)




                ###### uncomment when do ablation study where there is no long term CEM, namely the input does not contains hlayer
                # dec_out = self.decoder.forward(dec_in0,
                #                                reuse=layer_reuse,
                #                                trainable=self.trainable,
                #                                is_training=self.is_training)

                last_input = tf.split(my_dec_in0, [windowLength - 1, 1], axis=1)
                final_out = dec_out + last_input[1]   #residual connection


                if(sampling > 0):
                    new_gt = self.discriminator_input[:, self.length_input + it + 1 : self.length_input + it + 2, :, :]
                    dec_in0 = tf.split(dec_in0, [1, windowLength - 1], axis=1)
                    dec_in0 = tf.concat([dec_in0[1], sampling * final_out + (1 - sampling) * new_gt], axis=1)

                    ## update my_dec_in0 with final_out
                    temp = temp + [sampling * final_out + (1 - sampling) * new_gt] # the list of tensor
                    select = np.linspace(0, self.length_input + it, 20, dtype=int, endpoint=True) # the index
                    S_tensor = [temp[key] for key in select]
                    my_dec_in0 = tf.concat(S_tensor, axis=1)



                predicted_res += [final_out]

            predicted_res = tf.concat(predicted_res, axis=1)
            if (self.concat_input_output):
                generated_sample = tf.concat(
                    [self.encoder_input, self.discriminator_input[:, 50:51, :, :], predicted_res], axis=1)
            else:
                generated_sample = predicted_res

            self.res = predicted_res

            predicted_expected = self.discriminator_input[:, 50:, :, :]




            #############################################################################################
            ## The original loss is ReconstructError here. We comment it first
            ## ReconstructError is the loss of Ground truth and output !!! We will change it into D-loss

            #ReconstructError = tf.reduce_mean(tf.square(tf.subtract(predicted_expected, self.test_res)))
            # predicted_expected is ground truth, and predicted_res is output sequences.
            #############################################################################################
            # define my own Dloss
            batchsize = 64
            featuresize = 70 # CMU 70, H3.6M 54
            Dloss_weights = tf.constant(1, dtype=tf.float32, shape=[batchsize, 1, featuresize, 1])
            Fibonacci = tf.constant([1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765,10946,17711,28657,46368,75025,121393], dtype=tf.float32)
            FibonacciA = np.array([1,2,3,5,8,13,21,34,55,89,144,233,377,610,987,1597,2584,4181,6765,10946,17711,28657,46368,75025,121393])
            Exp1 = tf.constant([1,1.2,1.44,1.728,2.0736,2.48832,2.985984,3.583181,4.299817,5.1597805,6.191736,7.4300838,8.9161005,10.699321,12.839185,15.4070215,18.488426,22.186111,26.623333,31.948,38.3376,46.00512,55.206142,66.247375,79.49685], dtype=tf.float32)
            Exp2 = tf.constant([0.9,0.81,0.729,0.6561,0.59049,0.531441,0.4782969,0.43046722,0.38742048,0.34867844,0.3138106,0.28242955,0.25418657,0.22876793,0.20589113,0.18530202,0.16677181,0.15009463,0.13508517,0.12157665,0.10941899,0.09847709,0.08862938,0.07976644,0.0717898], dtype=tf.float32)
            Exp3 = tf.constant([1,0.8,0.64,0.512,0.4096,0.32768,0.262144,0.2097152,0.16777216,0.13421772,0.10737418,0.08589935,0.06871948,0.05497558,0.04398046,0.03518437,0.0281475,0.022518,0.0180144,0.01441152,0.01152921,0.00922337,0.0073787,0.00590296,0.00472237], dtype=tf.float32)
            a = tf.shape(predicted_expected)[0] # the number of output frames

            Z = tf.zeros([25], dtype=tf.float32)
            Use = tf.concat([Fibonacci[0:a], Z[a:25]], 0)
            Use2 = tf.concat([Exp2[0:a], Z[a:25]], 0)
            total = tf.reduce_sum(Use2)

            eta = 1 # eta**(num+1)


            for num in range(24):
                #weights1 = tf.div(tf.constant(FibonacciA[num], dtype=tf.float32, shape=[batchsize, 1, featuresize, 1]), total)
                weights1 = tf.constant(eta**(num+2), dtype=tf.float32, shape=[batchsize, 1, featuresize, 1])
                Dloss_weights = tf.concat([Dloss_weights, weights1], 1)

            #a = predicted_expected.shape.as_list()
            DW = tf.div(tf.multiply(tf.cast(a, tf.float32), Dloss_weights), total)




            #Dloss_weights = tf.ones_like(predicted_expected)# predicted_expected shape = [None, 25, 54, 1]

            Dloss = tf.reduce_mean(tf.multiply(Dloss_weights[0:a,:,:,:], tf.square(tf.subtract(predicted_expected, predicted_res))))
            ReconstructError = Dloss

            b = Dloss_weights[0, :, 0, 0]
            sess = tf.Session()
            print(sess.run(b))




            ###########################################################################################

            # self.test_error = tf.reduce_mean(tf.square(tf.subtract(predicted_expected, self.test_res)))
            self.test_error =ReconstructError

            ##############################################################################################
            ## global discriminator##########################################################
            G_d_logits = self.global_discriminator.forward(self.discriminator_input, self.yhat, reuse=self.reuse,
                                                  trainable=self.trainable, is_training=self.is_training)
            G_d_logits_ = self.global_discriminator.forward(generated_sample, self.yhat, reuse=True,
                                                   trainable=self.trainable, is_training=self.is_training)

            G_d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=G_d_logits, labels=tf.ones_like(G_d_logits)))
            G_d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=G_d_logits_, labels=tf.zeros_like(G_d_logits_)))
            G_Adv_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=G_d_logits_, labels=tf.ones_like(G_d_logits_)))

            self.Gloss_d = G_d_loss_real + G_d_loss_fake
            ##local discriminator ###########################################################
            L_d_logits = self.local_discriminator.forward(self.discriminator_input[:, 50:, :, :], self.yhat, reuse=self.reuse,
                                                           trainable=self.trainable, is_training=self.is_training)
            L_d_logits_ = self.local_discriminator.forward(generated_sample[:, 50:, :, :], self.yhat, reuse=True,
                                                            trainable=self.trainable, is_training=self.is_training)

            L_d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=L_d_logits, labels=tf.ones_like(L_d_logits)))
            L_d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=L_d_logits_, labels=tf.zeros_like(L_d_logits_)))
            L_Adv_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(logits=L_d_logits_, labels=tf.ones_like(L_d_logits_)))

            self.Lloss_d = L_d_loss_real + L_d_loss_fake


            ### calculate the total discriminator loss
            self.loss_d = 0.5 * (self.Gloss_d + self.Lloss_d)

            Adv_loss = 0.5 *(G_Adv_loss + L_Adv_loss)

            all_reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            print(all_reg)

            global_d_reg = [var for var in all_reg if self.global_discriminator.name_scope in var.name]
            local_d_reg = [var for var in all_reg if self.local_discriminator.name_scope in var.name]
            d_reg = global_d_reg + local_d_reg
            print(d_reg)

            print('discriminator regularizers')
            for w in d_reg:
                shp = w.get_shape().as_list()
                print("- {} shape:{} size:{}".format(w.name, shp, np.prod(shp)))

            self.loss_d_reg = self.loss_d + tf.reduce_sum(d_reg)

            enc_reg = [var for var in all_reg if self.encoder.name_scope in var.name]

            print('encoder regularizers')
            for w in enc_reg:
                shp = w.get_shape().as_list()
                print("- {} shape:{} size:{}".format(w.name, shp, np.prod(shp)))

            dec_reg = [var for var in all_reg if self.decoder.name_scope in var.name]

            print('decoder regularizers')
            for w in dec_reg:
                shp = w.get_shape().as_list()
                print("- {} shape:{} size:{}".format(w.name, shp, np.prod(shp)))

            self.loss_Auto = ReconstructError + self.gan_loss_weight * Adv_loss

            self.loss_Auto_reg = self.loss_Auto + tf.reduce_sum(enc_reg) + tf.reduce_sum(dec_reg)

            self.loss_d_summ = tf.summary.scalar('loss_d', self.loss_d)
            self.loss_g_summ = tf.summary.scalar('loss_g', self.loss_Auto)
            self.loss_c_summ = tf.summary.scalar('loss_c', loss)
            self.loss_l2_summ = tf.summary.scalar('loss_l2', ReconstructError)
            self.loss_test_summ = tf.summary.scalar('loss_test', self.test_error)

            with tf.variable_scope('TestErrors') as vs1:
                if self.dataset_name=='human3.6m':
                    self.walking_error_summ = tf.summary.scalar('loss_walking', self.test_error)
                    self.eating_error_summ = tf.summary.scalar('loss_eating', self.test_error)
                    self.smoking_error_summ = tf.summary.scalar('loss_smoking', self.test_error)
                    self.discussion_error_summ = tf.summary.scalar('loss_discussion', self.test_error)
                    self.directions_error_summ = tf.summary.scalar('loss_directions', self.test_error)
                    self.greeting_error_summ = tf.summary.scalar('loss_greeting', self.test_error)
                    self.phoning_error_summ = tf.summary.scalar('loss_phoning', self.test_error)
                    self.posing_error_summ = tf.summary.scalar('loss_posing', self.test_error)
                    self.purchases_error_summ = tf.summary.scalar('loss_purchases', self.test_error)
                    self.sitting_error_summ = tf.summary.scalar('loss_sitting', self.test_error)
                    self.sittingdown_error_summ = tf.summary.scalar('loss_sittingdown', self.test_error)
                    self.takingphoto_error_summ = tf.summary.scalar('loss_takingphoto', self.test_error)
                    self.waiting_error_summ = tf.summary.scalar('loss_waiting', self.test_error)
                    self.walkingdog_error_summ = tf.summary.scalar('loss_walkingdog', self.test_error)
                    self.walkingtogether_error_summ = tf.summary.scalar('loss_walkingtogether', self.test_error)
                else:
                    self.basketball_error_summ = tf.summary.scalar('loss_basketball', self.test_error)
                    self.basketball_signal_error_summ = tf.summary.scalar('loss_basketball_signal', self.test_error)
                    self.directing_traffic_error_summ = tf.summary.scalar('loss_directing_traffic', self.test_error)
                    self.jumping_error_summ = tf.summary.scalar('loss_jumping', self.test_error)
                    self.running_error_summ = tf.summary.scalar('loss_running', self.test_error)
                    self.soccer_error_summ = tf.summary.scalar('loss_soccer', self.test_error)
                    self.walking_error_summ = tf.summary.scalar('loss_walking', self.test_error)
                    self.washwindow_error_summ = tf.summary.scalar('loss_washwindow', self.test_error)

            # self.g_img_summ = tf.summary.image('generated_sample', generated_sample)
            # self.real_img_summ = tf.summary.image('real_sample', self.discriminator_input)

            self.log_file_write = tf.summary.FileWriter('logs/' + self.modelname)


        ##The test part
        test_predict_res = []
        print(seqStart)

        # This two line select 20 frames from 0 - current frames. It is my encoder
        select = np.linspace(0, (windowLength + seqStart - 1), 20, dtype=int, endpoint=True)
        S_index = tf.convert_to_tensor(select)
        my_dec_in0 = tf.gather(self.discriminator_input[:, :, :, :], S_index, axis=1)
        temp = tf.split(self.discriminator_input[:, :50, :, :], 50, axis=1)
        ##########################################################################
        dec_in0 = self.discriminator_input[:, (seqStart):(windowLength + seqStart), :, :]

        for it in range(self.length_output):    # no groundtruth during test
            if ((it==0) & self.is_sampling):
                layer_reuse=False
            else:
                layer_reuse=True

            dec_out = self.decoder.forward(hlayer, dec_in0, hcategory,
                                           reuse=layer_reuse,
                                           trainable=self.trainable,
                                           is_training=self.is_training)
            ###### uncomment when do ablation study where there is no long term CEM, namely the input does not contains hlayer
            # dec_out = self.decoder.forward(dec_in0,
            #                                reuse=layer_reuse,
            #                                trainable=self.trainable,
            #                                is_training=self.is_training)
            last_input = tf.split(my_dec_in0, [windowLength - 1, 1], axis=1)
            final_out = dec_out + last_input[1] #Residual link


            dec_in0 = tf.split(dec_in0, [1, windowLength - 1], axis=1)
            dec_in0 = tf.concat([dec_in0[1], final_out], axis=1)

            ## update my_dec_in0 with final_out
            temp = temp + [final_out]  # the list of tensor
            select = np.linspace(0, self.length_input + it, 20, dtype=int, endpoint=True)  # the index
            S_tensor = [temp[key] for key in select]
            my_dec_in0 = tf.concat(S_tensor, axis=1)


            test_predict_res += [final_out]

        self.test_res = tf.concat(test_predict_res, axis=1)






        
        
