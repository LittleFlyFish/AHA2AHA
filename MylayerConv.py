import VAE

import tensorflow as tf
import tensorflow.contrib.layers as tcl
from keras.models import Model
from keras.layers import Input, Dense

from attention import attention

import os
import numpy as np

import math
from tensorflow.python.ops import math_ops


from keras import backend as K
from keras.engine.topology import Layer

############################################################
# This layer change all the FCN calculation to Conv Calculation
# The Hlayer defination of Human3.6M
# dimensions to use (the corresponding si):
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 21, 22, 23, 24, 27, 28,
# 29, 30, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 51, 52, 53, 54,
# 55, 56, 57, 60, 61, 62, 75, 76, 77, 78, 79, 80, 81, 84, 85, 86]
# based on these dimension and the human body graph I draw and posed on the website
# https://github.com/una-dinosauria/human-motion-prediction/issues/46
## the si sj lined by limb will be send to one neuron
# the global rotation and translation are used in the 54 dimension but not calculate in error.
##############################################################
class HLayer1(tf.keras.layers.Layer):
  def __init__(self, re_term, node, kernel_size=2):
    super(HLayer1, self).__init__()
    self.re_term = re_term
    self.node = node
    self.kernel_size = kernel_size

  def call(self, input):

      lrelu = VAE.lrelu
      re_term = self.re_term
      s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, \
      s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, \
      s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, \
      s31, s32, s33, s34, s35, s36, s37, s38, s39, s40, \
      s41, s42, s43, s44, s45, s46, s47, s48, s49, s50, \
      s51, s52, s53 = tf.split(input, num_or_size_splits=54, axis=2)

      ### reshape the s from four dim to three dim
      R0 = tf.keras.layers.Reshape((-1, 1))
      s0 = R0(s0)
      R1 = tf.keras.layers.Reshape((-1, 1))
      s1 = R1(s1)
      R2 = tf.keras.layers.Reshape((-1, 1))
      s2 = R2(s2)
      R3 = tf.keras.layers.Reshape((-1, 1))
      s3 = R3(s3)
      R4 = tf.keras.layers.Reshape((-1, 1))
      s4 = R4(s4)
      R5 = tf.keras.layers.Reshape((-1, 1))
      s5 = R5(s5)
      R6 = tf.keras.layers.Reshape((-1, 1))
      s6 = R6(s6)
      R7 = tf.keras.layers.Reshape((-1, 1))
      s7 = R7(s7)
      R8 = tf.keras.layers.Reshape((-1, 1))
      s8 = R8(s8)
      R9 = tf.keras.layers.Reshape((-1, 1))
      s9 = R9(s9)
      R10 = tf.keras.layers.Reshape((-1, 1))
      s10 = R10(s10)
      R11 = tf.keras.layers.Reshape((-1, 1))
      s11 = R11(s11)
      R12 = tf.keras.layers.Reshape((-1, 1))
      s12 = R12(s12)
      R13 = tf.keras.layers.Reshape((-1, 1))
      s13 = R13(s13)
      R14 = tf.keras.layers.Reshape((-1, 1))
      s14 = R14(s14)
      R15 = tf.keras.layers.Reshape((-1, 1))
      s15 = R15(s15)
      R16 = tf.keras.layers.Reshape((-1, 1))
      s16 = R16(s16)
      R17 = tf.keras.layers.Reshape((-1, 1))
      s17 = R17(s17)
      R18 = tf.keras.layers.Reshape((-1, 1))
      s18 = R18(s18)
      R19 = tf.keras.layers.Reshape((-1, 1))
      s19 = R19(s19)
      R20 = tf.keras.layers.Reshape((-1, 1))
      s20 = R20(s20)
      R21 = tf.keras.layers.Reshape((-1, 1))
      s21 = R21(s21)
      R22 = tf.keras.layers.Reshape((-1, 1))
      s22 = R22(s22)
      R23 = tf.keras.layers.Reshape((-1, 1))
      s23 = R23(s23)
      R24 = tf.keras.layers.Reshape((-1, 1))
      s24 = R24(s24)
      R25 = tf.keras.layers.Reshape((-1, 1))
      s25 = R25(s25)
      R26 = tf.keras.layers.Reshape((-1, 1))
      s26 = R26(s26)
      R27 = tf.keras.layers.Reshape((-1, 1))
      s27 = R27(s27)
      R28 = tf.keras.layers.Reshape((-1, 1))
      s28 = R28(s28)
      R29 = tf.keras.layers.Reshape((-1, 1))
      s29 = R9(s29)
      R30 = tf.keras.layers.Reshape((-1, 1))
      s30 = R30(s30)
      R31 = tf.keras.layers.Reshape((-1, 1))
      s31 = R31(s31)
      R32 = tf.keras.layers.Reshape((-1, 1))
      s32 = R32(s32)
      R33 = tf.keras.layers.Reshape((-1, 1))
      s33 = R33(s33)
      R34 = tf.keras.layers.Reshape((-1, 1))
      s34 = R34(s34)
      R35 = tf.keras.layers.Reshape((-1, 1))
      s35 = R35(s35)
      R36 = tf.keras.layers.Reshape((-1, 1))
      s36 = R36(s36)
      R37 = tf.keras.layers.Reshape((-1, 1))
      s37 = R37(s37)
      R38 = tf.keras.layers.Reshape((-1, 1))
      s38 = R38(s38)
      R39 = tf.keras.layers.Reshape((-1, 1))
      s39 = R39(s39)
      R40 = tf.keras.layers.Reshape((-1, 1))
      s40 = R40(s40)
      R41 = tf.keras.layers.Reshape((-1, 1))
      s41 = R41(s41)
      R42 = tf.keras.layers.Reshape((-1, 1))
      s42 = R42(s42)
      R43 = tf.keras.layers.Reshape((-1, 1))
      s43 = R43(s43)
      R44 = tf.keras.layers.Reshape((-1, 1))
      s44 = R44(s44)
      R45 = tf.keras.layers.Reshape((-1, 1))
      s45 = R45(s45)
      R46 = tf.keras.layers.Reshape((-1, 1))
      s46 = R46(s46)
      R47 = tf.keras.layers.Reshape((-1, 1))
      s47 = R47(s47)
      R48 = tf.keras.layers.Reshape((-1, 1))
      s48 = R48(s48)
      R49 = tf.keras.layers.Reshape((-1, 1))
      s49 = R49(s49)
      R50 = tf.keras.layers.Reshape((-1, 1))
      s50 = R50(s50)
      R51 = tf.keras.layers.Reshape((-1, 1))
      s51 = R51(s51)
      R52 = tf.keras.layers.Reshape((-1, 1))
      s52 = R52(s52)
      R53 = tf.keras.layers.Reshape((-1, 1))
      s53 = R53(s53)

      #######################################################################
      # T is the output nodes, this is the network structure

      T0 = tf.concat([s0, s1, s2, s3, s4, s5, s6, s7, s8], axis=2)
      C0 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                     padding='same', activation=None,
                                     kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T0 = lrelu(C0(T0))
      
      T1 = tf.concat([s6, s7, s8, s9], axis=2)
      C1 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                     padding='same', activation=None,
                                     kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T1 = lrelu(C1(T1))

      T2 = tf.concat([s9, s10, s11, s12], axis=2)
      C2 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T2 = lrelu(C2(T2))

      T3 = tf.concat([s10, s11, s12, s13], axis=2)
      C3 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T3 = lrelu(C3(T3))

      T4 = tf.concat([s0, s1, s2, s3, s4, s5, s14, s15, s16], axis=2)
      C4 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T4 = lrelu(C4(T4))

      T5 = tf.concat([s14, s15, s16, s17], axis=2)
      C5 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T5 = lrelu(C5(T5))

      T6 = tf.concat([s17, s18, s19, s20], axis=2)
      C6 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T6 = lrelu(C6(T6))

      T7 = tf.concat([s18, s19, s20, s21], axis=2)
      C7 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T7 = lrelu(C7(T7))

      T8 = tf.concat([s0, s1, s2, s3, s4, s5, s22, s23, s24], axis=2)
      C8 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T8 = lrelu(C8(T8))

      T9 = tf.concat([s22, s23, s24, s25, s26, s27], axis=2)
      C9 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T9 = lrelu(C9(T9))

      T10 = tf.concat([s25, s26, s27, s28, s29, s30], axis=2)
      C10 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T10 = lrelu(C10(T10))

      T11 = tf.concat([s28, s29, s30, s31, s32, s33], axis=2)
      C11 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T11 = lrelu(C11(T11))

      T12 = tf.concat([s25, s26, s27, s34, s35, s36], axis=2)
      C12 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T12 = lrelu(C12(T12))

      T13 = tf.concat([s34, s35, s36, s37, s38, s39], axis=2)
      C13 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T13 = lrelu(C13(T13))
      T14 = tf.concat([s37, s38, s39, s40], axis=2)
      C14 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T14 = lrelu(C14(T14))

      T15 = tf.concat([s40, s41, s42, s43], axis=2)
      C15 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T15 = lrelu(C15(T15))

      T16 = tf.concat([s25, s26, s27, s44, s45, s46], axis=2)
      C16 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T16 = lrelu(C16(T16))

      T17 = tf.concat([s44, s45, s46, s47, s48, s49], axis=2)
      C17 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T17 = lrelu(C17(T17))

      T18 = tf.concat([s47, s48, s49, s50], axis=2)
      C18 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T18 = lrelu(C18(T18))

      T19 = tf.concat([s50, s51, s52, s53], axis=2)
      C19 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T19 = lrelu(C19(T19))




      result = tf.concat([T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19], axis=2)

      return result
#########################################################################################################
## This layer generate the parts of legs and arms and body
class HLayer2(tf.keras.layers.Layer):
  def __init__(self, re_term, node, kernel_size=2):
    super(HLayer2, self).__init__()
    self.re_term = re_term
    self.node = node
    self.kernel_size = kernel_size

  def call(self, input):

      lrelu = VAE.lrelu
      re_term = self.re_term
      T0, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19\
          = tf.split(input, num_or_size_splits=20, axis=2)

      #######################################################################
      # T is the output nodes, this is the network structure
      TT0 = tf.concat([T0,T1,T2,T3], axis=2)# Left leg
      CC0 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      TT0 = lrelu(CC0(TT0))

      TT1 = tf.concat([T4, T5, T6, T7], axis=2)# right leg
      CC1 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                   padding='same', activation=None,
                                   kernel_regularizer=tcl.l2_regularizer(self.re_term))
      TT1 = lrelu(CC1(TT1))

      TT2 = tf.concat([T8, T9, T10, T11], axis=2)# body
      CC2 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                   padding='same', activation=None,
                                   kernel_regularizer=tcl.l2_regularizer(self.re_term))
      TT2 = lrelu(CC2(TT2))

      TT3 = tf.concat([T12, T13, T14, T15], axis=2)# Right arm
      CC3 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                   padding='same', activation=None,
                                   kernel_regularizer=tcl.l2_regularizer(self.re_term))
      TT3 = lrelu(CC3(TT3))
      
      TT4 = tf.concat([T15, T16, T17, T18, T19], axis=2) # left arm
      CC4 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                   padding='same', activation=None,
                                   kernel_regularizer=tcl.l2_regularizer(self.re_term))
      TT4 = lrelu(CC4(TT4))





      result = tf.concat([TT0, TT1, TT2, TT3, TT4], axis=2)

      return result

#########################################################################################################
## This layer generate the parts of half and half
class HLayer3(tf.keras.layers.Layer):
  def __init__(self, re_term, node, kernel_size=2):
    super(HLayer3, self).__init__()
    self.re_term = re_term
    self.node = node
    self.kernel_size = kernel_size

  def call(self, input):

      lrelu = VAE.lrelu
      re_term = self.re_term
      TT0, TT1, TT2, TT3, TT4\
          = tf.split(input, num_or_size_splits=5, axis=2)

      #######################################################################
      # T is the output nodes, this is the network structure
      TTT0 = tf.concat([TT0, TT1, TT2], axis=2)# Left leg
      CCC0 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                   padding='same', activation=None,
                                   kernel_regularizer=tcl.l2_regularizer(self.re_term))
      TTT0 = lrelu(CCC0(TTT0))

      TTT1 = tf.concat([TT2, TT3, TT4], axis=2)# right leg
      CCC1 = tf.keras.layers.Conv1D(filters=self.node, kernel_size = self.kernel_size, strides=1,
                                    padding='same', activation=None,
                                    kernel_regularizer=tcl.l2_regularizer(self.re_term))
      TTT1 = lrelu(CCC1(TTT1))







      result = tf.concat([TTT0, TTT1], axis=2)

      return result



############################################################
# The Hlayer defination of CMU
# dimensions to use (the corresponding si):
# []
# based on these dimension and the human body graph I draw and posed on the website
# https://github.com/una-dinosauria/human-motion-prediction/issues/46
## the si sj lined by limb will be send to one neuron
# the global rotation and translation are used in the 70 dimension but not calculate in error.
##############################################################
class CMUHLayer1(tf.keras.layers.Layer):
  def __init__(self, re_term, node, kernel_size=2):
    super(CMUHLayer1, self).__init__()
    self.re_term = re_term
    self.node = node
    self.kernel_size = kernel_size

  def call(self, input):

      lrelu = VAE.lrelu
      re_term = self.re_term
      s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, \
      s11, s12, s13, s14, s15, s16, s17, s18, s19, s20, \
      s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, \
      s31, s32, s33, s34, s35, s36, s37, s38, s39, s40, \
      s41, s42, s43, s44, s45, s46, s47, s48, s49, s50, \
      s51, s52, s53, s54, s55, s56, s57, s58, s59, s60, \
      s61, s62, s63, s64, s65, s66, s67, s68, s69\
          = tf.split(input, num_or_size_splits=70, axis=2)

      ### reshape the s from four dim to three dim, from [None, 20, 54, 1] to be [None, 20, 3]
      R0 = tf.keras.layers.Reshape((-1, 1))
      s0 = R0(s0)
      R1 = tf.keras.layers.Reshape((-1, 1))
      s1 = R1(s1)
      R2 = tf.keras.layers.Reshape((-1, 1))
      s2 = R2(s2)
      R3 = tf.keras.layers.Reshape((-1, 1))
      s3 = R3(s3)
      R4 = tf.keras.layers.Reshape((-1, 1))
      s4 = R4(s4)
      R5 = tf.keras.layers.Reshape((-1, 1))
      s5 = R5(s5)
      R6 = tf.keras.layers.Reshape((-1, 1))
      s6 = R6(s6)
      R7 = tf.keras.layers.Reshape((-1, 1))
      s7 = R7(s7)
      R8 = tf.keras.layers.Reshape((-1, 1))
      s8 = R8(s8)
      R9 = tf.keras.layers.Reshape((-1, 1))
      s9 = R9(s9)
      R10 = tf.keras.layers.Reshape((-1, 1))
      s10 = R10(s10)
      R11 = tf.keras.layers.Reshape((-1, 1))
      s11 = R11(s11)
      R12 = tf.keras.layers.Reshape((-1, 1))
      s12 = R12(s12)
      R13 = tf.keras.layers.Reshape((-1, 1))
      s13 = R13(s13)
      R14 = tf.keras.layers.Reshape((-1, 1))
      s14 = R14(s14)
      R15 = tf.keras.layers.Reshape((-1, 1))
      s15 = R15(s15)
      R16 = tf.keras.layers.Reshape((-1, 1))
      s16 = R16(s16)
      R17 = tf.keras.layers.Reshape((-1, 1))
      s17 = R17(s17)
      R18 = tf.keras.layers.Reshape((-1, 1))
      s18 = R18(s18)
      R19 = tf.keras.layers.Reshape((-1, 1))
      s19 = R19(s19)
      R20 = tf.keras.layers.Reshape((-1, 1))
      s20 = R20(s20)
      R21 = tf.keras.layers.Reshape((-1, 1))
      s21 = R21(s21)
      R22 = tf.keras.layers.Reshape((-1, 1))
      s22 = R22(s22)
      R23 = tf.keras.layers.Reshape((-1, 1))
      s23 = R23(s23)
      R24 = tf.keras.layers.Reshape((-1, 1))
      s24 = R24(s24)
      R25 = tf.keras.layers.Reshape((-1, 1))
      s25 = R25(s25)
      R26 = tf.keras.layers.Reshape((-1, 1))
      s26 = R26(s26)
      R27 = tf.keras.layers.Reshape((-1, 1))
      s27 = R27(s27)
      R28 = tf.keras.layers.Reshape((-1, 1))
      s28 = R28(s28)
      R29 = tf.keras.layers.Reshape((-1, 1))
      s29 = R9(s29)
      R30 = tf.keras.layers.Reshape((-1, 1))
      s30 = R30(s30)
      R31 = tf.keras.layers.Reshape((-1, 1))
      s31 = R31(s31)
      R32 = tf.keras.layers.Reshape((-1, 1))
      s32 = R32(s32)
      R33 = tf.keras.layers.Reshape((-1, 1))
      s33 = R33(s33)
      R34 = tf.keras.layers.Reshape((-1, 1))
      s34 = R34(s34)
      R35 = tf.keras.layers.Reshape((-1, 1))
      s35 = R35(s35)
      R36 = tf.keras.layers.Reshape((-1, 1))
      s36 = R36(s36)
      R37 = tf.keras.layers.Reshape((-1, 1))
      s37 = R37(s37)
      R38 = tf.keras.layers.Reshape((-1, 1))
      s38 = R38(s38)
      R39 = tf.keras.layers.Reshape((-1, 1))
      s39 = R39(s39)
      R40 = tf.keras.layers.Reshape((-1, 1))
      s40 = R40(s40)
      R41 = tf.keras.layers.Reshape((-1, 1))
      s41 = R41(s41)
      R42 = tf.keras.layers.Reshape((-1, 1))
      s42 = R42(s42)
      R43 = tf.keras.layers.Reshape((-1, 1))
      s43 = R43(s43)
      R44 = tf.keras.layers.Reshape((-1, 1))
      s44 = R44(s44)
      R45 = tf.keras.layers.Reshape((-1, 1))
      s45 = R45(s45)
      R46 = tf.keras.layers.Reshape((-1, 1))
      s46 = R46(s46)
      R47 = tf.keras.layers.Reshape((-1, 1))
      s47 = R47(s47)
      R48 = tf.keras.layers.Reshape((-1, 1))
      s48 = R48(s48)
      R49 = tf.keras.layers.Reshape((-1, 1))
      s49 = R49(s49)
      R50 = tf.keras.layers.Reshape((-1, 1))
      s50 = R50(s50)
      R51 = tf.keras.layers.Reshape((-1, 1))
      s51 = R51(s51)
      R52 = tf.keras.layers.Reshape((-1, 1))
      s52 = R52(s52)
      R53 = tf.keras.layers.Reshape((-1, 1))
      s53 = R53(s53)
      R54 = tf.keras.layers.Reshape((-1, 1))
      s54 = R54(s54)
      R55 = tf.keras.layers.Reshape((-1, 1))
      s55 = R55(s55)
      R56 = tf.keras.layers.Reshape((-1, 1))
      s56 = R56(s56)
      R57 = tf.keras.layers.Reshape((-1, 1))
      s57 = R57(s57)
      R58 = tf.keras.layers.Reshape((-1, 1))
      s58 = R58(s58)
      R59 = tf.keras.layers.Reshape((-1, 1))
      s59 = R59(s59)
      R60 = tf.keras.layers.Reshape((-1, 1))
      s60 = R60(s60)
      R61 = tf.keras.layers.Reshape((-1, 1))
      s61 = R61(s61)
      R62 = tf.keras.layers.Reshape((-1, 1))
      s62 = R62(s62)
      R63 = tf.keras.layers.Reshape((-1, 1))
      s63 = R63(s63)
      R64 = tf.keras.layers.Reshape((-1, 1))
      s64 = R64(s64)
      R65 = tf.keras.layers.Reshape((-1, 1))
      s65 = R65(s65)
      R66 = tf.keras.layers.Reshape((-1, 1))
      s66 = R66(s66)
      R67 = tf.keras.layers.Reshape((-1, 1))
      s67 = R67(s67)
      R68 = tf.keras.layers.Reshape((-1, 1))
      s68 = R68(s68)
      R69 = tf.keras.layers.Reshape((-1, 1))
      s69 = R69(s69)

      #######################################################################
      # T is the output nodes, this is the network structure
      # each s is dim [None, 20, 3]


      T1 = tf.concat([s0,s1,s2,s3,s4,s5,s26,s27,s28], axis=2)
      C1 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T1 = lrelu(C1(T1))

      T2 = tf.concat([s26,s27,s28,s29,s30,s31], axis=2)
      C2 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T2 = lrelu(C2(T2))

      T3 = tf.concat([s29,s30,s31,s32,s33,s34], axis=2)
      C3 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T3 = lrelu(C3(T3))

      T4 = tf.concat([s32,s33,s34,s35,s36,s37], axis=2)
      C4 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T4 = lrelu(C4(T4))

      T5 = tf.concat([s35,s36,s37,s38,s39,s40], axis=2)
      C5 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T5 = lrelu(C5(T5))

      T6 = tf.concat([s38,s39,s40,s41,s42,s43], axis=2)
      C6 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T6 = lrelu(C6(T6))

      T7 = tf.concat([s6,s7,s8,s9,s10], axis=2)
      C7 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T7 = lrelu(C7(T7))

      T8 = tf.concat([s9,s10,s11,s12,s13], axis=2)
      C8 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T8 = lrelu(C8(T8))

      T9 = tf.concat([s11,s12,s13,s14,s15], axis=2)
      C9 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T9 = lrelu(C9(T9))

      T10 = tf.concat([s16,s17,s18,s19,s20], axis=2)
      C10 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T10 = lrelu(C10(T10))

      T11 = tf.concat([s19,s20,s21,s22,s23], axis=2)
      C11 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T11 = lrelu(C11(T11))

      T12 = tf.concat([s21,s22,s23,s24,s25], axis=2)
      C12 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T12 = lrelu(C12(T12))

      T13 = tf.concat([s44,s45,s46,s47,s48], axis=2)
      C13 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T13 = lrelu(C13(T13))

      T14 = tf.concat([s47,s48,s49], axis=2)
      C14 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T14 = lrelu(C14(T14))

      T15 = tf.concat([s49,s50,s51,s52], axis=2)
      C15 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T15 = lrelu(C15(T15))

      T16 = tf.concat([s50,s51,s52,s53], axis=2)
      C16 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T16 = lrelu(C16(T16))

      T17 = tf.concat([s49,s54,s55,s56], axis=2)
      C17 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T17 = lrelu(C17(T17))

      T18 = tf.concat([s57,s58,s59,s60,s61], axis=2)
      C18 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T18 = lrelu(C18(T18))

      T19 = tf.concat([s60,s61,s62], axis=2)
      C19 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T19 = lrelu(C19(T19))

      T20 = tf.concat([s62,s63,s64,s65], axis=2)
      C20 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T20 = lrelu(C20(T20))

      T21 = tf.concat([s63,s64,s65,s66], axis=2)
      C21 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T21 = lrelu(C21(T21))

      T22 = tf.concat([s62,s67,s68,s69], axis=2)
      C22 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                  padding='same', activation=None,
                                  kernel_regularizer=tcl.l2_regularizer(self.re_term))
      T22 = lrelu(C22(T22))



      result = tf.concat([T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22], axis=2)

      return result

#########################################################################################################
## This layer generate the parts of legs and arms and body
class CMUHLayer2(tf.keras.layers.Layer):
  def __init__(self, re_term, node, kernel_size=2):
    super(CMUHLayer2, self).__init__()
    self.re_term = re_term
    self.node = node
    self.kernel_size = kernel_size

  def call(self, input):

      lrelu = VAE.lrelu
      re_term = self.re_term
      T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17, T18, T19, T20, T21, T22\
          = tf.split(input, num_or_size_splits=22, axis=2)

      #######################################################################
      # T is the output nodes, this is the network structure
      TT1 = tf.concat([T1, T2, T3, T4, T5, T6], axis=2)# right leg
      CC1 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                   padding='same', activation=None,
                                   kernel_regularizer=tcl.l2_regularizer(self.re_term))
      TT1 = lrelu(CC1(TT1))

      TT2 = tf.concat([T7, T8, T9], axis=2)# body
      CC2 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                   padding='same', activation=None,
                                   kernel_regularizer=tcl.l2_regularizer(self.re_term))
      TT2 = lrelu(CC2(TT2))

      TT3 = tf.concat([T10, T11, T12], axis=2)# Right arm
      CC3 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                   padding='same', activation=None,
                                   kernel_regularizer=tcl.l2_regularizer(self.re_term))
      TT3 = lrelu(CC3(TT3))

      TT4 = tf.concat([T13, T14, T15, T16, T17], axis=2) # left arm
      CC4 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                   padding='same', activation=None,
                                   kernel_regularizer=tcl.l2_regularizer(self.re_term))
      TT4 = lrelu(CC4(TT4))

      TT5 = tf.concat([T18, T19, T20, T21, T22], axis=2)  # left arm
      CC5 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                   padding='same', activation=None,
                                   kernel_regularizer=tcl.l2_regularizer(self.re_term))
      TT5 = lrelu(CC5(TT5))

      result = tf.concat([TT1, TT2, TT3, TT4, TT5], axis=2)

      return result

#########################################################################################################
## This layer generate the parts of half and half
class CMUHLayer3(tf.keras.layers.Layer):
  def __init__(self, re_term, node, kernel_size=2):
    super(CMUHLayer3, self).__init__()
    self.re_term = re_term
    self.node = node
    self.kernel_size = kernel_size

  def call(self, input):

      lrelu = VAE.lrelu
      re_term = self.re_term
      TT1, TT2, TT3, TT4, TT5\
          = tf.split(input, num_or_size_splits=5, axis=2)

      #######################################################################
      # T is the output nodes, this is the network structure
      TTT1 = tf.concat([TT1, TT2, TT3], axis=2)# Left leg
      CCC1 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                    padding='same', activation=None,
                                    kernel_regularizer=tcl.l2_regularizer(self.re_term))
      TTT1 = lrelu(CCC1(TTT1))

      TTT2 = tf.concat([TT4, TT5, TT1], axis=2)# right leg
      CCC2 = tf.keras.layers.Conv1D(filters=self.node, kernel_size=self.kernel_size, strides=1,
                                    padding='same', activation=None,
                                    kernel_regularizer=tcl.l2_regularizer(self.re_term))
      TTT2 = lrelu(CCC2(TTT2))







      result = tf.concat([TTT1, TTT2], axis=2)

      return result