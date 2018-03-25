'''
write your own keras layers: https://keras.io/layers/writing-your-own-keras-layers/
'''

from keras.layers import Dense, Lambda, multiply
from keras.initializers import Constant
from keras import backend as K

#refer https://arxiv.org/pdf/1607.06450v1.pdf
#with help of https://github.com/ctmakro/canton/blob/master/canton/cans.py
class LayerNormDense(Dense):
    def __init__(self, *args, **kwargs):
        super(LayerNormDense, self).__init__(*args, **kwargs)
        
    def build(self, input_shape):
        self.g = self.add_weight(shape=(self.units,), initializer=Constant(1.),
                                        name='g', trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer=Constant(0.),
                                        name='b', trainable=True)
        
        super(LayerNormDense, self).build(input_shape)
        
    def layernorm(self, x):
        #TODO: axis to general case (for now, sufficient for one axis)
        var_ = K.var(x, axis=1, keepdims=True)
        mean_ = K.mean(x, axis=1, keepdims=True)
        var_ = K.maximum(var_, 1e-7)
        stddev = K.sqrt(var_)
        
        gain = self.g
        gain = Lambda(lambda gain: gain/stddev, output_shape=(self.units,))(gain)
        dev = Lambda(lambda x: x-mean_, output_shape=(self.units,))(x)
        normalized = multiply([gain,dev])
        normalized = K.bias_add(normalized,self.b)
        return normalized
        
    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        output = self.layernorm(output)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output
