"""
Attention mechanism modules for the transformer model.
"""

import tensorflow as tf
from typing import Optional


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention layer for transformer model.
    Allows the model to jointly attend to information from different representation subspaces.
    """
    
    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize the multi-head attention layer.
        
        Args:
            d_model: Dimensionality of the model
            num_heads: Number of attention heads
        """
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        
        # Check if dimensions are compatible with the number of heads
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.depth = d_model // num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
    
    def split_heads(self, x, batch_size: int):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result to shape (batch_size, num_heads, seq_len, depth).
        
        Args:
            x: Input tensor
            batch_size: Batch size
            
        Returns:
            Reshaped tensor
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, mask: Optional[tf.Tensor] = None):
        """
        Forward pass for multi-head attention.
        
        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            mask: Optional mask tensor
            
        Returns:
            Output tensor after multi-head attention
        """
        batch_size = tf.shape(q)[0]
        
        # Linear projections
        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)
        
        # Split heads
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # Scaled dot-product attention
        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask)
        
        # (batch_size, num_heads, seq_len_q, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        
        # Reshape to (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, 
                                     (batch_size, -1, self.d_model))
        
        # Final linear projection
        output = self.dense(concat_attention)
        
        return output
    
    def scaled_dot_product_attention(self, q, k, v, mask: Optional[tf.Tensor] = None):
        """
        Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
        
        Args:
            q: Query tensor with shape (..., seq_len_q, depth)
            k: Key tensor with shape (..., seq_len_k, depth)
            v: Value tensor with shape (..., seq_len_v, depth_v)
            mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k)
            
        Returns:
            output: Attention output
            attention_weights: Attention weights
        """
        # (..., seq_len_q, seq_len_k)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        
        # Scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        # Add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  
        
        # Softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        
        output = tf.matmul(attention_weights, v)
        
        return output, attention_weights


class FeedForward(tf.keras.layers.Layer):
    """
    Position-wise feed-forward network for transformer.
    Consists of two linear transformations with a ReLU activation in between.
    """
    
    def __init__(self, d_ff: int, d_model: int):
        """
        Initialize the feed-forward network.
        
        Args:
            d_ff: Dimensionality of the inner layer
            d_model: Dimensionality of the input and output
        """
        super(FeedForward, self).__init__()
        
        self.d_ff = d_ff
        self.d_model = d_model
        
        self.dense1 = tf.keras.layers.Dense(d_ff, activation='relu')
        self.dense2 = tf.keras.layers.Dense(d_model)
    
    def call(self, x):
        """
        Forward pass for the feed-forward network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after feed-forward transformation
        """
        x = self.dense1(x)
        x = self.dense2(x)
        return x


class LayerNormalization(tf.keras.layers.Layer):
    """
    Layer normalization for transformer model.
    Normalizes the inputs to have zero mean and unit variance.
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Initialize the layer normalization.
        
        Args:
            epsilon: Small constant for numerical stability
        """
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        """
        Build the layer weights when first called.
        
        Args:
            input_shape: Shape of the input tensor
        """
        self.gamma = self.add_weight(
            name='gamma',
            shape=input_shape[-1:],
            initializer='ones',
            trainable=True)
        
        self.beta = self.add_weight(
            name='beta',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        """
        Forward pass for layer normalization.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized output tensor
        """
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        normalized = (x - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta
