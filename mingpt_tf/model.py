"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
import pdb

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout, Embedding
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import count_params
import tensorflow_addons as tfa
from tensorflow_addons.optimizers import AdamW
from tensorflow_addons.layers import GELU

logger = logging.getLogger(__name__)

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class CausalSelfAttention(K.layers.Layer):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = Dense(config.n_embd, name='key', kernel_initializer='he_normal', bias_initializer='zeros')
        self.query = Dense(config.n_embd, name='query', kernel_initializer='he_normal', bias_initializer='zeros')
        self.value = Dense(config.n_embd, name='value', kernel_initializer='he_normal', bias_initializer='zeros')
        # regularization
        self.attn_drop = Dropout(config.attn_pdrop)
        self.resid_drop = Dropout(config.resid_pdrop)
        # output projection
        self.proj = Dense(config.n_embd, name='projection', kernel_initializer='he_normal', bias_initializer='zeros')
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.mask = 1 - tf.linalg.band_part(tf.ones((config.block_size, config.block_size)), -1, 0)#[tf.newaxis, tf.newaxis] # tf tril equivalent
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.depth = config.n_embd // config.n_head

    def split_heads(self, x, batch_size):
        """ Split the last dimension into (n_head, depth) """
        x = tf.reshape(x, (batch_size, -1, self.n_head, self.depth))    # (B, T, ne) -> (B, T, nh, d)
        return tf.transpose(x, perm=[0, 2, 1, 3])    # (B, nh, T, d)
    
    def reassemble_heads(self, x, batch_size):
        """ Concat the last two dimensions into (n_embd,) - inverse transform of `split_heads()` """
        x = tf.transpose(x, perm=[0, 2, 1, 3])    # (B, nh, T, d)
        return tf.reshape(x, (batch_size, -1, self.n_embd))    # (B, T, ne)
    
    def scaled_dot_product_attention(self, q, k, v, mask, training=True):
        """ Calculate the attention weights v(q*k) 
        q*k gives relevance measure - better matched a k is to q, the higher it will activate and thus will feature more in the v output
        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)    # (..., T, T)

        # scale matmul_qk
        dk = tf.cast(k.shape[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)    

        # softmax is normalized on the last axis (T) so that the scores
        # add up to 1.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)    # (..., T, T)
        attention_weights = self.attn_drop(attention_weights, training=training)

        # (B, nh, T, T) x (B, nh, T, d) -> (B, nh, T, d)
        output = tf.matmul(attention_weights, v)    # (..., T, d)

        return output, attention_weights
        
    def call(self, x, training=True):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # n_embd = n_heads * depth
        k = self.key(x)    # (B, T, ne)
        q = self.query(x)    # (B, T, ne)
        v = self.value(x)    # (B, T, ne)
        
        k = self.split_heads(k, B)    # (B, nh, T, d)
        q = self.split_heads(q, B)    # (B, nh, T, d)
        v = self.split_heads(v, B)    # (B, nh, T, d)

        # causal self-attention; Self-attend: (B, nh, T, d) x (B, nh, d, T)
        y, w = self.scaled_dot_product_attention(q, k, v, self.mask[:T,:T], training=training)    # (B, nh, T, d), (B, nh, T, T)
        y = self.reassemble_heads(y, B) # re-assemble all head outputs side by side (B, T, ne)

        # output projection
        y = self.resid_drop(self.proj(y), training=training)    # (B, T, ne)
        return y

class Block(K.layers.Layer):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()
        self.attn = CausalSelfAttention(config)
        self.mlp = K.Sequential([
            Dense(4 * config.n_embd, kernel_initializer='he_normal', bias_initializer='zeros'),
            GELU(),
            Dense(config.n_embd, kernel_initializer='he_normal', bias_initializer='zeros'),
            Dropout(config.resid_pdrop),
        ])

    def call(self, x, training=True):
        x = x + self.attn(self.ln1(x), training=training)
        x = x + self.mlp(self.ln2(x), training=training)
        return x

class GPT(K.Model):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, train_config):
        super().__init__()

        # input embedding stem
        self.tok_emb = Embedding(config.vocab_size, config.n_embd, embeddings_initializer='he_normal')
#         self.pos_emb = tf.Variable(tf.zeros(1, config.block_size, config.n_embd)) # learned pos embedding
        self.pos_emb = self.add_weight("position_embedding",
                                       shape=(1, config.block_size, config.n_embd),
                                       initializer='zeros',
                                       dtype=tf.float32)
        self.drop = Dropout(config.embd_pdrop)
        # transformer
        self.blocks = K.Sequential([Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = LayerNormalization()
        self.head = Dense(config.vocab_size, use_bias=False, kernel_initializer='he_normal')

        self.block_size = config.block_size
        # self.apply(self._init_weights)
        self.loss_fn = K.losses.CategoricalCrossentropy(from_logits=True)#K.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.optimizer = Adam()#self.configure_optimizers(train_config)
        self.compile(optimizer=self.optimizer,
                      loss=self.loss_fn,
                      metrics=['accuracy'])
#         self.build(input_shape=(None, config.block_size))
#         logger.info("number of parameters: %e", np.sum([K.count_params(w) for w in self.trainable_weights]))
        
    def get_block_size(self):
        return self.block_size

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # Todo: We weight decay all params whereas Karpathy excludes biases, LN, embeddings
        optimizer = AdamW(weight_decay=train_config.weight_decay, lr=train_config.learning_rate,
                          beta_1=train_config.betas[0], beta_2=train_config.betas[1])
        return optimizer

#     @tf.function
    def call(self, inputs: tf.Tensor, training=True):
        B, T = inputs.shape
        assert T <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(inputs) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :T, :] # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings, training=training)
        x = self.blocks(x, training=training)
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

#         # if we are given some desired targets also calculate the loss
#         loss = None
#         if targets is not None:
#             # loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
#             loss = self.loss_fn(logits, targets)

#         return logits, loss
