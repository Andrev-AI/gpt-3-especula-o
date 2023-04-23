import tensorflow as tf
import numpy as np

n_layers = 96
n_heads = 16
hidden_size = 3072
seq_length = 2048
vocab_size = 50_000


input_ids = tf.placeholder(tf.int32, shape=[None, seq_length])


def multihead_attention(inputs, num_heads):
    query = tf.layers.dense(inputs, hidden_size)
    key = tf.layers.dense(inputs, hidden_size)
    value = tf.layers.dense(inputs, hidden_size)

    query = tf.transpose(tf.reshape(query, [-1, seq_length, num_heads, hidden_size // num_heads]), [0, 2, 1, 3])
    key = tf.transpose(tf.reshape(key, [-1, seq_length, num_heads, hidden_size // num_heads]), [0, 2, 1, 3])
    value = tf.transpose(tf.reshape(value, [-1, seq_length, num_heads, hidden_size // num_heads]), [0, 2, 1, 3])

    score = tf.matmul(query, key, transpose_b=True) / np.sqrt(hidden_size // num_heads)
    attention_weights = tf.nn.softmax(score, axis=-1)

    context = tf.matmul(attention_weights, value)
    context = tf.transpose(tf.reshape(context, [-1, seq_length, hidden_size]), [0, 2, 1])

    return context

def transformer_block(inputs):
    attention_output = multihead_attention(inputs, n_heads)
    add_norm1 = tf.layers.batch_normalization(inputs + attention_output, axis=-1)
    dense_output = tf.layers.dense(add_norm1, hidden_size, activation=tf.nn.relu)
    add_norm2 = tf.layers.batch_normalization(add_norm1 + dense_output, axis=-1)
    return add_norm2

input_embeddings = tf.Variable(tf.random_uniform([vocab_size, hidden_size]))
input_embeddings = tf.nn.embedding_lookup(input_embeddings, input_ids)

for i in range(n_layers):
    input_embeddings = transformer_block(input_embeddings)

logits = tf.layers.dense(input_embeddings, vocab_size)
