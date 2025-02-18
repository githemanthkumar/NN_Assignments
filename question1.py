import tensorflow as tf

# 1. Create a random tensor of shape (4, 6)
tensor = tf.random.uniform(shape=(4, 6), minval=0, maxval=10, dtype=tf.float32)
print("Original Tensor:\n", tensor.numpy())

# 2. Find its rank and shape
rank = tf.rank(tensor)
shape = tf.shape(tensor)
print("\nRank of tensor:", rank.numpy())
print("Shape of tensor:", shape.numpy())

# 3. Reshape into (2, 3, 4)
reshaped_tensor = tf.reshape(tensor, (2, 3, 4))
print("\nReshaped Tensor (2, 3, 4):\n", reshaped_tensor.numpy())

# 4. Transpose into (3, 2, 4)
transposed_tensor = tf.transpose(reshaped_tensor, perm=[1, 0, 2])
print("\nTransposed Tensor (3, 2, 4):\n", transposed_tensor.numpy())

# 5. Broadcasting a smaller tensor (1, 6) to match larger tensor
small_tensor = tf.constant([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])  # Shape (1,6)
broadcasted_tensor = tf.broadcast_to(small_tensor, shape=(4, 6))  # Shape (4,6)
print("\nBroadcasted Tensor (4, 6):\n", broadcasted_tensor.numpy())

# 6. Adding the tensors
added_tensor = tensor + broadcasted_tensor
print("\nTensor After Broadcasting and Addition:\n", added_tensor.numpy())

# 7. Explanation of Broadcasting
explanation = """
Broadcasting in TensorFlow automatically expands the smaller tensor's dimensions 
to match the larger tensor's shape, allowing element-wise operations.
Here, (1,6) expands to (4,6) by repeating its values across rows.
"""
print(explanation)
