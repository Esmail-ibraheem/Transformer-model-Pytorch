import numpy as np
import matplotlib.pyplot as plt

def self_attention(input_sequence, tokens):
    # Assuming input_sequence is a 2D array where each row represents a token/vector
    
    # Step 1: Compute similarity scores
    similarity_matrix = np.dot(input_sequence, input_sequence.T)  # Dot product to compute similarities
    
    # Step 2: Compute attention weights
    attention_weights = softmax(similarity_matrix, axis=1)  # Applying softmax along rows
    
    # Step 3: Compute context vectors
    context_vectors = np.dot(attention_weights, input_sequence)
    
    return context_vectors, attention_weights

def softmax(x, axis=None):
    # Numerically stable softmax implementation
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def plot_attention(input_sequence, attention_weights, tokens):
    # Plotting the attention weights
    plt.figure(figsize=(8, 6))
    plt.imshow(attention_weights, interpolation='nearest', cmap='Blues')
    plt.xlabel('Input Tokens')
    plt.ylabel('Output Tokens')
    plt.title('Self-Attention Mechanism')
    plt.colorbar()
    plt.xticks(np.arange(len(tokens)), tokens, rotation=45)
    plt.yticks(np.arange(len(tokens)), tokens)
    plt.show()

# Example usage
tokens = ['Esmail', 'went', 'to', 'the', 'University']
input_sequence = np.array([[1, 2, 3, 4, 5],
                           [4, 5, 6, 7, 8],
                           [7, 8, 9, 10, 11],
                           [10, 11, 12, 13, 14],
                           [13, 14, 15, 16, 17]])

context_vectors, attention_weights = self_attention(input_sequence, tokens)
print("Context Vectors:")
print(context_vectors)
print("\nAttention Weights:")
print(attention_weights)

# Plotting attention weights
plot_attention(input_sequence, attention_weights, tokens)
