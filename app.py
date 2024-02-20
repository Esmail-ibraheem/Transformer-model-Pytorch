import torch 
import torch.nn as nn 
from torch.distributions.uniform import Uniform
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import streamlit as st

# Streamlit UI for user input
st.title('Word Embedding Visualization')

# Function to get user input
def get_user_input():
    tokens = []
    for i in range(4):
        token = st.text_input(f"Enter token {i+1}")
        tokens.append(token)
    return tokens

# Get user input
tokens = get_user_input()

# Define inputs and labels tensors
inputs = torch.tensor([[1, 0, 0, 0], 
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])

labels = torch.tensor([[0, 1, 0, 0], 
                       [0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 1, 0, 0]])

# Define the WordEmbeddingFromScratch model class
class WordEmbeddingFromScratch(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        min_value = -0.5 
        max_value = 0.5 

        self.input_w1 = nn.Parameter(Uniform(min_value, max_value).sample((4,)))
        self.input_w2 = nn.Parameter(Uniform(min_value, max_value).sample((4,)))
        self.output_w1 = nn.Parameter(Uniform(min_value, max_value).sample((4,)))
        self.output_w2 = nn.Parameter(Uniform(min_value, max_value).sample((4,)))

    def forward(self, input):
        input = input[0]

        inputs_to_top_hidden = torch.matmul(input, self.input_w1)
        inputs_to_bottom_hidden = torch.matmul(input, self.input_w2)
        
        output = (inputs_to_top_hidden[:, None] * self.output_w1) + \
                 (inputs_to_bottom_hidden[:, None] * self.output_w2)
        
        return output

# Create an instance of the model
model = WordEmbeddingFromScratch()

# Plot the embeddings
plt.figure(figsize=(8, 8))

for i, token in enumerate(tokens):
    plt.text(model.output_w1[i].item(), model.output_w2[i].item(), token, 
             horizontalalignment='left', 
             size='medium', 
             color='black',
             weight='semibold')

plt.scatter(model.output_w1.detach().numpy(), model.output_w2.detach().numpy())
plt.xlabel('w1')
plt.ylabel('w2')
plt.title('Word Embeddings')
st.pyplot()

# Display the model's output for the entered tokens
with torch.no_grad():
    outputs = model(inputs)
    print(torch.softmax(outputs, dim=1))
