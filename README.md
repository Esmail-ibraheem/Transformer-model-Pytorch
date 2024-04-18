# Transformer-model
<p align="center">
  <img src="https://github.com/Esmail-ibraheem/Transformer-model/blob/main/transformer.jpg" alt="Your Image Description" width="400" height=400">
</p>

I built the Transformer model itself from scratch from the paper "Attention is all you need", Feel free to use this model for your specific purposes: translation, text generation, etc...

---

### Embedding Visualizer:
built the word embedding from scratch, but used it for a small dataset I created, or actually just some tokens: 

- The input sequence is transformed into fixed-dimensional embeddings, typically composed of word embeddings and positional encodings. Word embeddings capture the semantic meaning of each word. 
```
         w1        w2   token   input
0  0.393158 -0.423314      My  input1
1 -0.324287 -0.350400    name  input2
2  0.094321  0.121907      is  input3
3 -0.489201  0.398983  Esmail  input4
```
![Figure_1](https://github.com/Esmail-ibraheem/Transformer-model/assets/113830751/92f4a1ee-862a-4b4b-a35e-671ce13fd709)

```
Before optimization, the parameters are...
input1_w1 tensor(0.3932) 
input1_w2 tensor(-0.4233)
input2_w1 tensor(-0.3243)
input2_w2 tensor(-0.3504)
input3_w1 tensor(0.0943)
input3_w2 tensor(0.1219)
input4_w1 tensor(-0.4892)
input4_w2 tensor(0.3990)
output1_w1 tensor(-0.0322)
output1_w2 tensor(-0.2084)
output2_w1 tensor(0.1547)
output2_w2 tensor(-0.0618)
output3_w1 tensor(0.1991)
output3_w2 tensor(0.2568)
output4_w1 tensor(-0.4561)
output4_w2 tensor(0.0075)
```

---

### Positional Encoding Visualizer:
- **positional encodings** indicate the word's position in the sequence using the sin and cos waves.
```
[[ 0.          1.          0.         ...  1.          0.
   1.        ]
 [ 0.84147098  0.54030231  0.8317052  ...  0.99994627  0.01018134
   0.99994817]
 [ 0.90929743 -0.41614684  0.92355454 ...  0.99978509  0.02036163
   0.99979268]
 ...
 [-0.99388865 -0.11038724 -0.03392828 ...  0.67538747  0.72739724
   0.68621662]
 [-0.62988799  0.77668598 -0.85006394 ...  0.66770653  0.73434615
   0.67877517]
 [ 0.31322878  0.9496777  -0.91001245 ...  0.65995384  0.74121893
   0.67126336]]
```
![image](https://github.com/Esmail-ibraheem/Transformer-model/assets/113830751/5efdbf12-5470-40c9-bc48-6fa54677fdb4)

![image](https://github.com/Esmail-ibraheem/Transformer-model/assets/113830751/1eb0781c-bba1-4d51-abbf-31b33c5e3e21)

---

### Self-attention Visualization:
- The core of the Transformer model is the self-attention mechanism. It allows each word in the input sequence to attend to all other words, capturing their relevance and influence. Self-attention computes three vectors for each word: Query, Key, and Value.
 ![Pasted image 20231221101118](https://github.com/Esmail-ibraheem/Transformer-model/assets/113830751/84274d78-ab56-4c17-8f43-b415fabf1f90)

```
Context Vectors:
[[13. 14. 15. 16. 17.]
 [13. 14. 15. 16. 17.]
 [13. 14. 15. 16. 17.]
 [13. 14. 15. 16. 17.]
 [13. 14. 15. 16. 17.]]

Attention Weights:
[[6.71418429e-079 2.34555134e-059 8.19401262e-040 2.86251858e-020
  1.00000000e+000]
 [4.50802707e-157 5.50161108e-118 6.71418429e-079 8.19401262e-040
  1.00000000e+000]
 [3.02677245e-235 1.29043112e-176 5.50161108e-118 2.34555134e-059
  1.00000000e+000]
 [2.03223080e-313 3.02677245e-235 4.50802707e-157 6.71418429e-079
  1.00000000e+000]
 [0.00000000e+000 7.09945017e-294 3.69388307e-196 1.92194773e-098
  1.00000000e+000]]
```
![Figure_1](https://github.com/Esmail-ibraheem/Transformer-model/assets/113830751/0c9f105a-f375-4459-bec4-a96892859663)

---

[developing process](https://youtu.be/uWchpx4J6MY?si=iPYmauKz1MoQV9bh)
