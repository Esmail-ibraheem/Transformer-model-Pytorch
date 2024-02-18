# Transformer-model
I built the Transformer model itself from scratch from the paper "Attention is all you need", Feel free to use this model for your specific purposes: translation, text generation, etc...

---

##### Embedding Visualizer:
built the word embedding from scratch, but used it for a small dataset I created, or actually just some tokens: 
```
         w1        w2   token   input
0  0.393158 -0.423314      My  input1
1 -0.324287 -0.350400    name  input2
2  0.094321  0.121907      is  input3
3 -0.489201  0.398983  Esmail  input4
```
![image](https://github.com/Esmail-ibraheem/Transformer-model/assets/113830751/ab7f13cb-6baa-45f0-b8b0-74c7174dc0a1)

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

##### Positional Encoding Visualizer:

developing process: 
https://youtu.be/uWchpx4J6MY?si=iPYmauKz1MoQV9bh
