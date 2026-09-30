Q = X * w_Q
K = X * w_K
V = X * w_V

Where w_Q, w_K and w_V are pretrained "lenses" and X is the current vectr.

Deep Dive: The Mathematics of Self-Attention (Q, K, V)

This guide provides a step-by-step breakdown of how the Self-Attention mechanism works under the hood in Transformers, using a concrete numerical example.
⸻
1. Matrix Multiplication vs. The Dot Product

The core of the confusion often lies in the notation $Q K^T$. While we call it matrix multiplication in code, matrix multiplication is conceptually just a highly efficient way to compute multiple individual dot products simultaneously.

Every single cell in the resulting matrix is calculated by taking a row from $Q$ and a column from $K^T$, multiplying their elements position by position, and adding them up. This is the exact definition of a dot product.
⸻
2. Walkthrough Example with Exact Values

Let's assume we have a tiny 2-token sentence processed by a model with an embedding dimension ($d$) of 3.

The Input Matrices
We have our Query matrix ($Q$) and our Key matrix transposed ($K^T$):

$$Q = \begin{bmatrix} 1 & 2 & 3 \ 4 & 5 & 6 \end{bmatrix}$$

$$K^T = \begin{bmatrix} 7 & 1 \ 8 & 2 \ 9 & 3 \end{bmatrix}$$

Step-by-Step Grid Calculation
When we perform the operation $Q \times K^T$, the computer calculates the values for all 4 cells by running 4 independent dot products:

* Cell (Row 1, Col 1): Dot product of $Q_1$ and $K^T_1$ 
* $$\text{Value} = (1 \times 7) + (2 \times 8) + (3 \times 9) = 7 + 16 + 27 = \mathbf{50}$$

* Cell (Row 1, Col 2): Dot product of $Q_1$ and $K^T_2$ 
* $$\text{Value} = (1 \times 1) + (2 \times 2) + (3 \times 3) = 1 + 4 + 9 = \mathbf{14}$$

* Cell (Row 2, Col 1): Dot product of $Q_2$ and $K^T_1$ 
* $$\text{Value} = (4 \times 7) + (5 \times 8) + (6 \times 9) = 28 + 40 + 54 = \mathbf{122}$$

* Cell (Row 2, Col 2): Dot product of $Q_2$ and $K^T_2$ 
* $$\text{Value} = (4 \times 1) + (5 \times 2) + (6 \times 3) = 4 + 10 + 18 = \mathbf{32}$$

The Resulting Raw Attention Matrix
Combining all these individual dot products gives us our raw attention grid:

$$Q K^T = \begin{bmatrix} 50 & 14 \ 122 & 32 \end{bmatrix}$$
⸻
3. Why the Dimension Size ($d$) Causes Values to Explode

Look closely at the addition step inside just one single cell: 
$$\text{Cell Value} = (Q_1 \times K_1) + (Q_2 \times K_2) + (Q_3 \times K_3)$$

* When our dimension $d = 3$, the computer only adds up 3 multiplication pairs. The resulting cell totals stay relatively small (like $50$ or $122$).
* When using a production model like Nomic or BERT, the dimension size scales up to $d = 768$. The grid dimensions remain a small $2 \times 2$ for a 2-token sentence, but the formula for every individual cell becomes a massive sum of 768 multiplication pairs!

$$\text{Cell Value} = (Q_1 \times K_1) + (Q_2 \times K_2) + \dots + (Q_{768} \times K_{768})$$

Even if the individual numbers are tiny decimals, forcing the computer to sum 768 numbers together causes the values inside the matrix cells to naturally accumulate into much larger magnitudes (e.g., values swelling up to $30$ or $40$).
⸻
4. The Softmax Problem and The Scaling Factor ($\sqrt{d}$)

Why do large values break the model? Because the next step in the formula applies Softmax across each row.

Softmax works using exponential growth ($e^x$). If we pass unscaled, massive numbers into it, the largest number in the row will completely obliterate everything else:
* $e^{14} \approx 1.2 \times 10^6$
* $e^{50} \approx 5.1 \times 10^{21}$

The row probability distribution becomes a stark, unhelpful [1.0, 0.0]. The token pays 100% attention to one word and completely blinds itself to the rest of the sentence. During training, this flattens the gradients to zero, causing the model to stop learning entirely (vanishing gradient problem).

The Mathematical Rescue
To stop this variance explosion, we divide the entire matrix by the square root of the dimension size ($\sqrt{d}$).

In our example where $d = 3$, we scale by $\sqrt{3} \approx 1.732$:

$$\frac{Q K^T}{\sqrt{d}} = \begin{bmatrix} 50/1.732 & 14/1.732 \ 122/1.732 & 32/1.732 \end{bmatrix} \approx \begin{bmatrix} 28.86 & 8.08 \ 70.43 & 18.47 \end{bmatrix}$$

By implementing this scaling factor, we neutralize the compounding addition loop, stabilize the numbers, and ensure Softmax can distribute contextual attention smoothly!