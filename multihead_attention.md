# Multihead attention
```text
                Input X
                   │
        ┌──────────┼──────────┐
        ↓          ↓          ↓
      Head 1     Head 2     Head 3
        ↓          ↓          ↓
     Attention  Attention  Attention
        ↓          ↓          ↓
        └──────────┼──────────┘
                   ↓
             Concatenate
                   ↓
                  Wₒ
                   ↓
          Attention output
```
The complete multi-head operation is roughly:

$$ \boxed{ MultiHead(X)=Concat(head_1,\dots,head_h)W_O } $$

where each:

$$ head_i = Attention(XW_Q^{(i)},XW_K^{(i)},XW_V^{(i)}) $$