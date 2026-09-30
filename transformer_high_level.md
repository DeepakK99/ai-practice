# High level Transformer architecture
```text
                 X
                 │
                 ↓
            Normalization
                 │
                 ↓
       Multi-Head Attention (project dimensions to a few ones and run parallel calculations)
                 │
                 ↓
          Residual Add (X + attention(X))
                 │
                 ↓
            Normalization (pre/post attention norm)
                 │
                 ↓
              FFN/MLP (thinking and memory: project to higher dim and collapse)
                 │
                 ↓
          Residual Add
                 │
                 ↓
              Output
```