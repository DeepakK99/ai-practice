# Autoregressive decoder-only LLM
```text
                  Text
                    ↓
                    Tokenizer
                    ↓
                    Token IDs
                    ↓
                    Embeddings
                    ↓
                    Transformer
                    │
                    ├─ Self-attention
                    │    ├─ Q
                    │    ├─ K
                    │    ├─ V
                    │    ├─ QKᵀ
                    │    ├─ scaling
                    │    ├─ causal mask
                    │    ├─ softmax
                    │    └─ weighted V
                    │
                    ├─ Multi-head attention
                    │
                    ├─ Residual
                    │
                    ├─ FFN
                    │
                    └─ Normalization
                    ↓
                    Final hidden state
                    ↓
                    LM Head
                    ↓
                    Logits
                    ↓
                    Softmax
                    ↓
                    Decoding
                    ↓
                    Next token
                    ↓
                    KV cache → repeat
```