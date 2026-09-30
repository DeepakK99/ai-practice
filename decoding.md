# Decoding
```text
        Logits
        ↓
        Temperature
        ↓
        Top-k / Top-p filtering
        ↓
        Softmax / normalized probabilities
        ↓
        Sampling
        ↓
        Next token
```
## Temperature
Temperature changes the distribution before sampling.

$$ P=softmax(z/T) $$
where, z = logits
```text
T = 1 -> normal distribution (rolls the dice)
T < 1 -> more sharp distriution -> more deterministic
T > 1 -> more flatter distribution -> more randomness
T = 0 -> skips "rolling" the dice -> completely deterministc
```

## Top k
consider top k highest-probability tokens.

## Top p
Keep the smallest set of tokens whose cumulative probability reaches p. p in included, so the word that crosses p is included in the sample pool.

| Method          | What it does                                            |
| --------------- | ------------------------------------------------------- |
| **Greedy**      | Always choose highest-probability token                 |
| **Temperature** | Changes how sharp/flat the distribution is              |
| **Top-k**       | Keep exactly the best K candidates                      |
| **Top-p**       | Keep enough candidates to cover probability mass P      |
| **Sampling**    | Randomly select according to the resulting distribution |
