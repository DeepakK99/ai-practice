# KV Cache
if we have trained on:
```text
The cat sat
```
then we already have:
```text
K_cache:
K(The)
K(cat)
K(sat)

V_cache:
V(The)
V(cat)
V(sat)
```
now processing the next token "on", will use the previously calculated Ks and Vs. 