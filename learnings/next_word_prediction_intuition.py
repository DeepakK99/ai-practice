# basic idea behind next word prediction
import random

sentences = [
    "i like cats",
    "i like dogs",
    "i love cats",
    "i love dogs",
    "cats are cute",
    "dogs are cute",
]

# gen freq
next_word_dict = {}

for sentence in sentences:
    words = sentence.split(" ")
    for i in range(len(words)):
        prev = words[i - 1]
        current = words[i]
        if i > 0:
            prev_dict = next_word_dict.get(prev, {})
            prev_dict[current] = prev_dict.get(current, 0) + 1
            next_word_dict[prev] = prev_dict
        else:
            next_word_dict[prev] = next_word_dict.get(prev, {})

print("fre_map: ", next_word_dict)

# gen prob
probabilitis = {} # use conditional prob for better results

for d_ in next_word_dict:
    total = sum(next_word_dict[d_].values())
    word_dict = next_word_dict[d_]
    for word in word_dict:
        word_prob = probabilitis.get(d_, {})
        word_prob[word] = round( word_dict[word] / total, 2)
        probabilitis[d_] = word_prob

print("Probabilities: ", probabilitis)

# predict next token
n = 10
initial = "i"
result = [initial]
for i in range(n - 1):
    max_prob = 0
    candidate = []
    for word in probabilitis.get(result[-1], []):
        if probabilitis[result[-1]][word] >= max_prob:
            max_prob = probabilitis[result[-1]][word]
            candidate.append(word)
    if candidate:
        result.append(random.choice(candidate))

print("Generated Output: ", " ".join(result))
        