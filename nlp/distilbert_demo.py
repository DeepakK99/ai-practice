from transformers import pipeline

# ---------------------------------------------------------
# 1. TEXT CLASSIFICATION
# ---------------------------------------------------------

classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

while True:
    text = input("comment: ")
    if text == "\q":
        break 
    print("class: ", classifier(text)[0]["label"])


exit(0)
text = "The customer support was excellent and very helpful."

result = classifier(text)

print("\nTEXT CLASSIFICATION")
print(result)


# ---------------------------------------------------------
# 2. NAMED ENTITY RECOGNITION
# ---------------------------------------------------------

ner = pipeline(
    "ner",
    model="elastic/distilbert-base-uncased-finetuned-conll03-english",
    aggregation_strategy="simple",
)

text = (
    "John works at Amazon in Bangalore "
    "and previously lived in Delhi."
)

result = ner(text)

print("\nNAMED ENTITY RECOGNITION")

for entity in result:
    print(entity)


# ---------------------------------------------------------
# 3. QUESTION ANSWERING
# ---------------------------------------------------------

qa = pipeline("question-answering", model='distilbert-base-uncased-distilled-squad')

context = """
Amazon is headquartered in Seattle, Washington.
The company was founded by Jeff Bezos in 1994.
"""

question = "Where is Amazon headquartered?"

result = qa(
    question=question,
    context=context,
)

print("\nQUESTION ANSWERING")
print(result)