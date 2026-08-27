from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-360M"
)

result = generator(
    "Hiiiiiiiii My name is Shruthi",
    max_new_tokens=50,         # Controls generated length past your prompt
    num_return_sequences=2,    # Generates 2 sequences
    do_sample=True,             # Required when num_return_sequences > 1
    top_k=50,
    temperature=0.7 
)

print(result)
