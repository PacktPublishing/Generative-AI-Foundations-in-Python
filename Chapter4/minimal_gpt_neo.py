# In a Colab or Jupyter notebook
# !pip install transformers

from transformers import pipeline

# Initialize a text generation pipeline with a generative
text_generator = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

# Example prompt for product description generation
prompt = "This high-tech running shoe with advanced cushioning and support"

# Generating the product description
generated_text = text_generator(prompt, max_length=100, do_sample=True)

# Printing the generated product description
print(generated_text[0]["generated_text"])
