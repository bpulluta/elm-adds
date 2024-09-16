import tiktoken

# Attempt to use a common encoding name such as 'gpt2'
try:
    tokenizer = tiktoken.get_encoding('gpt2')  # Change 'gpt2' to the correct encoding if known
    text = "Your text here"
    tokens = tokenizer.encode(text)
    print(tokens)
except ValueError as e:
    print(f"Error: {e}")
