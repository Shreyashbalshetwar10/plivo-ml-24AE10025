The final configuration has a simplified byte-level BPE as the tokenizer, with train_tokenizer.py training it on the given dataset.
The improved tokenizer increased the vocab size and decreased the number of tokens.
The model parameters are a 0.05 dropout and 176 n_embd, with initial lr = 1e-3 and cosine annealing applied on the last 600 iterations.
The changes above work due to the fact that the model was still improving down to the 2000th step, suggesting that the model had more space for growth, so the n_embd was increased and the dropout was decreased.
The Optimizer used is AdamW instead of Adam, because AdamW applies weight decay separately from the adaptive gradient updates, giving more effective regularization and making it the standard optimizer for transformer models.
The improved tokenizer increased the vocab size and decreased the number of tokens.
