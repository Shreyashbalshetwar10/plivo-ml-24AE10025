baseline: 
{"bpb": 2.3718, 
"n_params": 1339840, 
"steps": 2000, 
"tokens_in_eval": 159225, 
"tokens_scored": 159224}

Modification 1: adding dropout and changing opt to AdamW
{"bpb": 2.6892, 
"n_params": 1339840, 
"steps": 2000, 
"tokens_in_eval": 159225, 
"tokens_scored": 159224}

Modification 2: changing dropout to 0.1
{"bpb": 2.4264, 
"n_params": 1339840, 
"steps": 2000, 
"tokens_in_eval": 159225, 
"tokens_scored": 159224}

Modification 3: tie_weights
{"bpb": 2.654, 
"n_params": 1298880, 
"steps": 2000, 
"tokens_in_eval": 159225, 
"tokens_scored": 159224}

Modification 4: no tie_weights, dropout 0.1 and LR = 1e-3
{"bpb": 2.3577, 
"n_params": 1339840, 
"steps": 2000, 
"tokens_in_eval": 159225, 
"tokens_scored": 159224}

Modification 5: making a BPE tokenizer
{"bpb": 2.1299, 
"n_params": 1659840, 
"steps": 2000, 
"tokens_in_eval": 52188, 
"tokens_scored": 52187}

Modification 6: adding cosine annealing
{"bpb": 2.105, 
"n_params": 1659840, 
"steps": 2000, 
"tokens_in_eval": 52188, 
"tokens_scored": 52187}

Modification 7: decreasing dropout and increasing n_embd
{"bpb": 2.087, 
"n_params": 1960992, 
"steps": 2000, 
"tokens_in_eval": 52188, 
"tokens_scored": 52187}
