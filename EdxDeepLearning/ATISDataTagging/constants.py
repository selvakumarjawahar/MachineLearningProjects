vocab_size = 943
num_labels = 129
num_intents = 26

input_dim = vocab_size
label_dim = num_labels
emb_dim = 150
hidden_dim = 300 

epoch_size = 18000        # 18000 samples is half the dataset size 
minibatch_size = 70
lr_per_sample = [0.003]*4+[0.0015]*24+[0.0003]

