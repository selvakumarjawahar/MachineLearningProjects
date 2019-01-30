minibatch_size = 64
num_of_samples_per_sweep = 60000
num_of_sweeps_to_train_with = 10
num_of_minibatches_to_train = (num_of_samples_per_sweep*num_of_sweeps_to_train_with)/minibatch_size
training_file = 'Train-28x28_cntk_text.txt'
training_progress_output_frequency = 500
input_dim = 784
num_output_classes = 10
bits_per_pixel = 255
learning_rate = 0.2

num_hidden_layers = 2
hidden_layers_dim = 400
input_dim_model = (1,28,28)

test_minibatch_size = 512
num_samples = 10000
num_minibatches_to_test = num_samples / test_minibatch_size
test_file = 'Test-28x28_cntk_text.txt'

