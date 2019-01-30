from matplotlib import pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import random
import time
import cntk

import constants

def inititalize_cntk():
    print('cntk Version = ',cntk.__version__)
    cntk.device.try_set_default_device(cntk.device.cpu())
    np.random.seed(0)
    cntk.cntk_py.set_fixed_random_seed(1)
    cntk.cntk_py.force_deterministic_algorithms()

def create_model():
    with cntk.layers.default_options(initial_state=0.1):
        return cntk.layers.Sequential([
            cntk.layers.Embedding(constants.emb_dim, name='embed'),
            cntk.layers.Recurrence(cntk.layers.LSTM(constants.hidden_dim), go_backwards=False),
            cntk.layers.Dense(constants.num_labels, name='classify')
        ])

def create_reader(path, is_training):
    return cntk.io.MinibatchSource(cntk.io.CTFDeserializer(path, cntk.io.StreamDefs(
         query         = cntk.io.StreamDef(field='S0', shape=constants.vocab_size,  is_sparse=True),
         intent_unused = cntk.io.StreamDef(field='S1', shape=constants.num_intents, is_sparse=True),  
         slot_labels   = cntk.io.StreamDef(field='S2', shape=constants.num_labels,  is_sparse=True)
     )), randomize=is_training, max_sweeps = cntk.io.INFINITELY_REPEAT if is_training else 1)

def create_criterion_function(model, labels):
    ce   = cntk.cross_entropy_with_softmax(model, labels)
    errs = cntk.classification_error      (model, labels)
    return ce, errs # (model, labels) -> (loss, error metric)

def train_test(train_reader, test_reader, model_func, max_epochs=10):
    x = cntk.sequence.input_variable(constants.vocab_size)
    y = cntk.sequence.input_variable(constants.num_labels)

    model = model_func(x)
    
    loss, label_error = create_criterion_function(model, y)

    lr_per_minibatch = [lr * constants.minibatch_size for lr in constants.lr_per_sample]
    lr_schedule = cntk.learning_rate_schedule(lr_per_minibatch, cntk.UnitType.minibatch, constants.epoch_size)
    
    # Momentum schedule
    momentum_as_time_constant = cntk.momentum_as_time_constant_schedule(700)
    
    learner = cntk.adam(parameters=model.parameters,
                     lr=lr_schedule,
                     momentum=momentum_as_time_constant,
                     gradient_clipping_threshold_per_sample=15, 
                     gradient_clipping_with_truncation=True)

    progress_printer = cntk.logging.ProgressPrinter(tag='Training', num_epochs=max_epochs)
    
    # Uncomment below for more detailed logging
    #progress_printer = ProgressPrinter(freq=100, first=10, tag='Training', num_epochs=max_epochs) 

    trainer = cntk.Trainer(model, (loss, label_error), learner, progress_printer)

    cntk.logging.log_number_of_parameters(model)

    t = 0
    for epoch in range(max_epochs):         # loop over epochs
        epoch_end = (epoch+1) * constants.epoch_size
        while t < epoch_end:                # loop over minibatches on the epoch
            data = train_reader.next_minibatch(constants.minibatch_size, input_map={  # fetch minibatch
                x: train_reader.streams.query,
                y: train_reader.streams.slot_labels
            })
            trainer.train_minibatch(data)               # update model with it
            t += data[y].num_samples                    # samples so far
        trainer.summarize_training_progress()
    
    while True:
        minibatch_size = 500
        data = test_reader.next_minibatch(minibatch_size, input_map={  # fetch minibatch
            x: test_reader.streams.query,
            y: test_reader.streams.slot_labels
        })
        if not data:                                 # until we hit the end
            break
        trainer.test_minibatch(data)
    
    trainer.summarize_test_progress()

def do_train_test():
    z = create_model()
    train_reader = create_reader('atis.train.ctf', is_training=True)
    test_reader = create_reader('atis.test.ctf',is_training=False)
    train_test(train_reader, test_reader, z)

if __name__ == '__main__':
    inititalize_cntk()
    do_train_test()