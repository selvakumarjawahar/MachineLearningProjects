from matplotlib import pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import random
import time
import cntk

import constants
import loaddata

def create_model(features):
    """Create the model for time series prediction"""
    with cntk.layers.default_options(initial_state = 0.1):
        m = cntk.layers.Recurrence(cntk.layers.LSTM(constants.H_DIMS))(features)
        m = cntk.sequence.last(m)
        m = cntk.layers.Dropout(constants.DROPOUT)(m)
        m = cntk.layers.Dense(1)(m)
        return m

def create_trainer(features,labels,model):

    # the learning rate
    lr_schedule = cntk.learning_rate_schedule(constants.LEARNING_RATE, cntk.UnitType.minibatch)

    # loss and error function
    loss = cntk.squared_error(model, labels)
    error = cntk.squared_error(model, labels)

    # use adam optimizer
    momentum_time_constant = cntk.momentum_as_time_constant_schedule(constants.BATCH_SIZE / -math.log(0.9)) 
    learner = cntk.fsadagrad(model.parameters,lr = lr_schedule,momentum = momentum_time_constant)
    trainer = cntk.Trainer(model, (loss, error), [learner])
    return trainer

def train_test():
    loss_summary = []
    features = cntk.sequence.input_variable(1)
    model = create_model(features)
    labels = cntk.input_variable(1, dynamic_axes=model.dynamic_axes, name="y")
    trainer = create_trainer(features,labels,model)
    X,Y = loaddata.generate_solar_data('solar.csv',constants.TIMESTEPS,constants.NORMALIZE)
    start = time.time()
    for epoch in range(0, constants.EPOCHS):
        for x_batch, l_batch in loaddata.next_batch(X, Y, "train"):
            trainer.train_minibatch({features: x_batch, labels: l_batch})
            if epoch % (constants.EPOCHS / 10) == 0:
                training_loss = trainer.previous_minibatch_loss_average
                loss_summary.append(training_loss)
                print("epoch: {}, loss: {:.4f}".format(epoch, training_loss))
    print("Training took {:.1f} sec".format(time.time() - start))
    plt.plot(loss_summary,label='Training_Loss')
    plt.show()

if __name__ == '__main__':
    train_test()

