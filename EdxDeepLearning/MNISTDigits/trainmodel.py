import IPython.display as idply
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import cntk
import constants


def inititalize_cntk():
    print('cntk Version = ',cntk.__version__)
    cntk.device.try_set_default_device(cntk.device.cpu())
    np.random.seed(0)
    cntk.cntk_py.set_fixed_random_seed(1)
    cntk.cntk_py.force_deterministic_algorithms()

def create_reader(path, is_training, input_dim, num_label_classes):
    labelStream = cntk.io.StreamDef(field='labels', shape=num_label_classes, is_sparse=False)
    featureStream = cntk.io.StreamDef(field='features', shape=input_dim, is_sparse=False)
    deserailizer = cntk.io.CTFDeserializer(path, cntk.io.StreamDefs(labels = labelStream, features = featureStream))
    return cntk.io.MinibatchSource(deserailizer,randomize = is_training, max_sweeps = cntk.io.INFINITELY_REPEAT if is_training else 1)

def create_model(features):
    with cntk.layers.default_options(init = cntk.glorot_uniform()):
        r = cntk.layers.Dense(constants.num_output_classes, activation = None)(features)
        return r    

def create_model_multi(features):
    with cntk.layers.default_options(init=cntk.layers.glorot_uniform(),activation=cntk.ops.relu):
        h = features
        for _ in range(constants.num_hidden_layers):
            h = cntk.layers.Dense(constants.hidden_layers_dim)(h)
        r = cntk.layers.Dense(constants.num_output_classes,activation=None)(h)
        return r

def create_model_conv_pooling(features):
    with cntk.layers.default_options(activation=cntk.ops.relu):
        h = features
        h = cntk.layers.Convolution2D(filter_shape=(5,5),num_filters=8,strides=(1,1),pad=True,name='first_conv')(h)
        h = cntk.layers.AveragePooling(filter_shape = (5,5),strides=(2,2),name='first_pooling')(h)
        h = cntk.layers.Convolution2D(filter_shape = (5,5),num_filters=16,strides=(1,1),pad=True,name='second_conv')(h)
        h = cntk.layers.AveragePooling(filter_shape = (5,5),strides=(2,2),name='second_pooling')(h)
        r = cntk.layers.Dense(constants.num_output_classes,activation=None,name='dense_layer')(h)
        return r

def create_model_conv(features):
    with cntk.layers.default_options(activation=cntk.ops.relu):
        h = features
        h = cntk.layers.Convolution2D(filter_shape=(5,5),num_filters=8,strides=(2,2),pad=True,name='first_conv')(h)
        h = cntk.layers.Convolution2D(filter_shape = (5,5),num_filters=16,strides=(2,2),pad=True,name='second_conv')(h)
        r = cntk.layers.Dense(constants.num_output_classes,activation=None)(h)
        return r

def create_trainer(model,label):
    loss = cntk.cross_entropy_with_softmax(model,label)
    label_error = cntk.classification_error(model,label)
    lr_schedule = cntk.learning_rate_schedule(constants.learning_rate,cntk.UnitType.minibatch)
    learner = cntk.sgd(model.parameters,lr_schedule)
    trainer = cntk.Trainer(model,(loss,label_error),[learner])
    return trainer

def moving_average(a,w=5):
    if(len(a) < w):
        return a[:]
    return [val if idx < w else sum(a[(idx-w):idx])/w for idx, val in enumerate(a)]

def print_training_progress(trainer,mb,frequency,verbose=1):
    training_loss = 'NA'
    eval_error = 'NA'
    if mb%frequency == 0:
        training_loss = trainer.previous_minibatch_loss_average
        eval_error = trainer.previous_minibatch_evaluation_average
        if verbose:
            print('minibatch:{0},Loss:{1:.4f},Error:{2:.2f}%'.format(mb,training_loss,eval_error*100))
    return mb,training_loss,eval_error

def print_model_parameter_shape(z):
    print('First Convolution shape = ',z.first_conv.shape)
    print('First Pooling shape = ',z.first_pooling.shape)
    print('Second Convolution shape = ',z.second_conv.shape)
    print('second pooling shape = ',z.second_pooling.shape)
    print('dense layer shape',z.dense_layer.shape)

def do_training():
    reader_train = create_reader(constants.training_file,True,constants.input_dim,constants.num_output_classes)

    #inp = cntk.input_variable(constants.input_dim)
    inp = cntk.input_variable(constants.input_dim_model)
    label = cntk.input_variable(constants.num_output_classes)

    input_map = {
        label: reader_train.streams.labels,
        inp: reader_train.streams.features
    }

    plot_data = {'batch_size':[],'loss':[],'error':[]}
    inp_s = inp/constants.bits_per_pixel
    #inp_sq = cntk.square(inp_s)
    inp_sqrt = cntk.sqrt(inp_s)
    #inp_combo = cntk.splice(inp_s,inp_sq,inp_sqrt)


    #z = create_model_multi(inp_s)
    z = create_model_conv_pooling(inp_sqrt)
    print_model_parameter_shape(z)
    trainer = create_trainer(z,label)


    for i in range(0, int(constants.num_of_minibatches_to_train)):
        data = reader_train.next_minibatch(constants.minibatch_size,input_map=input_map)
        #print(data)
        trainer.train_minibatch(data)
        batch_size,loss,error = print_training_progress(trainer,i,constants.training_progress_output_frequency,1)
        if not (loss == 'NA' or error == 'NA'):
            plot_data['batch_size'].append(batch_size)
            plot_data['loss'].append(loss)
            plot_data['error'].append(error)

    reader_test = create_reader(constants.test_file,False,constants.input_dim,constants.num_output_classes)
    test_input_map = {
        label : reader_test.streams.labels,
        inp : reader_test.streams.features
    }
    inp = cntk.sqrt(inp/225)

    test_result = 0.0
    for i in range(0,int(constants.num_minibatches_to_test)):
        data = reader_test.next_minibatch(constants.test_minibatch_size,input_map = test_input_map)
        eval_error = trainer.test_minibatch(data)
        test_result = test_result + eval_error
    err_percent = test_result*100 / constants.num_minibatches_to_test

    return trainer,plot_data,err_percent

def show_plot(plot_data):
    plot_data['avgloss'] = moving_average(plot_data['loss'])
    plot_data['avgerror'] = moving_average(plot_data['error'])
    
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plot_data['batch_size'],plot_data['avgloss'],'b--')
    plt.xlabel('Minibatch Number')
    plt.ylabel('Loss')
    plt.title('Minibatch run Vs Training Loss')

        
    plt.subplot(212)
    plt.plot(plot_data["batch_size"], plot_data["avgerror"], 'r--')
    plt.xlabel('Minibatch number')
    plt.ylabel('Label Prediction Error')
    plt.title('Minibatch run vs. Label Prediction Error')

    plt.show()

if __name__ == '__main__':
    inititalize_cntk()
    trainer,plot_data,err_percent = do_training()
    show_plot(plot_data)
    print(err_percent)
    
    