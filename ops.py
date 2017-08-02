from tensorflow.examples.tutorials.mnist import input_data

def mnist_loader():
    '''
    mnist data loader
    
    Return :
        train - dict
            'input'
            'output'
        test - dict
            'input'
            'output'
        val - dict
            'input'
            'output'
    '''
    mnist = input_data.read_data_sets('./MNIST_data/', one_hot=True)
    train = {}
    test = {}
    val = {}
    train['input'] = mnist.train.images
    train['output'] = mnist.train.labels
    test['input'] = mnist.test.images
    test['output'] = mnist.test.labels
    val['input'] = mnist.validation.images
    val['output'] = mnist.validation.labels
    return train, test, val