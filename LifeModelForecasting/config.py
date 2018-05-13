#USE CPU ONLY
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"       #'-1' for cpu and [0,1,2,3,] for gpu
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"       # GPU 1
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"       # GPU 2
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"       # GPU 3
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"       # GPU 4
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        #https://goo.gl/bZ6eiD



###### CONFIG INPUT 
from config_input import *
#Sampling rate
if(seqToSeq):
    totalSamples, minibatchSize, testPercentage, epochs, minibatch_epochs = (
    40, 20, 0.1, 5, 1)
    #100, 20, 0.1, 2, 1)
    #1000, 100, 0.1, 10, 1)
    #1000, 100, 0.1, 10, 2)
    #1000, 100, 0.2, 50, 2)
    #1000, 10, 0.2, 10, 2)
    #5000, 100, 0.2, 50, 2)
else:
    if(mortality):        
        totalSamples, minibatchSize, testPercentage, epochs, minibatch_epochs = (numberOfSamples, 100, 0.2, 50, 1)   
        #(numberOfSamples, 100, 0.1, 2, 2)
    else:
        totalSamples, minibatchSize, testPercentage, epochs, minibatch_epochs = (numberOfSamples, numberOfSamples, 0.2, 50, 2)


no_mini_batches = int(totalSamples / minibatchSize)   

# fix random seed for reproducibility
import numpy
numpy.random.seed(7)

class Seq2SeqLossFunction(Enum):
    MeanSquaredErrorMSE=0
    MeanToleranceErrorMTE=1

lossFunction=Seq2SeqLossFunction.MeanToleranceErrorMTE

#Create the results directory
import datetime
dateformatFile = '%Y-%m-%d  %H-%M-%S'
directory = "Results " + str(datetime.datetime.now().strftime(dateformatFile)) + ' Samples-{0}_MB-{1}_TestPerc{2}_{3}'.format(totalSamples,minibatchSize, int(testPercentage * 100),method.name)
print(directory)


if not os.path.exists(directory):
    os.makedirs(directory)



#CONFIG LOG
import logging
from keras.callbacks import TensorBoard
#region Initialize Logging
#An empty file with the name of the input file!
#open(directory+'/'+filename.format("Input"),'wb').close()
logFile = open(directory + '/' + 'log.txt','wt')
logging.basicConfig(filename=directory + '/' + 'log.txt',format='%(message)s', level=logging.DEBUG)
#Adding log to console as well
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logging.Formatter('*\t%(message)s'))
logging.getLogger().addHandler(consoleHandler)
dateformat = "%Y/%m/%d  %H:%M:%S"
logging.info("Log file created at " + datetime.datetime.now().strftime("%Y/%m/%d  %H:%M:%S"))
logging.info("Directory: {0}".format(directory))
logging.info("Filename Template: {0}".format(filename.format(input_file_variable_name)))
logging.info("Filename Input: {0}".format(filename.format(input_file_variable_name)))
logging.info("Filename Output: {0}".format(filename.format("Output")))
tensorboard = TensorBoard(log_dir=directory, write_graph=True)
tensorboard_callbacks=[tensorboard]#,
                                           #tbCallBack
                                           #]
startTime = datetime.datetime.now()
logging.info('Start time: ' + str(startTime))
