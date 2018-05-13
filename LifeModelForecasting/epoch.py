import pickle
import jsonpickle
import datetime
import logging
from config import dateformat, totalSamples, minibatchSize, testPercentage, epochs

class Epoch(object):
    def __init__(self, e, **kwargs):
        self.number = e
        self.data = dict()
        self.input_output = dict()
        self.epoch_start_time=datetime.datetime.now()
        self.epoch_end_time=datetime.datetime.now()

    def Started(self):
        epoch_start_time = datetime.datetime.now()
        logging.info("\t##########################################################################")
        logging.info("\t############# Epoch {0} started @ {1} ####################".format(self.number + 1,str(self.epoch_start_time.strftime(dateformat))))
        logging.info("\t##########################################################################")


    def Finished(self):
        self.epoch_end_time=datetime.datetime.now()
        logging.info("")
        logging.info("*****************************************")
        logging.info("***Epoch Summary***")
        logging.info("Parameters: Samples:{0}, Minibatch Size: {1}, Test ratio: {2} ".format(totalSamples,minibatchSize,testPercentage))    
        logging.info(str(self.data))
        logging.info("***END OF EPOCH {0} of {1}** @ {2} in {3}*".format(self.number + 1, epochs,datetime.datetime.now().strftime(dateformat),str((self.epoch_end_time - self.epoch_start_time))))
        logging.info("*****************************************")
        logging.info("")        

    def SaveToFile(self, directory, model):
        pickleFile = open(directory + '/' + 'data{0}.bin'.format(self.number + 1),'wb') 
        jsonFile = open(directory + '/' + 'data{0}.txt'.format(self.number + 1),'wt')
        epochFile = open(directory + '/' + 'epoch{0}.txt'.format(self.number + 1),'wt')    
        model.save(directory + '/' + 'model_keras{0}.h5'.format(self.number + 1))
        import json
        pickle.dump(self,pickleFile)
        pickleFile.flush()
        pickleFile.close()

        epochFile.write(str(self.number))        
        epochFile.write("\n")
        epochFile.write(str(self.data))
        epochFile.write("\n")
        epochFile.write(str(self.input_output))
        epochFile.close()

        jsonFile.write(jsonpickle.encode(self))        
        #json.dump([epoch.number,epoch.data, epoch.input_output],jsonFile)
        jsonFile.flush()
        jsonFile.close()

