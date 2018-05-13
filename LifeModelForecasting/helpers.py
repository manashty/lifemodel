import pickle
import jsonpickle

class Epoch(object):
    def __init__(self, e, **kwargs):
        self.number = e
        self.data = dict()
        self.input_output = dict()

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


