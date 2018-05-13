import matplotlib.pyplot as plt
from epoch import Epoch
from enum import Enum
from config import directory
metricsDic = {}

class Metric(Enum):
    loss = 0    
    accuracy = 2    
    mse = 4    
    mte = 6

class MetricData(object):
    def __init__(self, title:str, ylabel: str, fileName:str, batch_train_data_index:str, epoch_test_data_index:str):
        self.title = title # Title of the plot
        self.ylabel = ylabel #Y Label for the plot
        self.fileName = fileName#The file name to save the metric
        self.batch_train_data_index = batch_train_data_index # The index of batch_data_index that stores all data for all mini batches for
                                                             # each epochs (We
                                                                                                                         # usually use the
                                                                                                                         # last one)
        self.epoch_test_data_index = epoch_test_data_index # The key of the "test" result for this metric in epoch.data dictionary
    
metricsDic[Metric.loss.name] = MetricData(title="Model Loss", ylabel="loss", fileName="Loss", batch_train_data_index='all_batches_training_history_loss', epoch_test_data_index='loss')
metricsDic[Metric.accuracy.name] = MetricData(title="Model Accuracy", ylabel="accuracy", fileName="Accuracy", batch_train_data_index='all_batches_training_history_accuracy', epoch_test_data_index='accuracy')
metricsDic[Metric.mse.name] = MetricData(title="Model Mean Square Error", ylabel="MSE", fileName="MSE", batch_train_data_index='all_batches_training_history_mse',epoch_test_data_index='mse')
metricsDic[Metric.mte.name] = MetricData(title="Model Mean Tolerance Error", ylabel="MTE", fileName="MTE", batch_train_data_index='all_batches_training_history_mte',epoch_test_data_index='mte')



class Visualization(object):
    @staticmethod
    def PlotMetric(all_epochs_Data, metric: Metric, metric_compare:Metric=None):

        metric_train = [epoch_iter.input_output[metricsDic[metric.name].batch_train_data_index][-1][-1] for epoch_iter in all_epochs_Data]
        metric_test_epoch = [epoch_iter.data[metricsDic[metric.name].epoch_test_data_index] for epoch_iter in all_epochs_Data]
        
        #if(metric_compare is not None):
        #    metric_train_2 = [epoch_iter.input_output[metricsDic[metric_compare.name].batch_train_data_index][-1][-1] for epoch_iter in all_epochs_Data]
        #    metric_test_epoch_2 = [epoch_iter.data[metricsDic[metric_compare.name].epoch_test_data_index] for epoch_iter in all_epochs_Data]

        # summarize history for loss
        plt.plot(metric_train)
        plt.plot(metric_test_epoch)
        plt.title(metricsDic[metric.name].title)
        plt.ylabel(metricsDic[metric.name].ylabel)
        legend=['train', 'test']        
        plt.legend(legend, loc='upper left')
        plt.xlabel('epoch')        
        #plt.show()
        plt.tight_layout()
        plt.savefig(directory + '/' + metricsDic[metric.name].fileName + str(all_epochs_Data[-1].number + 1) + '.pdf',bbox_inches='tight', pad_inches=0)
        plt.close()

    #@staticmethod
    #def PlotLoss(all_epochs_Data: list(Epoch), metric: Metric):
    #    loss_train = [epoch_iter.input_output['all_batches_training_history_loss'][-1][-1] for epoch_iter in all_epochs_Data]
    #    loss = [epoch_iter.data['loss'] for epoch_iter in all_epochs_Data]

    #    # summarize history for loss
    #    plt.plot(loss_train)
    #    plt.plot(loss)
    #    plt.title('model loss')
    #    plt.ylabel('loss')
    #    plt.xlabel('epoch')
    #    plt.legend(['train', 'test'], loc='upper left')
    #    #plt.show()
    #    plt.tight_layout()
    #    plt.savefig(directory + '/Loss' + str(e + 1) + '.pdf',bbox_inches='tight', pad_inches=0)
    #    plt.close()


