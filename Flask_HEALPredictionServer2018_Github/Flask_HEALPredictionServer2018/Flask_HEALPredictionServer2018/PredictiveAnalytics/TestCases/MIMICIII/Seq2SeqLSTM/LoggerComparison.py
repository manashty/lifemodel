script_version = '2.0'
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

import pickle
#import json
import jsonpickle
import logging
import matplotlib.pyplot as plt
import os
plt.rcParams.update({'font.size': 14})
show_figures=False
samples = 1000
#samples = 10000
#samples = 34000
data = []
labels = ['LM', 'FS']
#lines=['r--', 'bs', 'g^', '')]
from itertools import cycle
lines = ["-","--","-.",":"]
lines = ["-","--"]#,"-.",":"]
lines = ["-","-","--","--"]#,"-.",":"]
linecycler = cycle(lines)
markers = ["o","v","^","s"]
markers = ["o","o","^","^"]
markers = ["o","v"]
#markers = [""]
#markers = ["r--","bs","g^","g."]
markercycler = cycle(markers )
lw = 2


if samples == 10000:
    directories = ['@@Results/Results 2017-12-14  06-51-42 Samples-10000_MB-400_TestPerc5_LifeModel_Binary',
             '@@Results/Results 2017-12-14  06-52-25 Samples-10000_MB-400_TestPerc5_Fixed_Binary']
    epochNumbers = [28,28]
    epochNumbersSingle = [-1,-1]#The epoch number to compare best epoch (for all diagrams except training/loss)
    #labels = [x + ' 10k' for x in labels]
    
elif samples == 1000:
    directories = ['@@Results/Results 2017-12-14  02-55-28 Samples-1000_MB-100_TestPerc5_LifeModel_Binary',
             '@@Results/Results 2017-12-14  02-56-09 Samples-1000_MB-100_TestPerc5_Fixed_Binary']
    epochNumbers = [10,10]
    epochNumbersSingle = [-1,-1]#The epoch number to compare best epoch (for all diagrams except training/loss)
    #labels = [x + ' 1k' for x in labels]
elif samples == 34000:
    directories = ['@@Results/Results 2017-12-14  06-51-22 Samples-34000_MB-500_TestPerc5_LifeModel_Binary',
             '@@Results/Results 2017-12-14  06-53-01 Samples-34000_MB-500_TestPerc5_Fixed_Binary']
    epochNumbers = [6,6]
    epochNumbersSingle = [-1,-1]#The epoch number to compare best epoch (for all diagrams except training/loss)
    
labels = [x + ' '+str(int(samples/1000))+'k' for x in labels]
import datetime
dateformat="%Y/%m/%d  %H:%M:%S"
dateformatFile='%Y-%m-%d  %H-%M-%S'

figuresDirectory='@@Results/Figures-{size}-epochs{ep}-comp{compEp}-{date}'.format(
    size=str(int(samples/1000))+'k', ep=str(epochNumbers).replace(',','_').replace('[','').replace(']','').replace(' ','')
    ,compEp=str(epochNumbersSingle).replace(',','_').replace('[','').replace(']','').replace(' ',''),
    date=datetime.datetime.now().strftime(dateformatFile))
import os
if not os.path.exists(figuresDirectory):
    os.makedirs(figuresDirectory)

output_name_template=figuresDirectory+'/Figure-{size}-{diagram}.pdf'.format(
    size=str(int(samples/1000))+'k', diagram='{diagram}')

file=open(output_name_template.format(diagram='data').replace('pdf','txt'),'wt')

class Epoch(object):
    def __init__(self, e, **kwargs):
        self.number = e
        self.data = dict()
        self.input_output = dict()

for dir in range(len(directories)):    
    epochs = []
    acc = []
    loss = []
    for ep in range(epochNumbers[dir]):
        file_pickle = open(directories[dir] + '/data{0}.bin'.format(ep + 1),'rb')
        #file_json=open (directory+'/data1.txt','rt')
        epochs.append(pickle.load(file_pickle))
        print(labels[dir])
        print(directories[dir])
        print("Epoch " + str(epochs[-1].number))
        #print(epochs[-1].data)
        #print(epochs[-1].input_output)
    
#res2=pickle.load(file_pickle)

    acc_train = [epoch.input_output['all_batches_training_history_accuracy'][-1][-1] for epoch in epochs]
    loss_train = [epoch.input_output['all_batches_training_history_loss'][-1][-1] for epoch in epochs]

    acc = [epoch.data['accuracy'] for epoch in epochs]
    loss = [epoch.data['loss'] for epoch in epochs]

    e = epochs[epochNumbersSingle[dir]]

    fraction_of_positives = e.input_output['fraction_of_positives']
    mean_predicted_value = e.input_output['mean_predicted_value']
    clf_score = e.data['brier']
    prob_pos = e.input_output['prob_pos']
    fpr, tpr, _ = e.input_output['roc_curve']
    roc_auc = e.data['roc_auc']    

    data.append({'acc_train':acc_train,'loss_train':loss_train, 'acc':acc,'loss':loss
                 ,'fraction_of_positives':fraction_of_positives,'mean_predicted_value':mean_predicted_value,
                 'brier':clf_score,'prob_pos':prob_pos,
                 'fpr':fpr,'tpr':tpr,'roc_auc':roc_auc,'label':labels[dir]})
    file.writelines(directories[dir]+'\n')
    file.writelines('Accuracy:{0}, AUC_ROC:{1}, Brier:{2}'.format(acc[epochNumbersSingle[dir]],roc_auc,clf_score) +'\n\n\n')
    file.writelines(str([x for x in data[-1].items() if (x[0] in  ['brier','roc_auc','acc','loss', 'acc_train','acc_test'])])+'\n')
    file.writelines(''.join(['*' for s in range(50)])+'\n')


file.close()

for x in data:    
    plt.plot(x['acc_train'],marker=next(markercycler),lw=lw,linestyle=next(linecycler))
    plt.plot(x['acc'],marker=next(markercycler),lw=lw, linestyle=next(linecycler))
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend([t + x for x in labels for t in ['train ','test ']], loc='lower right')
plt.savefig(output_name_template.format(diagram='accuracy'),bbox_inches='tight', pad_inches=0)
if(show_figures):
    plt.show()
plt.close()

# summarize history for loss
for x in data:    
    plt.plot(x['loss_train'],marker=next(markercycler),lw=lw,linestyle=next(linecycler))
    plt.plot(x['loss'],marker=next(markercycler),lw=lw,linestyle=next(linecycler))
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend([t + x for x in labels for t in ['train ','test ']], loc='upper right')
plt.savefig(output_name_template.format(diagram='loss'),bbox_inches='tight', pad_inches=0)    
if(show_figures):
    plt.show()
plt.close()

fig = plt.figure(1, figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

e = epochs[0]

fraction_of_positives = e.input_output['fraction_of_positives']
mean_predicted_value = e.input_output['mean_predicted_value']
clf_score = e.data['brier']
prob_pos = e.input_output['prob_pos']
for x in data:
    ax1.plot(x['mean_predicted_value'], x['fraction_of_positives'], #"s-",
                label="%s (%1.3f)" % ('{0}'.format(x['label']), x['brier']),marker=next(markercycler),lw=lw,linestyle=next(linecycler))
#ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
 #               label="%s (%1.3f)" % ('LSTM', clf_score))
for x in data:#Separate for linestyle
    ax2.hist(x['prob_pos'], range=(0, 1), bins=10, label=x['label'],
                histtype="step",lw=lw, linestyle=next(linecycler))#, marker=next(markercycler),lw=lw,linestyle=next(linecycler))

ax1.set_ylabel("Fraction of positives")
ax1.set_xlabel("Mean predicted value")

ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
#plt.savefig(directory+'/Calibration Epoch
#'+str(e+1)+'.pdf',bbox_inches='tight', pad_inches=0)
plt.savefig(output_name_template.format(diagram='calibration'),bbox_inches='tight', pad_inches=0)    

if(show_figures):
    plt.show()
plt.close()



plt.figure()

for x in data:        
    #fpr, tpr, _ = x.input_output['roc_curve']
    plt.plot(x['fpr'], x['tpr'],
             label='ROC curve {0} (area = %0.2f)'.format(x['label']) % x['roc_auc']
            ,marker=next(markercycler),lw=lw,linestyle=next(linecycler))
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig(output_name_template.format(diagram='ROC'),bbox_inches='tight', pad_inches=0)

if(show_figures):
    plt.show()
plt.close()







