script_version='2.0'
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

import pickle
#import json
import jsonpickle
import logging
import matplotlib.pyplot as plt

directory='Results 2017-12-14  01-28-35 Samples-40_MB-20_TestPerc20_LifeModel_Binary'
class Epoch(object):
    def __init__(self, e, **kwargs):
        self.number=e
        self.data=dict()
        self.input_output=dict()

epochs=[]
acc=[]
loss=[]

import os

for ep in range(1,11):
        file_pickle=open (directory+'/data{0}.bin'.format(ep),'rb')
        #file_json=open (directory+'/data1.txt','rt')
        epochs.append(pickle.load(file_pickle))
        print(epochs[-1].number)
        print(epochs[-1].data)
        print(epochs[-1].input_output)
    
#res2=pickle.load(file_pickle)



acc_train=[epoch.input_output['all_batches_training_history_accuracy'][-1][-1] for epoch in epochs]
loss_train=[epoch.input_output['all_batches_training_history_loss'][-1][-1] for epoch in epochs]

acc=[epoch.data['accuracy'] for epoch in epochs]
loss=[epoch.data['loss'] for epoch in epochs]

plt.plot(acc_train)
plt.plot(acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()

# summarize history for loss
plt.plot(loss_train)
plt.plot(loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
#plt.show()

#print(str(res.data))
#print(str(res2.data))

'''file.close ()
dic={'test':[5, 5, "45"]}

print (dic)

file=open ('log.txt','wb')

pickle.dump (dic,file)

pickle.dump (dic,file)

d=jsonpickle.encode (dic)
print (d)



#jj=json.load(file_json.readline())
#txt=''
#for lines in file_json:
 #   txt+=lines
    

#j=jsonpickle.decode(txt)

#print (j)

print (j)

file.close ()

'''

#obj=pickle.load()

fig = plt.figure(1, figsize=(10, 10))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
#fraction_of_positives, mean_predicted_value = \
 #       calibration_curve(y_test_all, prob_pos, n_bins=10)

e=epochs[0]




fraction_of_positives=e.input_output['fraction_of_positives']
mean_predicted_value =e.input_output['mean_predicted_value']
clf_score=e.data['brier']
prob_pos=e.input_output['prob_pos']
for x in epochs:
    ax1.plot(x.input_output['mean_predicted_value'], x.input_output['fraction_of_positives'], "s-",
                label="%s (%1.3f)" % ('Epoch{0}'.format(str(x.number+1)), x.data['brier']))
#ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
 #               label="%s (%1.3f)" % ('LSTM', clf_score))

ax2.hist(prob_pos, range=(0, 1), bins=10, label='LSTM',
                histtype="step", lw=2)

ax1.set_ylabel("Fraction of positives")
ax1.set_xlabel("Mean predicted value")

ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right")
ax1.set_title('Calibration plots  (reliability curve)')

ax2.set_xlabel("Mean predicted value")
ax2.set_ylabel("Count")
ax2.legend(loc="upper center", ncol=2)

plt.tight_layout()
#plt.savefig(directory+'/Calibration Epoch '+str(e+1)+'.pdf',bbox_inches='tight', pad_inches=0)
    

#plt.show()

plt.close()




fpr, tpr, _= epochs[0].input_output['roc_curve']


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % epochs[0].data['roc_auc'])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()



