import numpy as np

def ITS2Array(ITS, F, V):    
    sample_row = ""
    for window in ITS:
			#print(x[0][window].reshape((1,no_of_variables*len(value_abstractions)))[0])
        S=np.array(window)
        a = np.array(S.reshape((1,F * V))[0])
	 #a = a / window_sizes[w]
     #f.write(str(a.tolist()).strip('[').strip(']')+",")
        sample_row+=str(a.tolist()).strip('[').strip(']') + ","
    return sample_row

def WriteSample(file,ITS,F,V):
    """Given an open file, and an ITS, it converts the ITS to a flat array and writes it to the file as a line"""
    #for x in (D[0]):	
    sample_row = ITS2Array(ITS,F,V)
    a=file.write(sample_row.strip(',')+"\n")