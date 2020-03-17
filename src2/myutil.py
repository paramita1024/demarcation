import matplotlib.pyplot as plt
import pickle
import numpy as np
from data_preprocess import *

	
def write_txt_file(data,file_name):
	with open(file_name,'w') as f:
		for line in data:
			f.write(" ".join(map(str,line))+'\n')

def ma(y, window):
	avg_mask = np.ones(window) / window
	y_ma=np.convolve(y, avg_mask, 'same')
	y_ma[0]=y[0]
	y_ma[-1]=y[-1]
	return y_ma

def get_FR(s,t):
	# print float(s.shape[0] - np.count_nonzero(np.sign(s) + np.sign(t))) / s.shape[0]
	if s.shape[0]>0:
		return float(s.shape[0] - np.count_nonzero( np.sign(s) + np.sign(t)))/s.shape[0]
	else:
		return 0

def get_MSE(s,t):
	return np.mean((s-t)**2)

def get_MAPE(s,t):
	return 100* np.mean( np.divide( s - t, s ))

def save(obj,output_file):
	with open(output_file+'.pkl' , 'wb') as f:
		pickle.dump( obj , f , pickle.HIGHEST_PROTOCOL)

def load_data(input_file,flag=None):
	if flag=='ifexists':
		if not os.path.isfile(input_file+'.pkl'):
			# print 'not found', input_file
			return {}
	# print 'found'
	with open(input_file+'.pkl','rb') as f:
		data = pickle.load(f)
	return data






