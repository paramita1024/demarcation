# import matplotlib.pyplot as plt
import pickle
import numpy as np
from data_preprocess import *


class results : 
	def __init__(self, result_val_MSE, result_val_FR, predicted_val, original_val):
		self.result_val_MSE = result_val_MSE 
		self.result_val_FR = result_val_FR 
		self.predicted_val = predicted_val
		self.original_val = original_val

def write_txt_file(data,file_name):
	with open(file_name,'w') as f:
		for line in data:
			f.write(" ".join(map(str,line))+'\n')

def ma(y, window):
	avg_mask = np.ones(window) / window
	y_ma=np.convolve(y, avg_mask, 'same')
	y_ma[0]=y[0]
	# if 0 in list(y):
	# 	idx=np.where(y==0)[0][0]
	# 	if idx>0:
	# 		y_ma[idx-1]=y[idx-1]
	y_ma[-1]=y[-1]
	return y_ma

# def plot_result( data , label_list , num_plot = 1, title_txt = None, xtitle = None, ytitle = None, image_title = None):
# 	f = plt.figure()
# 	if data == np.array([]):
# 		print 'empty'
# 	# title_txt = 'myplot'
# 	if num_plot == 1:
# 		plt.plot( data, label= label_list )
# 	else:
# 		for item, label  in zip(data, label_list)  :
# 			plt.plot(item, label = label)
# 	plt.legend()
# 	plt.title( title_txt)
# 	plt.xlabel( xtitle )
# 	plt.ylabel( ytitle )
# 	# plt.show()
# 	plt.savefig( image_title)
# 	plt.clf()
	# plt.show()
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
# def find_num_msg_plot( msg, num_node, mu):
# 	num_msg_per_user = np.zeros(num_node)
# 	for user in range(num_node):
# 		num_msg_per_user[user] = np.count_nonzero(msg[:,0]==user)
# 	plt.plot( num_msg_per_user, 'r')
# 	# plt.plot( self.curr_int, 'b')

# 	plt.plot( mu, 'b')
# 	plt.show()
# # Name as myutil
# class myutil:
# 	def __init__(self):
# 		pass
# 	def get_FR(self,s,t):
# 		# print float(s.shape[0] - np.count_nonzero(np.sign(s) + np.sign(t))) / s.shape[0]
# 		return float(s.shape[0] - np.count_nonzero( np.sign(s) + np.sign(t)))/s.shape[0]
# 	def get_MSE(self,s,t):
# 		return np.mean((s-t)**2)

	# def get_initial_opn_int( self, train, end_time ):
	# 	num_train_known = np.count_nonzero(train[:,1] < end_time)
		

	


	# def find_opn_markov( self, check_the_arguements ): # or perhaps next one 
	# 	pass

	# init_opn, init_intensity = myutil.get_initial_opn( self.train, end_time )
	# msg_set, last_opn_update, last_int_update =  myutil.simulate_events(time_span_input , init_opn, init_intensity, self.A, self.B)
	# prediction_array[simulation_no] = myutil.predict_from_events( last_opn_update,  user ) # or perhaps next one 
	# prediction_array[simulation_no] = myutil.find_opn_markov( check_the_arguements ) # or perhaps next one 
	# prediction_array[simulation_no] = myutil.predict_from_events( self.alpha[user], self.A[:,user] , last_opn_update,  msg_set, user , time )