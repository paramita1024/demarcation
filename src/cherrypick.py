import time
import matplotlib.pyplot as plt
import numpy as np
from slant import *
from myutil import *

class cherrypick:
	def __init__( self , file=None, obj = None, init_by = None, train = None , test = None , edges = None , param  = None, w=10.0, batch_size=1):
		if init_by == None:
			print "Please mention how to initialize "
		if init_by == 'object':
			if file <> None:
				obj = load_data( file )	
			else:
				if obj == None:
					print "please pass the object to initialize"
					return 
			self.train = obj.train
			self.edges = obj.edges
			self.test = obj.test

		if init_by == 'values':

			self.train = train
			self.edges = edges
			self.test = test	

		if init_by == 'dict':

			self.train = obj['train']
			self.test=obj['test']
			self.edges=obj['edges']

		self.num_node= self.edges.shape[ 0 ]
		self.num_train= self.train.shape[ 0 ]
		self.num_test = self.test.shape[ 0 ]
		self.nodes = np.arange( self.num_node )

		self.sigma_covariance = 1. # ck 
		self.lamb = 1.0
		self.w = w
		self.batch_size=batch_size
		# print 'value of w inside cherrypick module',self.w

		if param <> None:
			self.sigma_covariance = param['sigma_covariance']
			self.lamb = param['lambda']
			# self.w = 4.0

		# print self.num_train
		# print self.num_test
		# print self.num_node
		# print self.test.shape
	def create_influence_matrix(self):
		influence_matrix = np.zeros(( self.num_train, 1+self.num_node)) 
		influence_matrix[:,0] = 1 
		msg_index = 0
		time_old = 0

		reminder = {}
		for user, time, sentiment in self.train : 
			user = int(user)
			if msg_index > 0 :
				influence_matrix[msg_index, 1:] = influence_matrix[msg_index-1 , 1:]*np.exp(-self.w*(time - time_old) ) 
			 	influence_matrix[msg_index, reminder['user']+1] += reminder['sentiment']*np.exp(-self.w*(time - time_old)) # confirm whether we need to use reminder at all, I believe we do not need
			reminder['user'] = user 
			reminder['sentiment'] = sentiment
			msg_index += 1
			time_old = time
		self.influence_matrix = influence_matrix
		return influence_matrix
	def set_c( self ): # ck 
		max_msg_influ_mat = np.max( np.absolute( self.influence_matrix ) , axis = 1 ) ** 2 
		tmp = np.zeros( self.num_node )
		for user in self.nodes:
			msg_idx = np.where( self.train[:,0] == user )[0]
			tmp[ user ] = np.sum( max_msg_influ_mat[ msg_idx ]  )
		# print 'check c of cherrypick'
		# print np.max(tmp)
	
		self.lamb =  5*np.max(tmp)/(self.sigma_covariance**2)
	def create_neighbours(self):
		self.incremented_nbr={}
		for user in self.nodes:
			neighbours = np.nonzero(self.edges[:,user].flatten())[0]
			self.incremented_nbr[user]=np.concatenate((np.array([0]),neighbours+1))
	def create_covariance_inverse(self):
		self.covariance_inverse={}
		for user in self.nodes:
			self.covariance_inverse[user]=np.eye(self.incremented_nbr[user].shape[0])/self.lamb
	def create_init_data_structures( self ):
		self.create_neighbours()
		self.create_influence_matrix() 
		self.create_covariance_inverse()
		self.msg_end = np.zeros(self.num_train, dtype = bool)
		self.list_of_msg=[]
		# # not required
		# self.summation_term = np.ones( self.num_node )
		# self.function_val = np.log( self.summation_term )
		# self.inner_prod_influence_vector = np.zeros( self.num_train )
		# for msg_index , influence_vector in zip( range( self.num_train ) , self.influence_matrix ): # ck 
		# 	self.inner_prod_influence_vector[ msg_index ] = (influence_vector.dot( influence_vector ))/( self.lamb * (self.sigma_covariance**2))
		# self.set_c()
	def get_influence_vector(self,user, msg_num):
		return self.influence_matrix[msg_num][self.incremented_nbr[user]]
	def obtain_most_endogenius_msg_user(self): 
		inc = - float('inf') * np.ones( self.num_train) 
		for msg_no in np.nonzero(~self.msg_end)[0]: 
			user = int(self.train[msg_no,0]) 
			influence_vector= self.get_influence_vector(user, msg_no)
			tmp = influence_vector.dot(self.covariance_inverse[user].dot(influence_vector))
			inc[msg_no] = np.log(1+ tmp/(self.sigma_covariance**2) )
			
		for itr in range(self.batch_size):
			msg_to_choose = np.argmax(inc) 
			inc[msg_to_choose]= - float('inf')
			corr_user  = int( self.train[ msg_to_choose , 0 ] )
			
			if self.msg_end[msg_to_choose]:
				print " A message which is already endogenious has been selected again as endogenious msg"
			self.msg_end[ msg_to_choose ] = True
			self.list_of_msg.append( msg_to_choose)
			self.update_inverse( msg_to_choose , corr_user )
			if max(inc) == -float('inf'): 
				print "The maximum entry in increment array is - Infinity"
	
		# return msg_to_choose	
	def update_inverse(self, msg_num, user ):
		influence_vector=self.get_influence_vector(user,msg_num)
		w = (self.covariance_inverse[user].dot( influence_vector)).reshape(influence_vector.shape[0],1)
		self.covariance_inverse[user] -= (w.dot(w.T))/( self.sigma_covariance**2 + influence_vector.dot(w))
	def demarkate_process(self, frac_end): 
		#---------------CHANGE----------------------------------
		# create msg_end
		# nodes_end , nodes_exo
		# frac_nodes_end , frac_msg_end
		# return nodes_end , msg_end
		#-------------------------------------------------
		# max_end_user = int(frac_nodes_end*self.num_nodes)
		
		max_end_msg = int(frac_end * self.num_train)
		self.create_init_data_structures() 
		num_end_msg = 0 
		start=time.time()
		while  num_end_msg < max_end_msg: 
			
			self.obtain_most_endogenius_msg_user() 
			end=time.time()
			if num_end_msg%5000==0:
				print num_end_msg,' selected in ',end-start,' seconds'
				start=time.time()
			num_end_msg += self.batch_size
			if max_end_msg - num_end_msg < self.batch_size:
				self.batch_size=max_end_msg - num_end_msg
				print 'batch size',self.batch_size

		
	def init_slant(self):
		#----------------------------------------------------------
		# init slant 
		# call slant estimate
		#----------------------------------------------------------
		slant_obj = slant( init_by='values', edges = self.edges , train = self.train[ self.msg_end ], test = self.test,  data_type = 'real') 
		slant_obj.estimate()
		self.slant_obj = slant_obj
	def eval_using_slant( self , file , num_simulation_slant, sampling_time_span ):
		results = self.slant_obj.predict( num_simulation = num_simulation_slant , time_span_input = sampling_time_span)
		result_obj = [' enter description here']
		result_obj.append(  results  )
		save_data( file, result_obj )

		# ------------------------------ For list of time span -----------------------------------------------------------------------
		# result_list.append()
		# for time_span_input in np.linspace(0,.5,.1):
		# 	result_val_MSE, result_val_FR, predicted_val = slant_obj.predict( num_simulation = 20 , time_span_input = time_span_input )
		# 	result_obj = results( result_val_MSE, result_val_FR, predicted_val, slant_obj.test[:,2])
		# 	result_list.append( result_obj)
	def reinitialize_slant_data( self , set_train = False , set_test = False, train_idx = None, test_idx = None  ):
		# if flag == None:
		# 	print ' flag must be set to full train to indicate slant train data must be reinitialized with full training data'
		# can print number of train before and afterwards of slant obj 
		if set_train:
			self.slant_obj.train = self.train[ train_idx] 
			self.slant_obj.num_train = self.train.shape[0]
		if set_test : 
			self.slant_obj.test = self.test[ test_idx ]
			self.num_test = self.test.shape[0]
	def save_end_msg( self , file= None , save_msg = False , frac_end = None , total_time = None):
		num_end = int( frac_end* self.num_train)
		boolean_arr = np.zeros(self.num_train, dtype='bool')
		boolean_arr[ self.list_of_msg[:num_end]] = True
		result = {}
		result['w']=self.w
		result['type']='end_msg_boolean'
		result['data']= boolean_arr
		print('selected '+str(np.count_nonzero(boolean_arr))+' events out of '+str(boolean_arr.shape[0])+' events')
		result['frac_end'] = frac_end
		result['lambda'] = self.lamb
		result['time'] = total_time  
		result['sigma_covariance'] = self.sigma_covariance
		if save_msg:
			save( result, file )
		return result




def main():

	path = '../Cherrypick_others/Data_opn_dyn_python/'
	file_suffix = '_10ALLXContainedOpinionX.obj'#'_full'#

	run_single_instance = False
	run_multiple_instance = True # False # True
	sanitize_test = False #  True


	if run_multiple_instance : 
		# w,v=10,10
		file_prefix_list = ['barca','british_election','GTwitter','jaya_verdict', 'JuvTwitter' , 'MlargeTwitter', 'MsmallTwitter', 'real_vs_ju_703', 'trump_data', 'Twitter' , 'VTwitter']
		
		file_idx  = int( sys.argv[1] )
		w=int(sys.argv[2])
		# v=int(sys.argv[3])
		list_of_lambda =[float(sys.argv[3])]#[[.2,.5,.7,.9,2][int(sys.argv[2])]] # [.6,.7,.8,.9,1.1,2]
		file_prefix_list = [ file_prefix_list[file_idx] ]
		frac_of_end_msg_list = [0.8]#[.5,.6 , .7,.9]   # #
		for file_prefix in file_prefix_list :
			print 'DATASET: '+ file_prefix
			print 'sanitize :' + str(sanitize_test)
			file_to_read = path + file_prefix + file_suffix 
			file_to_write_prefix = '../result_subset_selection/' + file_prefix 
			
			obj = load_data( file_to_read)
			if sanitize_test : 
				file_to_write  = file_to_write + '.sanitize_test'	
				print 'Train sample'+ str(obj.train.shape[0])
				obj.train = np.concatenate(( obj.train , obj.test ), axis = 0 )
				print 'After adding test data, current train sample: '+str(obj.train.shape[0])
				
			# Sigma covariance 
			
			param={}
			# result_list_lamb = []
			for lamb in list_of_lambda:
				print 'Lambda: ' + str(lamb)
				param['lambda']=lamb
				# get sigma covar
				slant_obj = slant( obj=obj , init_by = 'object'  , data_type = 'real', tuning = True, tuning_param = [w,10,lamb] ) # define separately for sanitize test 
				param['sigma_covariance'] = slant_obj.get_sigma_covar_only()
				del slant_obj
				
				start = time.time()
				cherrypick_obj = cherrypick( obj = obj , init_by = 'object', param = param,w=w ) 
				cherrypick_obj.demarkate_process(frac_end=1)
				total_time = time.time() - start
				
				for frac_end in frac_of_end_msg_list :
					print 'FRAC END : ' + str(frac_end)
					file_to_write=file_to_write_prefix+'w'+str(w)+'f'+str(frac_end)+'l'+str(lamb)+'.res.cherrypick'#+'.full'
					result_obj = cherrypick_obj.save_end_msg(save_msg=True,file=file_to_write, frac_end = frac_end, total_time = total_time  ) 
					# result_list_frac.append( result_obj )
				# result_list_lamb.append( result_list_frac )
				del cherrypick_obj
			# info_details = {}
			# info_details['name'] = file_prefix 
			# info_details['method'] = 'cherrypick'
			# info_details['list_of_fraction'] = frac_of_end_msg_list
			# info_details['list_of_lambda'] = list_of_lambda
			# info_details['sigma_covariance'] = param['sigma_covariance'] 
			# result_list_lamb.append( info_details )
			# save( result_list_lamb , file_to_write )
			print file_prefix + ' done '




	if run_single_instance:	# frac_msg_end 
		frac_msg_end = 0.8 
		# num_simulation_slant = 5
		# sampling_time_span = 0.3

		file_prefix = sys.argv[1]
		data_type = 'real'
		if data_type=='real':
			# file_prefix = 'JuvTwitter'
			
			filename = path + file_prefix + file_suffix 
		else:
			filename = 'synthetic_data_5_node'

		# print ' file to read : ' + str( filename)
		
		file_to_read = filename + '.obj'
		file_to_write = file_prefix + '.res.cherrypick'
		
		# change init and estimate_param in slant 
		# return
		cherrypick_obj = cherrypick( file = file_to_read , init_by = 'object') 
		cherrypick_obj.demarkate_process(frac_msg_end)
		cherrypick_obj.save_end_msg( file_to_write , True )
		# cherrypick_obj.init_slant()
		# cherrypick_obj.reinitialize_slant_data( set_train = True ,  train_idx = np.arange( cherrypick_obj.num_train ) )
		# cherrypick_obj.eval_using_slant( file_to_write , num_simulation_slant, sampling_time_span ) 


	
if __name__== "__main__":
  main()




#--------------------------------------------------------------------------------------------------------------------------------------------

# class cherrypick:
# 	def __init__(self,filename):
# 		# read the data as a list of (user,msg,time)
# 		# split to create per user sorted list of msg
# 		# split each user list in test and train

# 		with open(filename,'rb') as f:
# 			data = pickle.load(f)
# 			# data is a class containing graph, test and train
# 		self.graph = data.graph
# 		self.train = data.train
# 		self.test = data.test
# 	def find_H_and_O(self):
# 		# init H,V,O,I
# 		H=[]
# 		V=range(ntrain)
# 		O=[]
# 		I=range(nuser)
# 		# number of user not exceeded, 
# 		while len(O) <= self.max_end_user:
# 			# select msg and user 
# 			H.append(m)
# 			V.remove(m)
# 			O.append(u)
# 			I.remove(u)

		
# 		# while msg limit has not reached
# 		# select a msg 
# 		# include in H , exclude from V
# 		# return H,O
# 		while len(H) <= self.max_end_msg:
# 			# select m
# 			H.append(m)
# 			V.remove(m)
# 		return H,V,O,I
# 	def train_model(self):
# 		H,V,O,I = self.find_H_and_O()
# 		# modify input train test graph
# 		self.train,self.test, self.train_ex, self.test_ex = self.reduce()
# 		self.slant_opt= slant(self.graph)
# 		self.slant_opt.estimate_param(self.train)# define and pass parameters	
# 	def forecast(self):
# 		self.result = self.slant_opt.predict_sentiment(self.test)
# 	# def create_graph(self):
# 	# 	self.graph={}
# 	# 	for v in num_v:
# 	# 		self.graph[v]=set([])
# 	# 	for node1,node2 in set_of_egdes:
# 	# 		self.graph[node1].add(node2)
# 	# def load(self):
# 	# 	data=np.genfromtxt(self.fp,delimiter=',')
# 	# 	user,index,count = np.unique(data,return_index=True, return_counts=True)
# 	# 	for i in range(nuser):
# 	# 		tr_idx = np.concatenate([tr_idx, index[i,:np.floor(self.split_ratio*count[i])]])
# 	# 		te_idx = np.concatenate([tr_idx, index[i,np.floor(self.split_ratio*count[i]):]])
# 	# 	train=data[tr_idx,:]
# 	# 	test=data[te_idx,:]
# 	# 	self.ntrain=train.shape[0]
# 	# 	self.ntest=test.shape[0]
# 	# 	self.nuser=user.shape[0]



#---------------------------------------------------------------------------------------------------------------






# def train_model(self):
	# 	H,V,O,I = self.find_H_and_O()
	# 	# modify input train test graph
	# 	self.train,self.test, self.train_ex, self.test_ex = self.reduce()
	# 	self.slant_opt= slant(self.graph)
	# 	self.slant_opt.estimate_param(self.train)# define and pass parameters	
	# def forecast(self):
	# 	self.result = self.slant_opt.predict_sentiment(self.test)
	# def create_graph(self):
	# 	self.graph={}
	# 	for v in num_v:
	# 		self.graph[v]=set([])
	# 	for node1,node2 in set_of_egdes:
	# 		self.graph[node1].add(node2)
	# def load(self):
	# 	data=np.genfromtxt(self.fp,delimiter=',')
	# 	user,index,count = np.unique(data,return_index=True, return_counts=True)
	# 	for i in range(nuser):
	# 		tr_idx = np.concatenate([tr_idx, index[i,:np.floor(self.split_ratio*count[i])]])
	# 		te_idx = np.concatenate([tr_idx, index[i,np.floor(self.split_ratio*count[i]):]])
	# 	train=data[tr_idx,:]
	# 	test=data[te_idx,:]
	# 	self.ntrain=train.shape[0]
	# 	self.ntest=test.shape[0]
	# 	self.nuser=user.shape[0]


#------------------------------------------------------------------------------------------------------------------

	# 	# read the data as a list of (user,msg,time)
	# 	# split to create per user sorted list of msg
	# 	# split each user list in test and train

	# 	with open(filename,'rb') as f:
	# 		data = pickle.load(f)
	# 		# data is a class containing graph, test and train
	# 	self.graph = data.graph
	# 	self.train = data.train
	# 	self.test = data.test
	# def find_argmax(self, nodes_end, nodes_exo, msg_end, msg_exo):
	# 	users_of_end_msgs = self.train[ msg_end, 0 ]
	# 	max_inc = - Inf
	# 	for user in nodes_exo.nonzero()[0]:
	# 		index_user = np.where(users_of_end_msgs == user)[0]
	# 		if index_user.shape[0] > 0 :
	# 			msg_end_indices #
	# 			user_msg_end = msg_end_indices[ index_user ]
	# 			# add msg of those indices
	# 			flag_change_user = True
	# 		for msg in msg_exo.nonzero()[0]:
	# 			user_curr, time_curr, sentiment_curr = self.train[msg,:]
	# 			if nodes_end[user_curr] | user_curr == user:
	# 				# add info of this msg
	# 				flag_change_msg = True
	# 			if flag_change_msg | flag_change_user:
	# 				# compute change or current increment
	# 				# if it is current max, set that
	# 				if current_inc > max_inc:
	# 					max_user = user
	# 					max_msg_no = msg
	# 					max_inc = current_inc

	# 			flag_change_msg  = False
	# 		flag_change_user = False
	# 	return max_msg_no , max_user


#----------------------------------------------------------------------------------------------------





# class slant_input_data: # check whether slant will take this input
# 	def __init__(self, edges, test, train):

# 		# check slant input
# 		# create it accordingly
# 		# pass to slant 
		
# 		self.edges = edges
# 		self.test = test
# 		self.train = train

# 		self.nodes = np.arange( self.edges.shape[0])
# 		self.num_node = self.nodes.shape[0]
# 		self.num_train = self.train.shape[0]
# 		self.num_test = self.test.shape[0]



#------------------------------------------------------------------------------------------------------




# def create_covariance_matrix(self):
	# 	self.covariance = np.zeros((self.num_node, self.num_node+1, self.num_node+1)) 
	# 	for user in self.nodes:
	# 		self.covariance[user] = self.c *np.eye( self.num_node+1 ) 
	
