import datetime
import time 
import numpy  as np
from numpy import linalg as LA
# from slant import slant
from myutil import * 
from data_preprocess import *
import sys
class Robust_Cherrypick:
	def __init__( self , file=None, obj = None, init_by = None, train = None , test = None , edges = None, lamb = None, w_slant=10.0  ):

		if init_by == 'object':
			if file <> None:
				obj = load_data( file )	
			self.train = obj.train
			self.edges = obj.edges
			self.test = obj.test

		if init_by == 'values': 

			self.train = train
			self.edges = edges
			self.test = test	

		if init_by == 'dict':

			self.train=obj['train']
			self.test=obj['test']
			self.edges=obj['edges']
		
		self.num_node= self.edges.shape[ 0 ]
		self.num_train= self.train.shape[ 0 ]
		self.num_test = self.test.shape[ 0 ]
		self.nodes = np.arange( self.num_node )

		# investigate from Paper " Robust regression via hard thresholding "
		
		self.lamb = 0.5
		if lamb:
			self.lamb = lamb
		self.step_length = .1
		self.delta_hybrid = 0  
		self.w_slant=w_slant

	def initialize_data_structures( self ):
		
		self.msg_index={}
		self.msg_mask={}
		self.nbr = {}
		
		for user in self.nodes:
			self.msg_index[ user ]=  np.where( self.train[:,0] == user )[0] 
			self.msg_mask[ user ] =np.ones( self.msg_index[ user ].shape[ 0 ] , dtype=bool)
			self.nbr[ user ] = np.nonzero( self.edges[:,user].flatten() )[0] 
				
		self.influence_matrix = {}
		for user in self.nodes : 
			if self.msg_index[ user ].shape[0] == 0 :
				influence_matrix_user = np.array([])
			else:
				influence_matrix_user = np.zeros(( self.msg_index[user].shape[0], 1+ self.nbr[user].shape[0]))
				nbr_idx = 0 
				influence_matrix_user[:,0] = 1
				for his_nbr in self.nbr[user]:
					if self.msg_index[ his_nbr ].shape[0] > 0 :
						time_old = 0 
						m_idx = 0
						opn = 0 
						merged_index = np.sort( np.concatenate( ( self.msg_index[ user ], self.msg_index[ his_nbr ])))
						for index in merged_index:
							user_curr, time_curr , sentiment = self.train[ index,: ]
							opn *= np.exp( - self.w_slant*( time_curr - time_old)) 
							if user_curr == user:
								influence_matrix_user[ m_idx , 1 + nbr_idx] = opn 
								m_idx += 1  
							else:
								opn += sentiment
							time_old = time_curr
					nbr_idx += 1
			self.influence_matrix[user] = influence_matrix_user  
	
	def create_mask_for_individual_user( self, active_set  ):
		# creates a self variable msg_mask
		# msg mask contains masking of msg for each user separately
		if ~np.any( active_set ): 
			print " no message is selected "

		for user in self.nodes : 
			self.msg_mask[ user ] = active_set[ self.msg_index[ user ] ]


	def robust_regression_via_hard_threshold( self, method , max_itr = 500 , frac_end = None ):
		# sets self.w and self.current_active_set
		self.frac = frac_end
		itr = 0 
		norm_of_residual = []
		self.current_active_set = np.ones( self.num_train, dtype=bool)
		self.create_mask_for_individual_user( self.current_active_set )
		# active_set_old =  np.copy( self.current_active_set ) 
		number_of_change_active_set = 0 
		# recidual_all_user = np.zeros( self.num_train )
		w = {}
		for user in self.nodes:
			w[ user ] = np.zeros( 1 + self.nbr[user].shape[ 0 ] )

		while True:
			residual_all_user = np.zeros( self.num_train )
			for user in self.nodes:
				if self.msg_index[user].shape[0] > 0 :
					# data preprocessing 
					X= self.influence_matrix[ user ]
					Y=self.train[ self.msg_index[ user ], 2] # ck
					w_user = w[user] 
					w_user = self.update( X = X[ self.msg_mask[user] ] , Y = Y[ self.msg_mask[user] ] , w = w_user, method = method , number_of_change_active_set = number_of_change_active_set )
					# print 'shape of msg of user ' + str( self.msg_index[user].shape[0])
					residual_all_user[ self.msg_index[ user ] ] =self.get_residual( Y, X, w_user)
					w[ user ] = w_user 
			active_set_new = self.hard_threshold( residual_all_user , self.frac ) 
			# sets self.msg_mask using active set new. msg mask contains masking of msg for each user separately
			self.create_mask_for_individual_user( active_set_new)
			number_of_change_active_set = self.get_difference( self.current_active_set , active_set_new )
			self.current_active_set = active_set_new
			norm_of_residual.append( LA.norm( residual_all_user[ self.current_active_set ] ) )
			itr += 1 
			# if itr > 1 :
			# 	if abs( norm_of_residual[-1] - norm_of_residual[ -2 ] ) < 1e-5:
			# 		break
			if itr == max_itr:
				break
			if LA.norm( residual_all_user[ self.current_active_set ] ) <= sys.float_info.epsilon : 
				break
			self.w = w 
		return w , self.current_active_set, norm_of_residual
	def get_residual(self, Y, X, w ):

		# print "shape of Y " + str( Y.shape[0])
		# print 'shape of Xw ' + str( X.dot(w).shape[0])
		return np.absolute( Y - X.dot( w ) )
	def get_difference( self, masked_array_1, masked_array_2):
		return np.count_nonzero( np.logical_xor( masked_array_1, masked_array_2 ) )
	def update( self , X, Y, w= None , method = None,  number_of_change_active_set = None  ):
		if method == None:
			print " No method has been provided "
		if method == 'FC' : 
			return self.update_fully_corrective( X, Y, self.lamb )
		if method == 'GD' :
			return self.update_grad_descent( X, Y, w, self.step_length )
		if method == 'HYB' :
			return self.update_hybrid( X , Y, w, self.lamb, self.step_length, number_of_change_active_set, self.delta_hybrid )
	def update_fully_corrective( self , X , Y , lamb ):
		return LA.solve( X.T.dot(X)+ lamb*np.eye( X.shape[1] ), X.T.dot(Y) ) 
	def update_grad_descent( self, X, Y, w, step_length ):
		grad = X.T.dot( X.dot( w ) - Y ) 
		w -= step_length* grad 
		return w  
	def update_hybrid( self, X, Y, w, lamb, step_length, number_of_change_active_set, delta_hybrid ): 
		print 'ERROR:----------------delta_hybrid is not yet defined.'
		if number_of_change_active_set >  delta_hybrid:
			return update_grad_descent( X, Y, w, step_length)
		else:
			return update_fully_corrective( X, Y, lamb)
	def hard_threshold( self , v , frac  ):
		num_smallest_val = int( frac * v.shape[ 0 ] )
		idx =np.argpartition( v , num_smallest_val-1 )[:num_smallest_val]  # ck 
		active_set = np.zeros( self.num_train, dtype=bool)
		active_set[ idx ] = True
		return active_set
	def initialize_alpha_A_slant( self  ):
		alpha = np.zeros( self.num_node )
		A = np.zeros(( self.num_node, self.num_node ))
		for user in self.nodes:
			alpha[user] = self.w[ user ][0] 
			A[ self.nbr[ user ] , user ] = self.w[ user ][1:]
		return alpha, A  
	def init_slant(self): 
		self.slant_obj = slant(init_by = 'values', edges = self.edges, train = self.train[ self.current_active_set ], test= self.test, data_type = 'real') 
		self.slant_obj.estimate_param( ) 
		return self.slant_obj
	def eval_using_slant( self , file , num_simulation_slant, time_span_input ):
		result_obj = [' enter description here']
		result_obj.append(  self.slant_obj.predict( num_simulation = num_simulation_slant , time_span_input = sampling_time_span) )
		save_data( file, result_obj ) 
	def save_active_set( self, file_to_write = None , norm_of_residual = None  , save_msg = False, total_time = None ):
		result = {}
		result['type']='end_msg_boolean'
		result['data']=  self.current_active_set
		print('selected '+str(np.count_nonzero(self.current_active_set))+' events out of '+str(self.current_active_set.shape[0])+' events')
		result['norm_of_residual' ] = norm_of_residual
		result['frac_end'] = self.frac 
		result['time'] = total_time
		result['lambda'] = self.lamb
		if save_msg :
			save( result, file_to_write )
		return result 
def get_diff(v):
	w = v[:-1]
	return v[1:]-w
def main():


	path = '../Cherrypick_others/Data_opn_dyn_python/'
	file_suffix = '_10ALLXContainedOpinionX.obj' # '_full'
	#****************************************************************
	run_single_instance = False
	run_multiple_instance = True
	# sanitize_test = False # True 

	# frac_of_end_msg_list =[.5,.6 , .7 , .9 ] #[.8]#
	# if sanitize_test :
	# 	frac_of_end_msg_list = [ 0.8 ]
		
	if run_multiple_instance : 
		print 'Robust Cherrypick'
		file_prefix_list = ['barca', 'british_election', 'GTwitter','jaya_verdict', 'JuvTwitter', 'MlargeTwitter', 'MsmallTwitter', 'real_vs_ju_703', 'trump_data', 'Twitter' , 'VTwitter'] 
		file_idx=int(sys.argv[1])
		w_slant=int(sys.argv[2])
		# v_slant=int(sys.argv[3])
		
		list_of_lambda =[float(sys.argv[3])]#[.2,.5,.7,.9,2]#[ .5,.7,.9,1.1] # [1.3,1.5,1.7,2] # [.5,.7,.9,1.1] # [ .01, .05, .1, .5, 1 ]
		frac_of_end_msg_list =[.8]#[.5,.6 , .7 , .9 ] #[.8]#
		# fig_idx = 0 
		for file_prefix in [file_prefix_list[file_idx]] :
			file_to_read = path + file_prefix + file_suffix			
			file_to_write_prefix = '../result_subset_selection/' + file_prefix
			# if sanitize_test : 
			# 	file_to_write  = file_to_write + '.sanitize_test'
			
			# result_list = [] # [ file_prefix , 'Robust_cherrypick'  ]
			print 'DATASET: ' + file_prefix
			for frac_end in frac_of_end_msg_list :
				print 'FRACTION : ' + str(frac_end)
				# result_list_frac = []
				for lamb in list_of_lambda :
					print 'LAMBDA: ' + str( lamb)
					start = time.time()
					obj = load_data( file_to_read)
					# if sanitize_test:
					# 	obj.train = np.concatenate(( obj.train , obj.test ), axis = 0 )
					Cherrypick_obj = Robust_Cherrypick( obj = obj , init_by = 'object', lamb = lamb , w_slant=w_slant ) 
					Cherrypick_obj.initialize_data_structures()
					w, active_set, norm_of_residual = Cherrypick_obj.robust_regression_via_hard_threshold( method = 'FC', max_itr = 50 , frac_end = frac_end) 
					total_time = time.time()-start
					# plt.plot(get_diff(np.array(norm_of_residual)))
					# fig_idx += 1
					# f = plt.figure()
					# plt.plot(np.array(norm_of_residual))
					# plt.savefig( file_prefix + '_' + str(fig_idx) +'.png')
					# print total_time
					# return #********
					file_to_write=file_to_write_prefix+'w'+str(w_slant)+'f'+str(frac_end)+'l'+str(lamb)+'.res.Robust_cherrypick'
					result_obj =  Cherrypick_obj.save_active_set( norm_of_residual = norm_of_residual  , total_time = total_time, save_msg=True,file_to_write=file_to_write)
					# result_list_frac.append( result_obj )
					print 'TIME : ' + str( datetime.datetime.now() ) + ' LAMBDA: ' + str( lamb ) + ' DURATION : ' + str(total_time)
				# result_list.append( result_list_frac )
			print file_prefix + ' done '

	if run_single_instance:
		file_prefix = 'jaya_verdict'
		file_to_read = path + file_prefix + file_suffix + '.obj'

		file_to_write = file_prefix +'.res.Robust_cherrypick'
		#---------------------------------------------------
		Cherrypick_obj = Robust_Cherrypick( file = file_to_read , init_by = 'object' , frac_end = .8) 
		Cherrypick_obj.initialize_data_structures()
		# # set method as "FC" , 'GD' or "HYB"
		w, active_set, norm_of_residual = Cherrypick_obj.robust_regression_via_hard_threshold( method = 'FC', max_itr = 500 ) 
		Cherrypick_obj.save_active_set( file_to_write, norm_of_residual  )


if __name__ == "__main__":
	main()


		# #--------------------------------------------------
	# file_to_write = file_prefix +'.res.Robust_cherrypick'
	# #---------------------------------------------------
	# Cherrypick_obj = Robust_Cherrypick( file = file_to_read , init_by = 'object') 
	# Cherrypick_obj.initialize_data_structures()
	# # # set method as "FC" , 'GD' or "HYB"
	# w, active_set, norm_of_residual = Cherrypick_obj.robust_regression_via_hard_threshold( method = 'FC', max_itr = 10 ) 
	# Cherrypick_obj.save_active_set( file_to_write, norm_of_residual  )

	# #--------------------------------------------------------------------------------
	# # Cherrypick_obj.init_slant( )
	# # Cherrypick_obj.eval_using_slant( file_to_write , num_simulation_slant = 20 , time_span_input = 0.3 )
