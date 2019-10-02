import getopt
from sklearn.linear_model import Ridge, HuberRegressor, Lasso
import datetime
import time 
import numpy  as np
from numpy import linalg as LA
# from slant import slant
from myutil import * 
from data_preprocess import *
import sys
# import Scipy
from numpy import linalg as LA
from baselines_class import *



class data_preprocess_baselines:

	def __init__( self, src_data_file, dest_data_file, w ):
		data = load_data( src_data_file )
		self.edges = data.edges
		self.nodes = data.nodes
		self.train = data.train 
		self.test = data.test 
		self.w_slant = w
		self.preprocess()
		self.save( dest_data_file )

	def preprocess( self ):
	
		self.msg_index={}
		self.nbr = {}
		
		for user in self.nodes:
			self.msg_index[ str(user) ]=  np.where( self.train[:,0] == user )[0] 
			self.nbr[ str(user) ] = np.nonzero( self.edges[:,user].flatten() )[0] 
				
		self.data_per_user = {}
		for user in self.nodes : 
			if self.msg_index[ str(user) ].shape[0] == 0 :
				influence_matrix_user = np.array([])
			else:
				influence_matrix_user = np.zeros(( self.msg_index[ str(user) ].shape[0], \
					1+ self.nbr[ str(user ) ].shape[0]))
				nbr_idx = 0 
				influence_matrix_user[:,0] = 1
				for his_nbr in self.nbr[ str(user) ]:
					if self.msg_index[ str(his_nbr)  ].shape[0] > 0 :
						time_old = 0 
						m_idx = 0
						opn = 0 
						merged_index = np.sort( np.concatenate( ( self.msg_index[ str( user) ], \
							self.msg_index[  str( his_nbr)  ])))
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
			self.data_per_user[ str(user) ]={ 'A': influence_matrix_user, \
				'b': self.train[ self.msg_index[ str( user ) ],2].flatten() }
			
	def save( self, dest_data_file ):
		full_data_dict = { 'train': self.train , 'test': self.test }
		data_dict = { 'nodes': self.nodes, 'edges':self.edges, 'all_user': full_data_dict,\
			 'per_user':self.data_per_user, 'msg_index': self.msg_index }
		save( data_dict , dest_data_file)

def preprocess( list_of_file_prefix, path, w_v_dict ):
	src_path = '../Cherrypick_others/Data_opn_dyn_python/'
	src_file_suffix = '_10ALLXContainedOpinionX.obj' 
	dest_path  = path + 'data/'
	for file_prefix in list_of_file_prefix :
		src_file = src_path + file_prefix + src_file_suffix
		dest_file = dest_path + file_prefix 
		data_preprocess_baselines( src_file, dest_file, w_v_dict[file_prefix]['w'])
		# return 

def subset_selection_baselines( path, file_prefix, baseline , param ):
	
	data_file = path  + 'data/' + file_prefix
	data = load_data( data_file )
	
	res_file = path + 'baselines/res.' + file_prefix 
	res=load_data( res_file, 'ifexists')
         
	# if baseline in res:
	# 	print baseline, ' already exists'
	# 	# print res[baseline]
	# 	if res[baseline]:
	# 		res[ 'copy_' + baseline] = dict(res[ baseline ])
	# 		del res[baseline] 
		
	
	if baseline == 'huber_regression':
		obj = Huber_loss_minimization( data, param['set_of_alpha'], param['set_of_epsilon'] )
	# if baseline == 'filtering':
	# 	obj = Heuristic_filtering( data )
	if baseline == 'robust_lasso':
		obj = Extended_robust_lasso( data, param['set_of_lamb_w'], param['set_of_lamb_e'] )
	# if  baseline == 'dense_err':
	# 	obj = Dense_error_correction_via_l1_norm( data )
	if  baseline == 'soft_thresholding':
		obj = Soft_thresholding( data, param['set_of_lamb_w'], param['set_of_lamb_e'] )

	if baseline in res:
		for key in obj.res.keys():
			res[ baseline ][ key ] = obj.res[key]
	else:
		res[baseline] = obj.res
	save( res, res_file )

	# if 'copy_'+baseline in res:
	# 	for key in res['copy_'+baseline].keys():
	# 		if key not in res[baseline]:
	# 			res[baseline][key] = dict( res['copy_'+baseline][key] )

	# print res[baseline].keys()

	

def parse_command_line_input( list_of_file_prefix, list_of_baselines):
	
	argv = sys.argv[1:]
	opts, args = getopt.getopt(argv, 'f:b:', ['file', 'baseline'])
	
	if len(opts) == 0 and len(opts) > 2:
		print ('usage: add.py -a <first_operand> -b <second_operand>')
	else:
		# Iterate the options and get the corresponding values
		for opt, arg in opts:
			# print opt,arg
			if opt == '-f':
				# print 'yes'
				for file_prefix in list_of_file_prefix:
					if file_prefix.startswith(arg):
						sel_file_prefix = file_prefix
						
			if opt == '-b':
				for baseline in list_of_baselines:
					if baseline.startswith(arg):
						sel_baseline = baseline
	return sel_file_prefix, sel_baseline 

def merge( main_dir, baseline_to_add , list_of_file_prefix ):

	for file_prefix in list_of_file_prefix:

		res = load_data( main_dir + file_prefix )
		res[ baseline_to_add ] = load_data( main_dir[:-4] + baseline_to_add + '/res.' + file_prefix )[ baseline_to_add ]
		save( res, main_dir + file_prefix )


def main():

	path = '../Real_Data/'
	list_of_file_prefix = ['barca', 'british_election', 'GTwitter','jaya_verdict', \
		'JuvTwitter',  'MsmallTwitter', 'real_vs_ju_703', \
			 'Twitter' , 'VTwitter']  # 'MlargeTwitter', 'trump_data',
	list_of_baselines = ['huber_regression','filtering', 'robust_lasso', 'dense_err', 'soft_thresholding']
	file_prefix, baseline = parse_command_line_input( list_of_file_prefix, list_of_baselines )

	param = {}

	# huber 
	param['set_of_alpha']=[ 0.6, .8, 1.0, 1.2 ]
	param['set_of_epsilon']=[ 1.5 ]
	
	# soft thresholding,  extended robust lasso, 
	param['set_of_lamb_w'] = [ 0.6, .8, 1.0, 1.2 ]
	param['set_of_lamb_e'] = [ .5 ]
	# # huber 
	# param['set_of_alpha']=[ 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
	# param['set_of_epsilon']=[ 1.35, 1.5 ]
	
	# # soft thresholding,  extended robust lasso, 
	# param['set_of_lamb_w'] = [ 0.0001,0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5 ]
	# param['set_of_lamb_e'] = [ 0.01, 0.1, .3, .5, .7, 1, 1.5 ]

	# print load_data(path + 'res.barca').keys()
	# merge( path + 'res.' , 'huber_regression' , list_of_file_prefix )	
	# ------------------------------------------------ 
	# for file_prefix in list_of_file_prefix:
	for baseline in list_of_baselines:
		start = time.time()
		subset_selection_baselines( path , file_prefix, baseline, param )
		end = time.time()
	print 'File:', file_prefix, ' , Baseline: ' , baseline, ', Time:' , end-start, ' seconds'
	#--------------------------------------------------
	# w_v_dict = load_data('w_v')
	# preprocess( list_of_file_prefix , path , w_v_dict )


if __name__ == "__main__":
	main()
