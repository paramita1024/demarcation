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
from synthetic_data_msg import synthetic_data_msg


class data_preprocess_baselines:

	def __init__( self, src_data_file, dest_data_file, w, init_by='obj', tr_frac=0.0 ):
		
		data = load_data( src_data_file )
		if init_by == 'obj':
			self.edges = data.edges
			self.nodes = data.nodes
			self.train = data.train 
			self.test = data.test 

		if init_by == 'dict_for_sanitize_test':
			self.edges = data['edges']
			self.nodes = data['nodes']
			self.train = data['all_user']['train']
			self.test = data['all_user']['test']
			self.train = np.vstack(( self.train, self.test))

		if init_by == 'dict_for_vary_train':
			self.edges = data['edges']
			self.nodes = data['nodes']
			self.train = data['all_user']['train']
			self.test = data['all_user']['test']
			num_train = self.train.shape[0]
			self.train = self.train[:int(num_train * tr_frac)]


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
    
	if baseline == 'huber_regression':
		obj = Huber_loss_minimization( data, param['set_of_alpha'], param['set_of_epsilon'] )
	if baseline == 'robust_lasso':
		obj = Extended_robust_lasso( data, param['set_of_lamb_w'], param['set_of_lamb_e'] )
	if  baseline == 'soft_thresholding':
		obj = Soft_thresholding( data, param['set_of_lamb_w'], param['set_of_lamb_e'] )

	if baseline in res:
		for key in obj.res.keys():
			res[ baseline ][ key ] = obj.res[key]
	else:
		res[baseline] = obj.res
	save( res, res_file )

	
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

def preprocess_for_sanitize_test( path, list_of_file_prefix):
	src_path = path + 'data/'
	dest_path  = path+'sanitize/data/'
	w_v_dict = load_data('w_v')
	for file_prefix in list_of_file_prefix :
		src_file = src_path + file_prefix 
		dest_file = dest_path + file_prefix 
		data_preprocess_baselines( src_file, dest_file, w_v_dict[file_prefix]['w'], init_by='dict_for_sanitize_test')
	
def sanitize_test( path , list_of_file_prefix, list_of_baselines):

	param = {}
	param['set_of_epsilon']=[ 1.5 ]
	param['set_of_lamb_e'] =[ 0.5 ]
	
	for file_prefix in list_of_file_prefix:
		res_file =  path + 'exp1/res.' + file_prefix
		lambdas = load_data(res_file)['lambda']['hawkes']
		
		
		for baseline in list_of_baselines:
			start = time.time()
			param['set_of_alpha']=[ lambdas[baseline][2] ]
			param['set_of_lamb_w'] = [ lambdas[baseline][2] ]
			subset_selection_baselines( path + 'sanitize/', file_prefix, baseline, param )
			end = time.time()
			print 'File:', file_prefix, ' , Baseline: ' , baseline, ', Time:' , end-start, ' seconds'

	
def vary_train( path, list_of_file_prefix, list_of_baselines, list_of_tr_frac, flag_selected = False):
	

	param = {}
	param['set_of_epsilon']=[ 1.5 ]
	param['set_of_lamb_e'] =[ 0.5 ]
	param['set_of_alpha']=[ 0.05, 0.1, 0.3,0.5, 0.7, 1.0]
	param['set_of_lamb_w'] = [ 0.05, 0.1, 0.3,0.5, 0.7, 1.0 ]

	for file_prefix in list_of_file_prefix:

		if flag_selected:
			res_lambda = load_data(path + '../exp1/res.' + file_prefix )['lambda']['hawkes']# check path 
		print file_prefix
		for baseline in list_of_baselines:
			print baseline
			for tr_frac in list_of_tr_frac:
				print tr_frac
				data_file = path + 'data/' + file_prefix + '_tr_frac_' + str(tr_frac) 
				start = time.time()
				if flag_selected:
					lamb = res_lambda[baseline][2]
					param['set_of_alpha'] = [ lamb ]
					param['set_of_lamb_w'] = [lamb ]
				subset_selection_baselines( path , file_prefix + '_tr_frac_' + str(tr_frac) , baseline, param )
				end = time.time()
				print 'File:', file_prefix, ' , Baseline: ' , baseline, ', Time:' , end-start, ' seconds'

def preprocess_for_vary_train( path, list_of_file_prefix, list_of_tr_frac):
	src_path = path + 'data/'
	dest_path  = path+'vary_train/data/'
	w_v_dict = load_data('w_v')
	for file_prefix in list_of_file_prefix :
		for tr_frac in list_of_tr_frac:
			src_file = src_path + file_prefix 
			dest_file = dest_path + file_prefix + '_tr_frac_' + str(tr_frac) 
			data_preprocess_baselines( src_file, dest_file, \
				w_v_dict[file_prefix]['w'], init_by='dict_for_vary_train', tr_frac=tr_frac)

def preprocess_synthetic_files_vary_noise():
	src_path = '../various_results/result_synthetic_dataset/dataset/'
	dest_path  = '../Synthetic/vary_noise/data/'
	w_v=load_data('../Synthetic/vary_noise/w_v_synthetic')
	list_of_file_prefix1 = ['barabasi',  'kron_std_512', 'kron_CP_512','kron_Homo_512', \
	'kron_Hetero_512', 'kron_Hier_512' ]
	list_of_file_prefix2 = ['bar_al','k_512','cp512_k', 'm_homo_k', 'r_hetero_k', 'i_hier_k']
	list_of_noise = ['0.5','0.75','1.0','1.5','2.0','2.5']
	
	for file_prefix1, file_prefix2 in zip(list_of_file_prefix1, list_of_file_prefix2) :
		for suffix in list_of_noise: 
			src_file = src_path + file_prefix1 + '.noise.'+suffix
			dest_file = dest_path + file_prefix2 + '_tr_frac_'+suffix
			data_preprocess_baselines( src_file, dest_file, w_v[file_prefix2]['w'])

def preprocess_synthetic_files_vary_train():
	src_path = '../various_results/result_synthetic_dataset_old/'
	dest_path  = '../Synthetic/vary_train/data/'
	w_v=load_data('../Synthetic/vary_train/w_v_synthetic')
	# print w_v
	# return
	list_of_file_prefix1 = ['barabasi_albert_500_5_2', 'kronecker_512', 'kroneckerCP512', 'kroneckerHeterophily512', \
	'kroneckerHier512', 'kroneckerHomophily512' ]
	list_of_file_prefix2 = ['bar_al','k_512','cp512_k', 'r_hetero_k', 'i_hier_k','m_homo_k' ]
	list_of_tr_frac = ['25','50','100','200','400']
	

	for file_prefix1, file_prefix2 in zip(list_of_file_prefix1, list_of_file_prefix2 ) :
		for suffix in list_of_tr_frac:
			print file_prefix1
			print suffix  
			print w_v[file_prefix2]['w']
			src_file = src_path + file_prefix1 + '_msg_'+str(int(suffix)*500)+'_data'
			dest_file = dest_path + file_prefix2 + '_tr_frac_'+suffix
			# data_preprocess_baselines( src_file, dest_file, w_v[file_prefix2]['w'])
			data_obj = load_data(src_file)
			# print  data_obj['0.5'].keys()
			# # print data_obj[keys[0]].keys()
			# # M = data_obj.edges
			# # N=M.T 
			# # print np.sum( np.sum(M-N) )
			data_dict = load_data(dest_file)
			param_dict = {'A':data_obj.A, 'alpha':data_obj.alpha}
			data_dict['param']=param_dict
			save( data_dict, dest_file)


def main():

	path = '../Real_Data/'
	list_of_file_prefix = ['barca', 'british_election', 'GTwitter','jaya_verdict', \
		'JuvTwitter',  'MsmallTwitter',  \
			 'Twitter' , 'VTwitter']  # 'real_vs_ju_703','MlargeTwitter', 'trump_data',
	list_of_baselines = ['huber_regression', 'robust_lasso', 'soft_thresholding']
	list_of_tr_frac = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	# file_prefix, baseline = parse_command_line_input( list_of_file_prefix, list_of_baselines )ststs
	#--------------------------------------

	#---------Synthetic vary noise ---------

	#--------------------------------------
	# preprocess_synthetic_files_vary_noise()
	# return 
	# list_of_file_prefix=['bar_al', 'k_512', 'cp512_k', 'm_homo_k', 'r_hetero_k', 'i_hier_k']
	# list_of_noise = ['0.5','0.75','1.0','1.5','2.0','2.5']
	# path = '../Synthetic/vary_noise/'
	# vary_train( path, list_of_file_prefix, list_of_baselines, list_of_noise, flag_selected=False)
	# return 
	
	#--------------------------------------

	#---------Synthetic vary train----------

	#--------------------------------------
	# preprocess_synthetic_files_vary_train()
	# return 
	list_of_file_prefix=['bar_al', 'k_512', 'cp512_k', 'm_homo_k', 'r_hetero_k', 'i_hier_k']
	list_of_tr_frac = ['25','50','100','200','400']
	path = '../Synthetic/vary_train/'

	file_prefix, baseline = parse_command_line_input( list_of_file_prefix, list_of_baselines )
	list_of_file_prefix=[ file_prefix ]
	vary_train( path, list_of_file_prefix, list_of_baselines, list_of_tr_frac, flag_selected=False)
	return 
	#----------------------------------------

	#-------Vary Train Set-------------------

	#----------------------------------------
	# preprocess_for_vary_train(path , list_of_file_prefix, list_of_tr_frac)
	vary_train( path + 'vary_train/', list_of_file_prefix, list_of_baselines, list_of_tr_frac)
	return 
	#----------------------------------------

	#-------SANITIZATION---------------------

	#----------------------------------------
	# preprocess_for_sanitize_test( path , list_of_file_prefix)
	# sanitize_test( path, list_of_file_prefix, list_of_baselines)
	# return 
	#----------------------------------------

	#-------SUBSET SELECTION BASELINE EXP1,2-

	#----------------------------------------
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
