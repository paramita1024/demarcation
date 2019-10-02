import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import getopt
from sklearn.linear_model import Ridge, HuberRegressor, Lasso
import datetime
import time 
import numpy  as np
from numpy import linalg as LA
# from slant import slant
from myutil import * 
# from data_preprocess import *
import sys
# import Scipy
from numpy import linalg as LA
from baselines_class import *

class exp2_plots:

	def __init__( self , path, list_of_file_prefix, list_of_baselines, param  ):
		
		self.path = path
		self.list_of_file_prefix = list_of_file_prefix 
		self.list_of_baselines = list_of_baselines
		self.param = param

	def retrieve_results_exp2(self ):
		src_dir = '../result_variation_fraction_pkl/'
		dest_dir = self.path + 'exp2/res.'
		for file_prefix in self.list_of_file_prefix:
			res = load_data( src_dir + file_prefix + '.selected')
			save( res, dest_dir + file_prefix )

	def merge_baselines( self ):

		def get_reg2(baseline):
			if baseline == 'huber_regression':
				return '1.5'
			else:
				return '0.5'

		for file_prefix in self.list_of_file_prefix:
			if file_prefix[0] in ['V']: # ['j','M','T']:
				print file_prefix
				res_slant_file = self.path + 'baselines_slant/exp2/res.exp2.'+file_prefix
				res_slant = load_data( res_slant_file)
				res_subset = load_data( self.path + 'baselines/res.'+file_prefix)
				for baseline in self.list_of_baselines:
					print baseline
					# print res_subset[baseline].keys()

					if True: # 'huber' not in baseline :

						for reg1 in self.param['set_of_lamb_w']:
							reg2=get_reg2(baseline)
							dest_res_file = self.path + 'baselines_slant_short/exp2/res.exp2.' \
							+ file_prefix + '_'+ baseline +'_'+ str(reg1) + '_' + str(reg2)
							res_slant_part = load_data( dest_res_file,'ifexists')
							if res_slant_part:
								res_slant[baseline][str(reg1)]=res_subset[baseline][str(reg1)]
								res_slant[baseline][ str(reg1) ][ str(reg2) ]['slant']=res_slant_part
				save( res_slant, res_slant_file )

	def merge_selected_baseline_with_final_res( self ):

		def reg2(baseline):
			if baseline == 'huber_regression':
				return '1.5'
			else:
				return '0.5'

		for file_prefix in self.list_of_file_prefix:
			print file_prefix
			res = load_data( self.path + 'exp2/res.' + file_prefix )
			res_baseline = load_data( self.path + 'baselines_slant/exp2/res.exp2.'+file_prefix)
			res_lambda = load_data( self.path + 'exp1/res.'+file_prefix)
			# print res_lambda.keys()
			# return 
			data_arr = res['y-axis']['MSE']
			# data_arr = np.vstack((res['y-axis']['MSE'][:6,:], res['y-axis']['MSE'][9:,:]))
			for baseline in self.list_of_baselines:
				print '\t',baseline
				# print '*'*10, '\n',res_baseline[baseline].keys(),'\n','*'*10
				lamb_arr = [ res_lambda['lambda']['hawkes'][baseline][2] ]* len(self.param['list_of_frac'])
				# print lamb_arr
				res_arr = []
				for l,frac in zip(lamb_arr, self.param['list_of_frac']):
					# pass

					res_arr.append(res_baseline[baseline][str(l)][reg2(baseline)]['slant']\
						[str(frac)]['prediction'][str(0.2)]['MSE'])
				data_arr = np.vstack(( data_arr, np.array(res_arr)))
				data_arr[-1,-1]=data_arr[0,-1]
				res['labels'].append(baseline)
			res['y-axis']['MSE']=data_arr

			# res['labels']=['h_s','h_c','h_R_c','p_s','p_c','p_R_c','h_r','r_l','s_t']
			# res['labels']=res['labels'][:6].extend( res['labels'][9:])
			
			save( res, self.path + 'exp2/res.' + file_prefix )


	def merge_selected_baseline_with_final_res_FR( self ):

		def reg2(baseline):
			if baseline == 'huber_regression':
				return '1.5'
			else:
				return '0.5'


		def get_polarity(s,thres):
			polarity_s=np.zeros(s.shape[0])
			polarity_s[np.where(s>thres)[0]]=1
			polarity_s[np.where(s<-thres)[0]]=-1
			return polarity_s

		
		def compute_FR( result, threshold):
			# print result.keys()
			set_of_predictions= result['predicted']
			mean_prediction = np.mean( set_of_predictions, axis = 1 ) 
			true_values = result['true_target'] 
			num_test=true_values.shape[0]
			polar_pred=get_polarity(mean_prediction,threshold)
			polar_true=get_polarity(true_values,threshold)
			return float(np.count_nonzero(polar_true-polar_pred))/num_test

		for file_prefix in self.list_of_file_prefix:
			print file_prefix
			res = load_data( self.path + 'exp2/res.' + file_prefix )

			res_baseline = load_data( self.path + 'baselines_slant/exp2/res.exp2.'+file_prefix)
			res_lambda = load_data( self.path + 'exp1/res.'+file_prefix)
			threshold = res_lambda['threshold']

			# print res_lambda.keys()
			# return 
			data_arr = res['y-axis']['FR']
			# data_arr = np.vstack((res['y-axis']['MSE'][:6,:], res['y-axis']['MSE'][9:,:]))
			for baseline in self.list_of_baselines:
				print '\t',baseline
				# print '*'*10, '\n',res_baseline[baseline].keys(),'\n','*'*10
				lamb_arr = [ res_lambda['lambda']['hawkes'][baseline][2] ]* len(self.param['list_of_frac'])
				# print lamb_arr
				res_arr = []
				for l,frac in zip(lamb_arr, self.param['list_of_frac']):
					res_arr.append( compute_FR(res_baseline[baseline][str(l)][reg2(baseline)]['slant']\
						[str(frac)]['prediction'][str(0.2)],threshold))
				data_arr = np.vstack(( data_arr, np.array(res_arr)))
				data_arr[-1,-1]=data_arr[0,-1]
			res['y-axis']['FR']=data_arr
			
			save( res, self.path + 'exp2/res.' + file_prefix )



	def print_figure_singleton( self, filename, caption):

		print '\\begin{subfigure}{7cm}'
		print '\t \\centering\\includegraphics[width=6cm]{'+filename+'}'
		print '\t \\caption{'+caption+'}'
		print '\\end{subfigure}'

	def smooth( self, vec ):
		vec = np.array(vec)
		tmp = np.array( [ (.25*vec[ind]+.5*vec[ind-1]+.25*vec[ind+1]) for ind  in range(1,vec.shape[0]-1)  ])
		vec[0] = vec[0]*.75+vec[1]*.25
		vec[-1] = vec[-1]*.75+vec[-2]*.25
		vec[1:-1] = tmp
		return vec

	def plot_exp2( self):
		
		for file_prefix in self.list_of_file_prefix:
			
			print '\\begin{figure}[H]'
			res = load_data( self.path + 'exp2/res.' + file_prefix )
			# print res.keys()
			# print res['labels']
			# print res['x-axis']
			# print res['y-axis']['MSE']
			# return 
			image_path  = self.path +  'exp2/plots/'
			image_path_latex = '../exp2/plots/'

			for key in ['MSE', 'FR']:	
				for flag_smooth in ['smooth','no_smooth']:
					for arr,l in zip(res['y-axis'][key], res['labels']) :
						l_list = l.split('_')
						if flag_smooth =='smooth':
							plt.plot( self.smooth(arr) , label = '_'.join( [ l_word[0] for l_word in l_list] ) )#, \
								# linewidth=8,linestyle='--',marker='o', markersize=10 )
						else:
							plt.plot( arr , label = '_'.join( [ l_word[0] for l_word in l_list] ) )#, \
								# linewidth=8,linestyle='--',marker='o', markersize=10 )
					plt.legend()
					plt.grid()
					plt.savefig( image_path + file_prefix + '_'+ key + '_'+flag_smooth+'.pdf' )
					self.print_figure_singleton( image_path_latex + file_prefix + '_' + key+ '_'+flag_smooth+ '.pdf', key )
					plt.close()
			print '\\caption{',file_prefix.replace('_','-'),'}'
			print '\\end{figure}'

	def set_baseline_param( self ):
		# print '\\begin{figure}[H]'
		# file_prefix='barca'
		# res = load_data( self.path + 'exp1/res.' + file_prefix )
		# print res['lambda']['hawkes'].keys()
		f= input("file")
		for file in self.list_of_file_prefix:
			if file.startswith(f):
				filename = file
		res_file =  self.path + 'exp2/res.' + filename
		res = load_data(res_file)
		param = res['lambda']['hawkes']
		
		while True:
			b= input('baseline')	
			for b_elm in self.list_of_baselines:
				if b_elm.startswith(b):
					baseline = b_elm
			if baseline in param:
				print param[baseline]
			flag=input('init or update')
			if flag=='i':
				l = input('lamb')
				param[baseline]=6*[ l ]
			else:
				while True:
					flag = input('wish to alter intermediate point')
					if flag=='yes':
						index  = input('index')
						lamb = input('lamb')
						param[baseline][int(index)]=float(lamb)
					else:
						break
			flag = input('Exit')
			if flag == 'y':
				break

		res['lambda']['hawkes']=param
		save( res, res_file)
	
	def plot_exp2_with_baselines_all( self ):
        
		def print_figure_singleton( filename, caption):

			print '\\begin{figure}{15cm}'
			print '\t \\centering\\includegraphics[width=14cm]{'+filename+'}'
			print '\t \\caption{'+caption.replace('_', ' ')+'}'
			print '\\end{figure}'

		def map_params( baseline ):
			if baseline == 'huber_regression':
				outer_keys = self.param['set_of_alpha']
				inner_keys = self.param['set_of_epsilon']
			else:
				outer_keys = self.param['set_of_lamb_w']
				inner_keys = self.param['set_of_lamb_e']

			return outer_keys, inner_keys

		for file_prefix in self.list_of_file_prefix:
			if True:#  'V' in file_prefix[0]:
				# print file_prefix
				res_file = self.path + 'exp2/res.' + file_prefix
				res_baseline_file = self.path + 'baselines_slant/exp2/res.exp2.' + file_prefix
				res = load_data( res_file  )
				image_path  = self.path +  'baselines/plots/exp2/'
				image_path_latex = '../baselines/plots/exp2/'

				for key in ['MSE'] : #, 'FR']:	

					for baseline in self.list_of_baselines:
						# print baseline
						res_baseline = load_data( res_baseline_file )[ baseline ]
						outer_keys, inner_keys = map_params( baseline )

						for arr,l in zip(res['y-axis'][key], res['labels']) :
							l_list = l.split('_')
							plt.plot( arr , label = '_'.join( [ l_word[0] for l_word in l_list] ))# , \
							# linewidth=8,linestyle='--',marker='o', markersize=10 )

						for outer_key in outer_keys:
							for inner_key in inner_keys:
								if str(outer_key) in res_baseline:
									print True
									val = [ res_baseline[ str(outer_key) ][ str(inner_key) \
									]['slant'][str(f)]['prediction'][str(0.2)]['MSE'] \
									for f in self.param['list_of_frac'] ]
									# print val
									# val =  res_baseline[ str(outer_key) ][ str(inner_key) \
									# ]['slant']['0.8']['prediction']['0.0']['MSE']
									plt.plot( val, label = '_'.join( [ str(outer_key) , str(inner_key) ] ) )
									# plt.plot( self.smooth( val * np.ones( res['y-axis'][key].shape[1] )) \
									# , label = '_'.join( [ str(outer_key) , str(inner_key) ] ))# , \
									# linewidth=8,linestyle='--',marker='o', markersize=10 )

						plt.legend()
						plt.grid()
						plot_file = file_prefix + '_'+ baseline + '_' + key + '.pdf'
						plt.savefig( image_path + plot_file )
						# plt.show()
						print_figure_singleton( image_path_latex + plot_file, file_prefix  + '  ' + baseline + ' ' + key )
						plt.close() 
						# return  
	

def main():

	path = '../Real_Data/'
	list_of_file_prefix = ['barca', 'british_election', 'GTwitter','jaya_verdict', \
		'JuvTwitter',  'MsmallTwitter',  \
			 'Twitter' , 'VTwitter']  # 'MlargeTwitter', 'trump_data', 'real_vs_ju_703',
	list_of_baselines = ['huber_regression','robust_lasso',  'soft_thresholding'] #['huber_regression' ,  'filtering', 'dense_err',
	# file_prefix, baseline = parse_command_line_input( list_of_file_prefix, list_of_baselines )

	param = {}
	reg_arr=[.05,.1,.5,.6,.8,1.,1.2]
	# huber 
	param['set_of_alpha']=reg_arr #[    0.05, 0.1, 0.5] # 0.0001, 0.0005, 0.005, 0.01,
	param['set_of_epsilon']=[  1.5 ]
	# soft thresholding,  extended robust lasso, 
	param['set_of_lamb_w'] =reg_arr # [ 0.05, 0.1, 0.5]# [  0.005]#, 0.01, 0.05, 0.1, 0.5 ] # 0.0001, 0.0005,
	param['set_of_lamb_e'] = [ 0.5 ]
	param['list_of_time_span'] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
	param['list_of_frac'] = [0.5, 0.6, .7,.8,.9,1.0]
	obj = exp2_plots( path , list_of_file_prefix, list_of_baselines, param )
	# obj.merge_selected_baseline_with_final_res_FR()
	# obj.merge_baselines()
	# obj.plot_exp2_with_baselines_all(  )
	obj.plot_exp2()
	# obj.retrieve_results_exp2()
	# merge( path + 'res.' , 'huber_regression' , list_of_file_prefix )	
	# obj.merge_results_baselines()
	

if __name__ == "__main__":
	main()





# def parse_command_line_input( list_of_file_prefix, list_of_baselines):
	
# 	argv = sys.argv[1:]
# 	opts, args = getopt.getopt(argv, 'f:b:', ['file', 'baseline'])
	
# 	if len(opts) == 0 and len(opts) > 2:
# 		print ('usage: add.py -a <first_operand> -b <second_operand>')
# 	else:
# 		# Iterate the options and get the corresponding values
# 		for opt, arg in opts:
# 			# print opt,arg
# 			if opt == '-f':
# 				# print 'yes'
# 				for file_prefix in list_of_file_prefix:
# 					if file_prefix.startswith(arg):
# 						sel_file_prefix = file_prefix
						
# 			if opt == '-b':
# 				for baseline in list_of_baselines:
# 					if baseline.startswith(arg):
# 						sel_baseline = baseline
# 	return sel_file_prefix, sel_baseline 