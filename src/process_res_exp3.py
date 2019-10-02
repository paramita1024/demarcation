import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

def retrieve_results_exp1( path , list_of_file_prefix ):
    src_dir = '../result_performance_forecasting_pkl/'
    dest_dir = path + 'exp1/res.'
    for file_prefix in list_of_file_prefix:
		res = load_data( src_dir + file_prefix + '.selected')
		save( res, dest_dir + file_prefix )
    
class exp1_plots:

	def __init__( self , path, list_of_file_prefix, list_of_baselines, param  ):
		
		self.path = path
		self.list_of_file_prefix = list_of_file_prefix 
		self.list_of_baselines = list_of_baselines
		self.param = param

	def merge_baselines( self ):
		for file_prefix in self.list_of_file_prefix:
			if file_prefix[0] in ['G', 'j', 'J', 'Ms', 'T']:
				res_slant_file = self.path + 'baselines_slant/exp1/res.'+file_prefix
				res_slant = load_data( res_slant_file)
				res_subset = load_data( self.path + 'baselines/res.'+file_prefix)
				for baseline in self.list_of_baselines:
					for reg1 in self.param['set_of_lamb_w']:
						if 'h' == baseline[0]:
							reg2 = self.param['set_of_epsilon'][0]
						else:
							reg2 = self.param['set_of_lamb_e'][0]
						dest_res_file = self.path + 'baselines_slant_short/exp1/res.' \
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
			res = load_data( self.path + 'exp1/res.' + file_prefix )
			# res_baseline = load_data( self.path + 'baselines_slant/exp1/res.'+file_prefix)
			# data_arr = np.vstack((res['y-axis']['MSE'][:6,:], res['y-axis']['MSE'][9:,:]))
			# for baseline in self.list_of_baselines:
			# 	lamb_arr = res['lambda']['hawkes'][baseline]
			# 	res_arr = []
			# 	for l,t in zip(lamb_arr, self.param['list_of_time_span']):
			# 		res_arr.append(res_baseline[baseline][str(l)][reg2(baseline)]['slant']\
			# 			['0.8']['prediction'][str(t)]['MSE'])
			# 	data_arr = np.vstack(( data_arr, np.array(res_arr)))
			# 	res['labels'].append(baseline)
			# res['y-axis']['MSE']=data_arr
			res['labels']=['h_s','h_c','h_R_c','p_s','p_c','p_R_c','h_r','r_l','s_t']
			# res['labels']=res['labels'][:6].extend( res['labels'][9:])
			save( res, self.path + 'exp1/res.' + file_prefix )

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

	def plot_exp1( self):
		
		for file_prefix in self.list_of_file_prefix:
			
			print '\\begin{figure}[H]'
			res = load_data( self.path + 'exp1/res.' + file_prefix )
			image_path  = self.path +  'exp1/plots/'
			image_path_latex = '../exp1/plots/'
			# print res['labels']
			# return
			for key in ['MSE']:#, 'FR']:
				for smooth_flag in ['smooth','no_smooth']:	
					for arr,l in zip(res['y-axis'][key], res['labels']) :
						l_list = l.split('_')
						if smooth_flag=='smooth':
							plt.plot( self.smooth(arr) , label = '_'.join( [ l_word[0] for l_word in l_list] ) )#, \
								# linewidth=8,linestyle='--',marker='o', markersize=10 )
						else:
							plt.plot( arr , label = '_'.join( [ l_word[0] for l_word in l_list] ) )#, \
								# linewidth=8,linestyle='--',marker='o', markersize=10 )
					plt.legend()
					plt.grid()
					plt.savefig( image_path + file_prefix + '_'+ key + '_'+smooth_flag+'.pdf' )
					self.print_figure_singleton( image_path_latex + file_prefix + '_' + key + '_'+smooth_flag+'.pdf', key )
					plt.close()
			print '\\caption{',file_prefix.replace('_','-'),'}'
			print '\\end{figure}'
  
	def plot_exp1_with_baselines_all( self ):

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

		file_prefix_0 = input('file_prefix')
		for file_prefix in self.list_of_file_prefix:
			if file_prefix[0] == file_prefix_0:
				# print file_prefix
				res_file = self.path + 'exp1/res.' + file_prefix
				res_baseline_file = self.path + 'baselines_slant/exp1/res.' + file_prefix
				res = load_data( res_file  )
				image_path  = self.path +  'baselines/plots/'
				image_path_latex = '../baselines/plots/'

				for key in ['MSE'] : #, 'FR']:	

					for baseline in self.list_of_baselines:
						res_baseline = load_data( res_baseline_file )[ baseline ]
						outer_keys, inner_keys = map_params( baseline )

						for arr,l in zip(res['y-axis'][key], res['labels']) :
							l_list = l.split('_')
							plt.plot( self.smooth(arr) , label = '_'.join( [ l_word[0] for l_word in l_list] ))# , \
								# linewidth=8,linestyle='--',marker='o', markersize=10 )

						for outer_key in outer_keys:
							for inner_key in inner_keys:

								val = [ res_baseline[ str(outer_key) ][ str(inner_key) \
									]['slant']['0.8']['prediction'][str(t)]['MSE'] for t in self.param['list_of_time_span'] ]
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
						plt.show()
						print_figure_singleton( image_path_latex + plot_file, file_prefix  + '  ' + baseline + ' ' + key )
						plt.close() 
						# return  

	def set_baseline_param( self ):
		# print '\\begin{figure}[H]'
		# file_prefix='barca'
		# res = load_data( self.path + 'exp1/res.' + file_prefix )
		# print res['lambda']['hawkes'].keys()
		f= input("file")
		for file in self.list_of_file_prefix:
			if file.startswith(f):
				filename = file
		res_file =  self.path + 'exp1/res.' + filename
		res = load_data(res_file)
		param = res['lambda']['hawkes']
		for baseline in self.list_of_baselines:
			if baseline in param:
				print param[baseline]
		
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
					if flag=='y':
						index  = input('index')
						lamb = input('lamb')
						param[baseline][int(index)]=float(lamb)
					else:
						break
			flag = input('Exit')
			if flag == 'y':
				break

		res['lambda']['hawkes']=param

		for baseline in self.list_of_baselines:
			if baseline in param:
				print param[baseline]

		save( res, res_file)
				
def main():

	path = '../Real_Data/'
	list_of_file_prefix = ['barca', 'british_election', 'GTwitter','jaya_verdict', \
		'JuvTwitter',  'MsmallTwitter',  \
			 'Twitter' , 'VTwitter']  # 'MlargeTwitter', 'trump_data', 'real_vs_ju_703',
	list_of_baselines = ['huber_regression','robust_lasso',  'soft_thresholding'] #['huber_regression' ,  'filtering', 'dense_err',
	# file_prefix, baseline = parse_command_line_input( list_of_file_prefix, list_of_baselines )
	param = {}
	# huber 
	arr = [0.5,0.6]#, 0.8, 1.0, 1.2]
	param['set_of_alpha']=arr # 0.0001, 0.0005, 0.005, 0.01,
	param['set_of_epsilon']=[  1.5 ]	
	# soft thresholding,  extended robust lasso, 
	param['set_of_lamb_w'] =arr # [  0.005]#, 0.01, 0.05, 0.1, 0.5 ] # 0.0001, 0.0005,
	param['set_of_lamb_e'] = [ 0.5 ]
	param['list_of_time_span'] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
	obj = exp3_plots( path , list_of_file_prefix, list_of_baselines, param)
	# obj.set_baseline_param()
	# obj.merge_selected_baseline_with_final_res()
	# obj.plot_exp1_with_baselines_all()
	# obj.merge_baselines()
	# obj.plot_exp1( )
	# retrieve_results_exp1( path , list_of_file_prefix )
	# merge( path + 'res.' , 'huber_regression' , list_of_file_prefix )	
	# ------------------------------------------------ 
	# for file_prefix in list_of_file_prefix:
	# 	start = time.time()
	# 	subset_selection_baselines( path , file_prefix, baseline, param )
	# 	end = time.time()
	# 	print 'File:', file_prefix, ' , Baseline: ' , baseline, ', Time:' , end-start, ' seconds'
	#--------------------------------------------------
	# w_v_dict = load_data('w_v')
	# preprocess( list_of_file_prefix , path , w_v_dict )


if __name__ == "__main__":
	main()
			
