import getopt 
import time
import copy
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from slant import *
from myutil import *


def parse_command_line_input( list_of_file_name ):

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, 'l:f:', ['lamb','file_name'])

    lamb=0.5
    file_name=''
    
    for opt, arg in opts:
        if opt == '-l':
            lamb = float(arg)
        if opt == '-f':
            for file_name_i in list_of_file_name:
            	if file_name_i.startswith( arg ):
            		file_name = file_name_i

    return file_name, lamb 

class exp1_cpk_variant_plot:
	
	def __init__(self, set_of_lamb, set_of_t ):

		self.set_of_lamb = set_of_lamb
		self.set_of_t = set_of_t
	
	def combine_results(self, src_res_file, dest_res_file, measure, method, threshold = 0 ):
		
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


		res = load_data( dest_res_file,'ifexists')
		for l in self.set_of_lamb:
			res_list = []
			for t in self.set_of_t:
				res_file =   src_res_file.split('.break')[0] + '.l' + str(l) + '.t' + str(t) + \
				src_res_file.split('.break')[1]
				res_part = load_data( res_file )
				if measure == 'MSE':
					res_list.append( res_part[measure] ) 
				if measure == 'FR':
					res_list.append( compute_FR(res_part, threshold ) )
			res[ str(l) ] = np.array( res_list )
		
		save( res, dest_res_file ) 

	def plot_res_part( self, res_file):

		res=  load_data( res_file )
		for l in self.set_of_lamb:

			plt.plot( res[ str(l)]  , label= str(l))
		plt.legend()
		plt.show()


	def plot_res_with_slant( self, slant_res_file , res_file ):

		slant_res = load_data( slant_res_file)
		for label in sorted(slant_res['y-axis']['MSE'].keys()):
			if not label.startswith('p_') and 'cpk' not in label:
				plt.plot( slant_res['y-axis']['MSE'][label], label = label, linestyle = '-' )

		res=  load_data( res_file )
		for l in self.set_of_lamb:
			plt.plot( res[ str(l)]  , label= str(l), linestyle = '--')
		plt.grid()
		plt.legend()
		# plt.ylim([0.08,.12])
		plt.savefig( res_file + '_slant_new_combined.pdf')
		plt.show()

		plt.grid(True)
		plt.close()


	def modify_res(self, slant_res_file):

		res = load_data( slant_res_file )
		print res['labels']
		res['old_y_axis']= copy.deepcopy( res['y-axis'])
		
		for measure in ['MSE','FR']:
			data = np.copy( res['y-axis'][measure])
			data_dict={}
			for l, d  in zip(res['labels'], data):
				data_dict[str(l)] = d
			res['y-axis'][measure]=data_dict
		new_file_name = slant_res_file.split('res.')[0]+'/'+slant_res_file.split('res.')[1]+'.res'
		save( res, new_file_name)


	def update_res( self, slant_res_file, param_file, res_file, method , measure):

		file_name = slant_res_file.split('/')[-1].split('.res')[0]
		with open(param_file, 'r') as fr:
			for line in fr:
				words = line.split(',')
				if file_name.startswith(words[0]):
					if len(words) > 2 :
						params = np.array([ float( value) for value in words[1:]])
					else:
						params = np.ones( len( self.set_of_t))*float( words[1])
						
		res = load_data(slant_res_file)
		all_res  = load_data(res_file)
		res_curr = np.array( [ all_res[ str(params[t]) ][t] for t in range(6) ] )
		res['y-axis'][measure][method]=res_curr
		if method not in res['labels']:
			res['labels'].append( method )
		# print res['lambda']['hawkes']['cherrypick']
		# print res['lambda']['hawkes']['Robust_cherrypick']
		# return 
		res['lambda']['hawkes'][method] = params
		save( res, slant_res_file )

	def plot_final_res( self, res_file, measure):

		def map_l(l):
			if 'cpk' in l:
				return l
			if l == 'h_R_c':
				return 'rcpk'
			if l == 'h_c':
				return 'dcpk'
			if l == 'h_r':
				return 'huber_reg'
			if l == 'h_s':
				return 'slant'
			if l == 'r_l':
				return 'robust_lasso'
			if l == 's_t':
				return 'soft_thres'
			
		# print load_data( res_file)['lambda']['hawkes']['acpk']
		file_name = res_file.split('/')[-1].split('.res')[0]
		res=  load_data( res_file )['y-axis'][measure]
		if file_name[0] == 'T':
			res['ecpk'] = res['tcpk']
		for l in sorted( res.keys() ):
			if not ( l.startswith('p_') ) :
				if not ( np.amax( res[l] ) == 0 and np.amin( res[l] ) == 0 ):
					if 'cpk' in map_l(l):
						plt.plot(res[l]  , label= map_l(l), linestyle='--')
					else:
						plt.plot(res[l]  , label= map_l(l), linestyle='-')
		# plt.legend()
		# plt.show()

		
		plt.title( file_name , rotation=0,fontsize=20.7)
		if file_name[:2] in {'ba','Ju'}:
			plt.ylabel(measure, rotation=90,fontsize=20.)
		plt.yticks(fontsize=20.7)
		plt.xticks(range(len(self.set_of_t)) , self.set_of_t, rotation=0, fontsize=20.7)
		plt.xlabel('Time Span ')
		plt.savefig('../../Writing/2018-demarcation-paramita/FIG_new/exp1_'+file_name+\
			'_'+measure+'.pdf')
		plt.close()

	def plot_legend( self, res_file, measure):

		def map_l(l):
			if 'cpk' in l:
				return l
			if l == 'h_R_c':
				return 'rcpk'
			if l == 'h_c':
				return 'dcpk'
			if l == 'h_r':
				return 'huber_reg'
			if l == 'h_s':
				return 'slant'
			if l == 'r_l':
				return 'robust_lasso'
			if l == 's_t':
				return 'soft_thres'
			
		# print load_data( res_file)['lambda']['hawkes']['acpk']
		res=  load_data( res_file )['y-axis'][measure]
		# print res.keys()
		# return 
		for l in sorted( res.keys() ):
			if not l.startswith('p_') :
				if not ( np.amax( res[l] ) == 0 and np.amin( res[l] ) == 0 ):
					if 'cpk' in map_l(l):
						plt.plot(res[l]  , label= map_l(l), linestyle='--')
					else:
						plt.plot(res[l]  , label= map_l(l), linestyle='-')
		leg=plt.legend(ncol=5,fontsize=9)
		leg.get_frame().set_visible(False)
		# plt.legend()
		plt.ylim([0,1])
		plt.tight_layout()
		file_name = res_file.split('/')[-1].split('.res')[0]
		plt.savefig('../../Writing/2018-demarcation-paramita/FIG_new/exp1_legend.pdf',bbox_inches='tight')
		plt.close()


	def update_FR(self,slant_res_file,src_res_file,threshold,method):

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


		res = load_data(slant_res_file)
		params=res['lambda']['hawkes'][method]		
		res_list = []
		for l,t in zip(params,self.set_of_t):
			res_file =   src_res_file.split('.break')[0] + '.l' + str(l) + '.t' + str(t) + \
				src_res_file.split('.break')[1]
			res_part = load_data( res_file,'ifexists' )
			if res_part:
				res_list.append( compute_FR(res_part, threshold ) )
		if res_list:
			res['y-axis']['FR'][method]=np.array( res_list)
		save( res, slant_res_file )

def main():
	
	set_of_file_name = ['barca','british_election','GTwitter',\
	'jaya_verdict', 'JuvTwitter' ,  \
	'MsmallTwitter',  'Twitter' , 'VTwitter']
        file_name,lamb = parse_command_line_input( set_of_file_name )
	set_of_lamb =[0.02,0.04,0.07,0.6,0.8,1.2,1.7]# [0.01,0.05,0.1,0.2,0.3,0.4,.5,.7,1.0,1.5,2.0]#[.5,.7,1.,1.5,2.]
	set_of_t = [ 0.0 , 0.1, 0.2, 0.3, 0.4, 0.5 ] 
	sns.set_style('dark')	
	
	for file_name in set_of_file_name:
		if file_name[0] in [ 'T']:#,'J','M', 'V']:
			for method in ['acpk']:#,'ecpk','tcpk']:#'acpk','ecpk',
				for measure in ['MSE','FR']:
					plot_obj = exp1_cpk_variant_plot( set_of_lamb, set_of_t ) 
					src_res  = '../Result_Slant/' + file_name + '.fr0.8.break.' + method   
					dest_res  = '../Result_Slant_Short/' + file_name + '.' + method + '.' + measure  
					slant_res = '../Result_Slant_Short/res.'+file_name
					slant_res_new = '../Result_Slant_Short/'+file_name + '.res'
					# print load_data(slant_res_new)['threshold']
					threshold = load_data(slant_res)['threshold']

					# plot_obj.combine_results( src_res, dest_res, measure, method, threshold )
					# plot_obj.plot_res_with_slant( slant_res_new, dest_res )
					# plot_obj.modify_res(slant_res)
					# plot_obj.update_res(slant_res_new, '../Result_Slant_Short/'+method+'.txt', dest_res, method, measure)
					plot_obj.plot_final_res(slant_res_new, measure)
					# plot_obj.update_FR(slant_res_new,src_res,threshold,method)
					# plot_obj.plot_legend( slant_res_new, measure)
					# return 
if __name__== "__main__":
  main()


