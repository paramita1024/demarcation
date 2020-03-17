import getopt
import numpy as np
import datetime
import os 
import sys
import matplotlib.pyplot as plt
import numpy.random as rnd
import time
from slant import slant 
from myutil import *

def parse_command_line_input( list_of_file_name ):

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, 'l:f:r:m:', ['lamb','file_name', 'frac','method'])

    lamb=0.5
    file_name=''
    frac=0.8
    method = 'acpk'
    
    for opt, arg in opts:
        if opt == '-l':
            lamb = float(arg)
        if opt == '-f':
            for file_name_i in list_of_file_name:
            	if file_name_i.startswith( arg ):
            		file_name = file_name_i
        if opt == '-r':
        	frac = float(arg)

        if opt == '-m':
        	method = arg

    return file_name, lamb, frac , method



class eval_slant_module:
	
	def __init__(self, data, subset ):

		self.data = data
                subset.sort()
		self.subset = subset 
	
	def eval_slant(self, w, v, l, t, num_simul , int_gen):
		
		full_train =  np.copy( self.data['train'] )	
		
		self.data['train'] = self.data['train'][ self.subset ]
		
		slant_obj=slant( obj=self.data,init_by='dict',data_type='real',tuning_param=[ w, v, l],int_generator= int_gen)
		slant_obj.estimate_param()
		
		self.data['train'] = full_train
		
		if t==0:
			result_obj = slant_obj.predict( num_simulation=1, time_span_input = t )
		else:
			result_obj = slant_obj.predict( num_simulation=num_simul, time_span_input=t )

		return result_obj
			
	


def main():

	list_of_file_name = ['barca','british_election','GTwitter',\
	'jaya_verdict', 'JuvTwitter' , 'MlargeTwitter', \
	'MsmallTwitter', 'real_vs_ju_703', 'trump_data', 'Twitter' , 'VTwitter']

        file_name,lamb, frac, method = parse_command_line_input( list_of_file_name )
	
	list_of_time_span = [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5 ]
	int_gen='Hawkes'
	num_simul=20
	w=load_data('w_v')[file_name]['w']
	v=load_data('w_v')[file_name]['v']
	
	data_file = '../Data/' + file_name 
	data_all = load_data(data_file)
        #print(data_all['all_user'].keys())
        #eturn
        data = {'nodes': data_all['nodes'], 'edges': data_all['edges'] , 'train': data_all['all_user']['train'] , \
            'test': data_all['all_user']['test']  }

        subset_file = '../Result_Subset/' + file_name +  '.l' + str(lamb) + '.' + method 
        num_train_trunc = int(frac * data['train'].shape[0])
        subset  = load_data( subset_file)['data'].flatten()[ : num_train_trunc ]
        print(subset.shape)

	
	for t in list_of_time_span:

		start= time.time()
		obj=eval_slant_module( data, subset)
		res = obj.eval_slant( w, v, lamb, t, num_simul, int_gen )
		del res['msg_set']
		res_file = '../Result_Slant/' + file_name + '.fr' + str(frac) + '.l' + str(lamb) + '.t' + str(t) + '.' + method 
		save( res, res_file )

		print 'Slant evaluation for t =  ',t, ' required ', str(time.time()-start),' seconds'
		


if __name__=='__main__':
	main()
