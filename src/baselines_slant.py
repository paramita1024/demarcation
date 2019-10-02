import getopt
from sklearn.linear_model import Ridge, HuberRegressor, Lasso
import datetime
import time 
import numpy  as np
from numpy import linalg as LA
from myutil import * 
from data_preprocess import *
import sys
from numpy import linalg as LA
from baselines_class import *
import os 
import matplotlib.pyplot as plt
import numpy.random as rnd
from slant import slant 

def parse_command_line_input( list_of_file_prefix, list_of_baselines):

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, 'f:b:r:', ['file', 'baseline','reg'])

    if len(opts) == 0 and len(opts) > 3:
        print ('usage: add.py -a <first_operand> -b <second_operand> -c tmp  ')
    else:
        
        for opt, arg in opts:
            if opt == '-f':
                for file_prefix in list_of_file_prefix:
                    if file_prefix.startswith(arg):
                        sel_file_prefix = file_prefix
                        
            if opt == '-b':
                for baseline in list_of_baselines:
                    if baseline.startswith(arg):
                        sel_baseline = baseline

            if opt == '-r':
                reg = float( arg )

            #if opt == '-r2':
                #reg2 = float( arg )

    return sel_file_prefix, sel_baseline, reg

def merge( main_dir, baseline_to_add , list_of_file_prefix ):

    # for file_prefix in list_of_file_prefix:

    #     res = load_data( main_dir + file_prefix )
    #     res[ baseline_to_add ] = load_data( main_dir + baseline_to_add + '/' + file_prefix )[ baseline_to_add ]
    #     save( res, main_dir + file_prefix )


class baselines_slant:

    def __init__( self, data_file , param ):
        
        self.num_simul = 10
        self.list_of_frac = param[ 'list_of_frac']
        self.list_of_time_span = param['list_of_time_span']
        self.w = param['w']
        self.v = param['v']
        self.int_gen = param['int_gen']

        self.data = load_data( data_file )

    def predict_over_time(self,slant_obj):

        res_dict={}
        
        for time_span_input in self.list_of_time_span:
        
            start = time.time()
            if time_span_input==0:
                result_obj = slant_obj.predict( num_simulation=1, time_span_input = time_span_input )
            else:
                result_obj = slant_obj.predict( num_simulation=self.num_simul,time_span_input=time_span_input )
            print '----'
            print 'Time Span:' + str( time_span_input ) 
            print 'Prediction time :' + str( time.time() - start ) 
            print 'Current TIME: ' + str( datetime.datetime.now() )
            print '----'
            result_obj['time_span_input'] = time_span_input 
            res_dict[str(time_span_input)]=result_obj
        
        return res_dict

    def eval_slant(self, sorted_index, lamb):

        res={}
        for frac in self.list_of_frac:
        
            res[str(frac)]={}
            num_train = int( frac* self.data['all_user']['train'].shape[0] )
            train_idx = np.copy( sorted_index[:num_train] )
            train_idx.sort()
            data_dict = { 'edges': self.data['edges'], 'nodes': self.data['nodes'],\
                    'train': np.copy( self.data['all_user']['train'][  train_idx ,: ] ) , 'test':self.data['all_user']['test'] } 
            slant_obj = slant( init_by = 'dict', obj = data_dict, tuning_param = [ self.w, self.v, lamb] , \
                int_generator= self.int_gen )
            slant_obj.estimate_param()
            slant_obj.set_train( self.data['all_user']['train'] )
            res[str(frac)]['prediction']= self.predict_over_time(slant_obj)
            # print '----'
            # print 'Frac:' , frac
            # print 'Prediction time :' + str( time.time() - start ) 
            # print 'Current TIME: ' + str( datetime.datetime.now() )
            # # print '----'
        
        return res


def Eval_slant_for_baselines_individual( path , file_prefix, baseline, reg1, reg2, param):

    def add_res( res_part, res_dict ):
        if 'slant' not in res_part:
            res_part['slant']=res_dict
        else:
            for frac in param['slant']['list_of_frac']:
                res_part['slant'][str(frac)]=res_dict[str(frac)]
        return res_part
    
    data_file = path + 'data/' + file_prefix 
    obj = baselines_slant( data_file, param['slant'] )
    

    res_file  = path + 'baselines/res.' + file_prefix 
    dest_res_file = path + 'baselines_slant/res.' + file_prefix + '_'+ baseline +'_'+ str(reg1) + '_' + str(reg2)
    start= time.time()
    res_part = res[baseline][ str(reg1) ][ str(reg2) ]
    sorted_index = res_part['indices']
    res_dict = obj.eval_slant( sorted_index, reg1)
    res[baseline][ str(reg1) ][ str(reg2) ] = add_res( res_part, res_dict )
    print 'Regularizer 1 :', reg1, ',   Regularizer 2:', reg2, ',  Frac:', param['slant']['list_of_frac']
    end = time.time()
    
    print 'File:', file_prefix, ' , Baseline: ' , baseline, ', Time:' , end-start, ' seconds'
    
    save( res, dest_res_file )

def Eval_slant_for_baselines( path , file_prefix, baseline, param):

    def add_res( res_part, res_dict ):
        if 'slant' not in res_part:
            res_part['slant']=res_dict
        else:
            for frac in param['slant']['list_of_frac']:
                res_part['slant'][str(frac)]=res_dict[str(frac)]
                
        return res_part
    

    data_file = path + 'data/' + file_prefix 
    obj = baselines_slant( data_file, param['slant'] )
    
    res_file  = path + 'baselines/res.' + file_prefix 
    res = load_data( res_file )  
    for 
    # print (res['huber_regression']['0.5']['1.5']['slant'])[0.8]['prediction'].keys()
    # return 
    start= time.time()
    
    if baseline == 'huber_regression':
        for alpha in param['set_of_alpha']:
            for epsilon in param['set_of_epsilon']:

                res_part = res[baseline][ str(alpha) ][ str(epsilon) ]
                sorted_index = res_part['indices']
                res_dict = obj.eval_slant( sorted_index, alpha)
                res[ baseline ][ str( alpha )][str( epsilon )] = add_res( res_part, res_dict )
                print 'Alpha:', alpha, ',   Epsilon:', epsilon, ',  Frac:', param['slant']['list_of_frac']

    if baseline in ['robust_lasso', 'soft_thresholding' ]:
        for lamb_w in param['set_of_lamb_w']:
            for lamb_e in param['set_of_lamb_e']:

                res_part = res[baseline][ str(lamb_w) ][ str(lamb_e) ]
                sorted_index = res_part['indices']
                res_dict = obj.eval_slant( sorted_index, lamb_w)
                res[ baseline ][ str( lamb_w )][str( lamb_e )] = add_res( res_part, res_dict )
                print 'Lamb_w:', lamb_w, ',   Lamb_e:', lamb_e, ',  Frac:', param['slant']['list_of_frac']
    
    end = time.time()
    
    print 'File:', file_prefix, ' , Baseline: ' , baseline, ', Time:' , end-start, ' seconds'
    
    save( res, res_file )

def compress_params( list_of_file_prefix, list_of_baselines, path ):

    def np_array_to_dict( np_array ):

        d = {}
        for row in np_array:
            d[ str( row )] = { i: row[i] for i in np.flatnonzero( row )}
        return d 

    res_file_path = path + 'baselines/res.'
    for file_prefix in list_of_file_prefix:

        res_file = res_file_path + file_prefix
        res = load_data( res_file )
        for baseline in list_of_baselines:
            for outer_key in res[baseline].keys():
                for inner_key in res[baseline][outer_key].keys():
                    param_dict = res[baseline][outer_key][inner_key]['slant'][str(0.8)]['param']
                    param_dict['A'] = np_array_to_dict( param_dict['A'])
                    param_dict['B'] = np_array_to_dict( param_dict['B'])
                    res[baseline][outer_key][inner_key]['slant'][str(0.8)]['param'] = param_dict
        save( res, res_file)



def main():
    #---------------------
    path = '../Real_Data/'
    list_of_file_prefix = ['barca', 'british_election', 'GTwitter','jaya_verdict', \
        'JuvTwitter',  'MsmallTwitter', 'real_vs_ju_703', \
                'Twitter' , 'VTwitter']  
    list_of_baselines = ['huber_regression', 'robust_lasso',  'soft_thresholding'] # 'filtering', 'dense_err', 
    file_prefix, baseline, reg = parse_command_line_input( list_of_file_prefix, list_of_baselines )
    #-----------------------------
    param = {}
    if True:   
        # huber 
        param['set_of_alpha']=[0.6, 0.8, 1.0, 1.2]
        param['set_of_epsilon']=[ 1.5 ]
        # soft thresholding,  extended robust lasso, 
        param['set_of_lamb_w'] = [0.6, 0.8, 1.0, 1.2 ]
        param['set_of_lamb_e'] = [ .5 ]
        #-------------INPUT PARAMETERS--------------------------------------------------------------------------------
        param_slant={}
        param_slant['list_of_frac'] =np.array([.8])
        param_slant['int_gen'] =['Hawkes','Poisson'][0] # [int(sys.argv[4])]
        param_slant['list_of_time_span'] = np.array([ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5 ])
        param_slant['w'] = load_data('w_v')[file_prefix]['w']
        param_slant['v'] = load_data('w_v')[file_prefix]['v']
        param[ 'slant'] = param_slant
    if baseline == 'huber_regression':
        reg_o = 1.5
    else:
        reg_o = 0.5

    res_file  = path + 'baselines/res.' + file_prefix 
    res = load_data( res_file )  
    for baseline in list_of_baselines:
    	print res[baseline].keys()
    return 
    Eval_slant_for_baselines_individual( path, file_prefix, baseline, reg, reg_o, param)
    # compress_params( list_of_file_prefix, list_of_baselines, path )
    # for file_prefix in list_of_file_prefix:
    #     for baseline in list_of_baselines:
    #         Eval_slant_for_baselines( path, file_prefix, baseline, param)
    
    
if __name__=='__main__':
	main()


				
