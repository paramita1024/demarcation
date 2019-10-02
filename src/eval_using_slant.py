import numpy as np
import datetime
import os 
import sys
import matplotlib.pyplot as plt
import numpy.random as rnd
import time
from slant import slant 
# from data_preprocess import *
from myutil import *

def eval_using_slant_Robust_cherrypick( file_prefix, time_span_input_list,list_of_lambda = None,  num_simulation=1 ):
	# time_span_input_slant = # ***************************
	# print ' file name ' + file_prefix 
	method = 'Robust_cherrypick'
	path = '../Cherrypick_others/Data_opn_dyn_python/'
	file_suffix = '_10ALLXContainedOpinionX.obj'

	directory = '../result'
	if not os.path.exists(directory):
		os.makedirs(directory)
	file_to_read_obj = path + file_prefix + file_suffix 
	file_to_read_result = '../result/' + file_prefix + '.res.' + method
	file_to_write_result = '../result/' + file_prefix + '.res.' + method +'.slant'

	obj = load_data( file_to_read_obj) 
	result = load_data( file_to_read_result)
	full_train =  np.copy( obj.train )
	
	info_details = result[-1]
	# result  = [result[ fraction_index ]]
	result_list = []
	num_frac = len( result) -1 
	for res_obj, frac_index in zip(result[:-1], range(num_frac)) : 
		result_list_frac = []
		for res_lamb in res_obj :
			# print 'LAMBDA: ' + str(lamb)
			frac = res_obj_lamb['frac_end']
			end_msg = res_obj_lamb['data']
			lamb = res_obj_lamb['lambda']
			result_list_lamb = []
			obj.train = full_train[ end_msg ]
			start = time.time()
			slant_obj = slant( obj=obj, init_by='object' , data_type = 'real', tuning = True, tuning_param = [4,2,lamb] )
			print 'NUM_TRAIN ' + str( slant_obj.train.shape[0])
			slant_obj.estimate_param()
			slant_obj.set_train( full_train )
			print 'SLANT TRAIN TIME : ' + str( time.time() - start )
			# for each item of time span input list 
			print 'NUM_TRAIN ' + str( slant_obj.train.shape[0])
			for time_span_input in time_span_input_list :
				start = time.time()
				result_obj = slant_obj.predict( num_simulation = num_simulation , time_span_input = time_span_input )
				result_obj['time_span_input'] = time_span_input 
				result_obj['lambda'] = lamb
				result_obj['frac_end'] = frac 
				result_obj['name'] = file_prefix
				print 'Time Span:' + str( time_span_input ) + '		Prediction time :' + str( time.time() - start ) + ', Current TIME: ' + str( datetime.datetime.now() )
				result_list_lamb.append( result_obj )
			result_list_frac.append( result_list_lamb)
		result_list.append( result_list_frac )
		save( result_list, file_to_write_result + '.fraction.' + str(frac_index))
	result_list.append( info_details )
	save( result_list , file_to_write_result )
def eval_using_slant_Robust_cherrypick_tuning( file_prefix, num_simulation=1 ):
	print '******************************************************************************'
	print 'Eval slant is not correctly passed arguemets. It is changed. check '
	print '******************************************************************************'
	# time_span_input_slant = # ***************************
	# print ' file name ' + file_prefix 
	method = 'Robust_cherrypick'
	path = '../Cherrypick_others/Data_opn_dyn_python/'
	file_suffix = '_10ALLXContainedOpinionX.obj'
	w,v=10,10
	directory = '../result'
	if not os.path.exists(directory):
		os.makedirs(directory)
	file_to_read_obj = path + file_prefix + file_suffix 
	file_to_read_result = '../result/' + file_prefix + '.res.' + method
	file_to_write_result = '../result/' + file_prefix + '.res.' + method +'.slant'

	obj = load_data( file_to_read_obj) 
	result = load_data( file_to_read_result)
	full_train =  np.copy( obj.train )
	info_details = result[-1]
	num_frac = len( result)-1 
	if num_frac <> 4:
		print 'Error'
	list_of_result_frac = []
	for res_obj in result[:-1]: 
		list_of_MSE=[]
		list_of_slant_results=[]
		for res_lamb in res_obj:
			frac = res_obj_lamb['frac_end']
			end_msg = res_obj_lamb['data']
			lamb = res_obj_lamb['lambda']
			time_span_input_list=np.zeros(1)
			res_slant=eval_slant( obj, full_train, end_msg, time_span_input_list,1,w,v,lamb)[0]
			list_of_MSE.append(res_slant['MSE'])
			list_of_slant_results.append(res_slant)
		l_idx=np.argmin(np.array(list_of_MSE))
		lamb_best=info_details['set_lambda'][l_idx]
		end_msg=res_obj[l_idx]['data']
		frac_end=res_obj[l_idx]['frac_end']
		time_span_input_list=np.linspace(.1,.5,5)
		list_of_result=eval_slant( obj, full_train, end_msg, time_span_input_list,num_simulation,w,v,lamb_best,frac)
		list_of_result.insert(0,list_of_slant_results[l_idx])
		list_of_result_frac.append(list_of_result)
	save( list_of_result_frac, file_to_write_result)
	# # result_list_lamb = []
	# obj.train = full_train[ end_msg ]
	# start = time.time()
	# slant_obj = slant( obj=obj, init_by='object' , data_type = 'real', tuning = True, tuning_param = [w,v,lamb] )
	# print 'NUM_TRAIN ' + str( slant_obj.train.shape[0])
	# slant_obj.estimate_param()
	# slant_obj.set_train( full_train )
	# result_obj = slant_obj.predict( num_simulation = 1 , time_span_input = time_span_input )
	# list_of_MSE.append(result_obj['MSE'])
	# 	obj.train = full_train[ end_msg ]
	# 	start = time.time()
	# 	slant_obj = slant( obj=obj, init_by='object' , data_type = 'real', tuning = True, tuning_param = [w,v,lamb] )
	# 	print 'NUM_TRAIN ' + str( slant_obj.train.shape[0])
	# 	slant_obj.estimate_param()
	# 	slant_obj.set_train( full_train )
	# 	res_list_tm_span=[]
	# 	for time_span_input in time_span_input_list[1:] :
	# 		start = time.time()
	# 		result_obj = slant_obj.predict( num_simulation=num_simulation , time_span_input = time_span_input )
	# 		result_obj['time_span_input'] = time_span_input 
	# 		result_obj['lambda'] = lamb
	# 		result_obj['frac_end'] = frac 
	# 		result_obj['name'] = file_prefix
	# 		print 'Time Span:' + str( time_span_input ) + '		Prediction time :' + str( time.time() - start ) + ', Current TIME: ' + str( datetime.datetime.now() )
	# 		result_list_tm_span.append( result_obj )
	# 	result_list_frac.append( result_list_tm_span)
	# result_list_frac.append( info_details)
	# save( result_list_frac, file_to_write_result)


def eval_using_slant_cherrypick( file_prefix, time_span_input_list,list_of_lambda = None, num_simulation=1):
	path = '../Cherrypick_others/Data_opn_dyn_python/'
	file_suffix = '_10ALLXContainedOpinionX.obj'
	method = 'cherrypick'
	
	
	directory = '../result'
	if not os.path.exists(directory):
		os.makedirs(directory)

	file_to_read_obj = path + file_prefix + file_suffix 
	file_to_read_result = '../result/' + file_prefix + '.res.' + method
	file_to_write_result = '../result/' + file_prefix + '.res.' + method +'.slant'

	obj = load_data( file_to_read_obj) 
	result = load_data( file_to_read_result)
	full_train =  np.copy( obj.train )
	num_frac = len( result) -1
	num_lambda_cherrypick = len( result[0] )


	info_details = result[-1]
	result_list = []
	for res_frac in result[:-1] : 
		result_list_frac = []
		for res_lamb in res_frac:
			frac = res_lamb['frac_end']
			end_msg = res_lamb['data']
			lambda_cherrypick = res_lamb['lambda']
			result_list_lamb_cherrypick = []
			for lambda_slant in list_of_lambda : 
				obj.train = full_train[ end_msg ]
				start = time.time()
				slant_obj = slant( obj=obj, init_by='object' , data_type = 'real', tuning = True, tuning_param = [4,2,lambda_slant] )
				print 'NUM_TRAIN ' + str( slant_obj.train.shape[0])
				slant_obj.estimate_param()
				slant_obj.set_train( full_train )
				print 'NUM_TRAIN ' + str( slant_obj.train.shape[0])
				print 'SLANT TRAIN TIME : ' + str( time.time() - start )
				result_list_lamb_slant = []
				for time_span_input in time_span_input_list :
					start = time.time()
					result_obj = slant_obj.predict( num_simulation = num_simulation , time_span_input = time_span_input )
					result_obj['time_span_input'] = time_span_input 
					result_obj['lambda_cherrypick'] = lambda_cherrypick
					result_obj['lambda_slant'] = lambda_slant
					result_obj['frac_end'] = frac 
					result_obj['name'] = file_prefix
					print 'Prediction time :' + str( time.time() - start ) + ', Current Time: ' + str( datetime.datetime.now() )
					print 'FRAC ' + str( frac )
					print 'l_cpk ' + str(lambda_cherrypick)
					print 'l_slant ' + str(lambda_slant)
					print 'time ' +  str(time_span_input)
					print 'sigma' + str(res_lamb['sigma_covariance'])
					result_list_lamb_slant.append( result_obj )
				result_list_lamb_cherrypick.append( result_list_lamb_slant)
				# save( result_list_lamb_cherrypick , file_to_write_result + '.fraction.' + str( frac_index) + '.lambda.' + str( lambda_index_cherrypick ) )
			result_list_frac.append( result_list_lamb_cherrypick)
			save( result_list_frac , file_to_write_result + '.fraction' + str( frac ) )
		result_list.append( result_list_frac )
	result_list.append( info_details )
	save( result_list , file_to_write_result )

		
	
def eval_using_slant_cherrypick_tuning( file_prefix, time_span_input_list,list_of_lambda = None, num_simulation=1):
	path = '../Cherrypick_others/Data_opn_dyn_python/'
	file_suffix = '_10ALLXContainedOpinionX.obj'
	method = 'cherrypick'
	
	
	directory = '../result'
	if not os.path.exists(directory):
		os.makedirs(directory)

	file_to_read_obj = path + file_prefix + file_suffix 
	file_to_read_result = '../result/' + file_prefix + '.res.' + method
	file_to_write_result = '../result/' + file_prefix + '.res.' + method +'.slant'

	obj = load_data( file_to_read_obj) 
	result = load_data( file_to_read_result)
	full_train =  np.copy( obj.train )
	num_frac = len( result) -1
	num_lambda_cherrypick = len( result[0] )


	info_details = result[-1]
	result_list = []
	for res_frac in result[:-1] : 
		result_list_frac = []
		for res_lamb in res_frac:
			frac = res_lamb['frac_end']
			end_msg = res_lamb['data']
			lambda_cherrypick = res_lamb['lambda']
			result_list_lamb_cherrypick = []
			for lambda_slant in list_of_lambda : 
				obj.train = full_train[ end_msg ]
				start = time.time()
				slant_obj = slant( obj=obj, init_by='object' , data_type = 'real', tuning = True, tuning_param = [4,2,lambda_slant] )
				print 'NUM_TRAIN ' + str( slant_obj.train.shape[0])
				slant_obj.estimate_param()
				slant_obj.set_train( full_train )
				print 'NUM_TRAIN ' + str( slant_obj.train.shape[0])
				print 'SLANT TRAIN TIME : ' + str( time.time() - start )
				result_list_lamb_slant = []
				for time_span_input in time_span_input_list :
					start = time.time()
					result_obj = slant_obj.predict( num_simulation = num_simulation , time_span_input = time_span_input )
					result_obj['time_span_input'] = time_span_input 
					result_obj['lambda_cherrypick'] = lambda_cherrypick
					result_obj['lambda_slant'] = lambda_slant
					result_obj['frac_end'] = frac 
					result_obj['name'] = file_prefix
					print 'Prediction time :' + str( time.time() - start ) + ', Current Time: ' + str( datetime.datetime.now() )
					print 'FRAC ' + str( frac )
					print 'l_cpk ' + str(lambda_cherrypick)
					print 'l_slant ' + str(lambda_slant)
					print 'time ' +  str(time_span_input)
					print 'sigma' + str(res_lamb['sigma_covariance'])
					result_list_lamb_slant.append( result_obj )
				result_list_lamb_cherrypick.append( result_list_lamb_slant)
				# save( result_list_lamb_cherrypick , file_to_write_result + '.fraction.' + str( frac_index) + '.lambda.' + str( lambda_index_cherrypick ) )
			result_list_frac.append( result_list_lamb_cherrypick)
			save( result_list_frac , file_to_write_result + '.fraction' + str( frac ) )
		result_list.append( result_list_frac )
	result_list.append( info_details )
	save( result_list , file_to_write_result )
def merge_over_fractns( file_prefix, start_index, end_index , method ):
	file_to_read_prefix = '../result/' + file_prefix + '.res.' + method + '.' 
	file_to_write = '../result/' + file_prefix + '.res.' + method + '.slant'
	result_list_outer = [ ]
	for index in range( start_index, end_index+1):
		file_to_read = file_to_read_prefix + str( index ) + '.slant'
		result = load_data( file_to_read )
		print 'len of res is ' + str( len(result) )
		print 'len of each sublist is ' + str( len( result[0]) ) 
		result_list_outer.append( result[0] )
		if index == end_index:
			result_list_outer.append( result[ -1 ] )
	save( result_list_outer, file_to_write )
def merge_subset_selection_results( file_prefix, method, num_fraction, num_lambda ):
	file_to_read_prefix = '../result_with_lambda/slant/' + file_prefix + '.res.' + method  
	file_to_write = '../result_with_lambda/slant/' + file_prefix + '.res.' + method + '.slant'
	result_list_frac = [ ]
	for fraction_index in range( num_fraction ):
		result_list_lambda=[]
		for lambda_index in range( num_lambda ):
			file_to_read = file_to_read_prefix + '.fraction.'+ str( fraction_index ) + '.lambda.' + str( lambda_index ) + '.slant'
			result = load_data( file_to_read )
			# print 'number of frac  ' + str( len(result) )
			# print 'number of lamb ' + str( len( result[0]) )
			# print 'number of tm span ' + str( len( result[0][0])) 
			
			# print result[0][0][0]['lambda']
			result_list_lambda.append( result[0][0] )
		result_list_frac.append( result_list_lambda )
	save( result_list_frac, file_to_write )
def merge_slant_results( file_prefix , number_of_w ):
	file_to_write = '../slant_result_merged/' + file_prefix + 'res.slant.tuning' 
	result_list_outer = [ ]
	for i in range(1,3,1):
		file_to_read = '../slant_result_merged/file' + str(i) + '/' + file_prefix + 'res.slant.tuning' 	
		result_list = load_data( file_to_read )
		for res_obj in result_list:
			result_list_outer.append( res_obj )


	save( result_list_outer, file_to_write )

	
def shuffle_results( file_prefix, num_frac  ):
	file_to_read ='../result_cherrypick/set_selection/before_shuffle/'+file_prefix + '.res.cherrypick'
	file_to_save='../result_cherrypick/set_selection/after_shuffle/'+file_prefix+'.res.cherrypick'
	list_of_result_input = load_data( file_to_read)
	list_of_result=[]
	for fraction in range(num_frac) : 
		list_of_result.append([])
	for res_lamb in list_of_result_input[:-1]:
		for res_frac, list_res_frac in zip(res_lamb, list_of_result):
			list_res_frac.append( res_frac )
	list_of_result.append( list_of_result_input[-1])
	save( list_of_result, file_to_save)

def get_subset_of_set_selection_results( file_prefix, f_index_set, l_index_set):
	file_to_read ='../result_cherrypick/set_selection/after_shuffle/'+file_prefix + '.res.cherrypick'
	file_to_save='../result_cherrypick/set_selection/subset_of_results/'+file_prefix+'.res.cherrypick'
	res_list = load_data( file_to_read)
	outer_list=[]
	for f_ind in f_index_set:
		res_obj = res_list[f_ind]
		inner_list=[]
		for l_index in l_index_set:
			inner_list.append( res_obj[l_index] )
		outer_list.append(inner_list)
	outer_list.append( res_list[-1])	
	save( outer_list, file_to_save)
def merge_over_lambda_fractions( flist, lambda_list, file_prefix, file_to_save ):
	outer_list =[]
	for f in flist:
		inner_list = []
		for lamb in lambda_list:
			result = load_data( file_prefix + '.fraction.' +str( f ) + '.lambda.' + str( lamb) )
			inner_list.append(result)
		outer_list.append(inner_list)
	info_details = {}
	outer_list.append( info_details )
	save(outer_list, file_to_save)

def remove_suffix(folder):

	for directory in os.listdir( folder ):
		print directory
		for files in os.listdir( folder + '/'+directory):
			new_file = files.split('.')[0].split('_')[0] + '.png'
			print files
			print new_file
			os.rename( folder + '/' + directory + '/' + files, folder + '/' + directory + '/' + new_file)
def process_slant_msg(file):
	result = load_data( file )
	num_test = result[0]['true_target'].shape[0]
	num_time_span = len( result)
	# print num_time_span
	# print num_test
	num_msg=np.zeros( (num_time_span, num_test) )
	for res_tm_span, time_index in zip(result, range(num_time_span)):

		msg_set = res_tm_span['msg_set']
		
		for msg_set_index in range(num_test):
			msg_set_simul = msg_set[msg_set_index]
			# num_simul = res_tm_span['predicted'].shape[1]
			msg_set_instance = msg_set_simul[0]
			num_msg[ time_index , msg_set_index ] = msg_set_instance.shape[0]
	print 'plot start'
	# f  = plt.figure()
	plt.plot( np.mean(num_msg, axis =  0))
	plt.savefig( file + '.eps')
	# plt.clf()
	print 'plot finished'
	# for time_index in range( num_time_span):
		# plot

def eval_slant( obj, full_train, end_msg, time_span_input_list,num_simulation,w,v,lamb,frac,file_prefix,file_suffix,method,int_generator) :
	obj.train=full_train[end_msg]
	slant_obj = slant( obj=obj, init_by='object' , data_type = 'real', tuning = True, tuning_param = [w,v,lamb], int_generator=int_generator )
	print 'NUM_TRAIN ' + str( slant_obj.train.shape[0])
	start=time.time()
	slant_obj.estimate_param()
	print 'SLANT TRAIN TIME : ' + str( time.time() - start )
	slant_obj.set_train( full_train )
	# result_list_tm_span=[]
	for time_span_input in time_span_input_list:
		start = time.time()
		if time_span_input==0:
			result_obj = slant_obj.predict( num_simulation=1, time_span_input = time_span_input )
		else:
			result_obj = slant_obj.predict( num_simulation=num_simulation, time_span_input = time_span_input )
		print 'Time Span:' + str( time_span_input ) + '		Prediction time :' + str( time.time() - start ) + ', Current TIME: ' + str( datetime.datetime.now() )
		
		result_obj['time_span_input'] = time_span_input 
		result_obj['lambda'] = lamb
		result_obj['frac_end'] = frac
		msg =dict( result_obj['msg_set'])
		del result_obj['msg_set']
		if method==0:
			file_to_write=file_prefix+'w'+str(w)+'v'+str(v)+'f'+str(frac)+'lc'+str(lamb)+'ls'+str(lamb)+'t'+str(time_span_input)+file_suffix
		else:
			file_to_write=file_prefix+'w'+str(w)+'v'+str(v)+'f'+str(frac)+'ls'+str(lamb)+'t'+str(time_span_input)+file_suffix
		save(result_obj, file_to_write)
		save(msg,file_to_write+'.msg')
		# result_obj['name'] = file_prefix
		# save msg separate
		
		# result_list_tm_span.append(result_obj)
	
def eval_using_slant_cherrypick_manual_tuning( file_prefix, time_span_input_list,list_of_fractions,list_of_lambdas, num_simulation,w,v,int_generator='Hawkes'):
	file_to_read_obj = '../Cherrypick_others/Data_opn_dyn_python/'+file_prefix+'_10ALLXContainedOpinionX.obj'
	obj = load_data( file_to_read_obj) 
	full_train =  np.copy( obj.train )

	file_to_write_prefix = '../result_subset_selection_slant/' + file_prefix 
	file_to_write_suffix='.res.cherrypick.slant'
	if int_generator=='Poisson':
		file_to_write_suffix+='.Poisson'
				
	for frac in list_of_fractions:
		for lamb in list_of_lambdas:
			# file_to_read='../result_subset_selection/'+file_prefix+'w10f'+str(frac)+'l'+str(lamb)+'.res.cherrypick'
			file_to_read='../result_subset_selection/'+file_prefix+'w'+str(w)+'f'+str(frac)+'l'+str(lamb)+'.res.cherrypick'
			result_obj=load_data(file_to_read)
			end_msg=result_obj['data']
			eval_slant(obj, full_train, end_msg, time_span_input_list,num_simulation,w,v,lamb,frac,file_to_write_prefix,file_to_write_suffix,0,int_generator)

def eval_using_slant_Robust_cherrypick_manual_tuning( file_prefix, time_span_input_list,list_of_fractions,list_of_lambdas, num_simulation,w,v,int_generator='Hawkes'):
	file_to_read_obj = '../Cherrypick_others/Data_opn_dyn_python/'+file_prefix+'_10ALLXContainedOpinionX.obj'
	obj = load_data( file_to_read_obj) 
	full_train =  np.copy( obj.train )
	
	file_to_write_suffix='.res.Robust_cherrypick.slant'
	if int_generator=='Poisson':
		file_to_write_suffix+='.Poisson'
	file_to_write_prefix = '../result_subset_selection_slant/' + file_prefix 

	for frac in list_of_fractions:
		for lamb in list_of_lambdas:
			file_to_read='../result_subset_selection/'+file_prefix+'w'+str(w)+'f'+str(frac)+'l'+str(lamb)+'.res.Robust_cherrypick'
			result_obj=load_data(file_to_read)
			end_msg=result_obj['data']
			eval_slant(obj, full_train, end_msg, time_span_input_list,num_simulation,w,v,lamb,frac,file_to_write_prefix,file_to_write_suffix,1,int_generator)

		
def eval_using_slant_manual_tuning(file_prefix,list_of_lambdas,list_of_time_span,num_simulation,w,v,int_generator='Hawkes'):
	# w=10
	# v=10
	lamb=list_of_lambdas[0]

	file_to_read_obj = '../Cherrypick_others/Data_opn_dyn_python/'+file_prefix+'_10ALLXContainedOpinionX.obj'
	obj = load_data( file_to_read_obj) 
	slant_obj = slant( obj=obj, init_by='object' , data_type = 'real', tuning = True, tuning_param = [w,v,lamb], int_generator=int_generator )
	file_to_write_prefix = '../result_subset_selection_slant/' + file_prefix 
	start=time.time()
	slant_obj.estimate_param()
	print 'SLANT TRAIN TIME : ' + str( time.time() - start )
	for time_span_input in list_of_time_span:
		start = time.time()
		if time_span_input==0:
			result_obj = slant_obj.predict( num_simulation=1, time_span_input = time_span_input )
		else:
			result_obj = slant_obj.predict( num_simulation=num_simulation, time_span_input = time_span_input )
		print 'Time Span:' + str( time_span_input ) + '		Prediction time :' + str( time.time() - start ) + ', Current TIME: ' + str( datetime.datetime.now() )
		result_obj['time_span_input'] = time_span_input 
		result_obj['lambda'] = lamb
		result_obj['frac_end'] = 1
		msg =dict( result_obj['msg_set'])
		del result_obj['msg_set']
		file_to_write=file_to_write_prefix+'w'+str(w)+'v'+str(v)+'l'+str(lamb)+'t'+str(time_span_input)+'.res.slant'
		if int_generator=='Poisson':
			file_to_write+='.Poisson'
		save(result_obj, file_to_write)
		save(msg,file_to_write+'.msg')
def main():
	#   WARNING : DO NOT USE EVAL ---------------------------
	# time_span_input_list =np.array([0,.1,.2,.3,.4,.5]) # ,.6,.7,.8,.9,1.])#np.array([0.2])#
	file_prefix_list =   ['barca','british_election','GTwitter','jaya_verdict', 'JuvTwitter' , 'MlargeTwitter','MsmallTwitter', 'real_vs_ju_703', 'trump_data' ,'Twitter','VTwitter','food','reddit']
	eval_slant =  False
	merge_files_over_fractns = False # True
	merge_slant_result_files = False # True
	merge_subset_sel_results = False # True 
	shuffle_result = False # True
	get_subset_of_set_selection_result = False # True
	merge_result_over_fraction_lamda = False # True # 
	remove_suffix_flag=False #True
	process_slant_msg_flag = False # True
	eval_slant_tuning=False#True
	eval_slant_for_manual_tuning=False
	eval_slant_combined=True#False# True#True
	# eval_slant_full_train_data_poisson=False # True
	# data = rnd.rand(8,3)
	# # print data.shape
	# print_for_kile( data, row_titles = file_prefix_list , column_titles = [ 'slant', 'cherrypick','Robust_cherrypick'] )

	if eval_slant_combined:
		file_index = int(sys.argv[1])
		method_index= int(sys.argv[2])
		# list_of_lambdas=[float(sys.argv[3])]
		list_of_lambdas=[float(sys.argv[3])]
		list_of_int_generators=['Hawkes','Poisson']
		int_generator=list_of_int_generators[int(sys.argv[4])]
		# l_idx=int(sys.argv[3])  #  -1#
		# list_of_lambda_full =[.4,.6]#[.2,.5,.7,.9,2] # [.5,.7,.9,1,2]#.5,.7,.9,1.1]#[.6,.7,.8,.9,1.1]# [.01,.05,.1,.5,1] #[ 1e-02, 1e-01, .5, 1, 10 ]
		# if l_idx==-1:
		# 	print 'slant is not ready for this option'
		# 	list_of_lambdas=list_of_lambda_full
		# else:
		# 	list_of_lambdas=[list_of_lambda_full[l_idx]]
		time_span_input_list=np.array([ 0.0])#,.1,.2,.3,.4,.5])#np.array([ .2])#
		list_of_fractions=[.8]#[.5,.6,.7,.9]#[.8]
		num_simulation=20

		# w=10
		# v=10

		# flag_tune_kernel_param=int(sys.argv[5])
		# if flag_tune_kernel_param:
		w=int(sys.argv[5])
		v=int(sys.argv[6])
		# int_generator='Poisson'
		list_of_methods = ['cherrypick' , 'Robust_cherrypick','slant'] 
		
		print 'File:' + file_prefix_list[ file_index ] + ', Method:' + list_of_methods[method_index]
		print('lambda',list_of_lambdas)
		start=time.time()
		for file_prefix in [file_prefix_list[ file_index ]] :
			if method_index == 0 :
				eval_using_slant_cherrypick_manual_tuning( file_prefix, time_span_input_list,list_of_fractions,list_of_lambdas, num_simulation,w,v,int_generator)
			if method_index==1:
				eval_using_slant_Robust_cherrypick_manual_tuning(file_prefix, time_span_input_list,list_of_fractions,list_of_lambdas, num_simulation,w,v,int_generator)
			if method_index==2:
				eval_using_slant_manual_tuning(file_prefix,list_of_lambdas,time_span_input_list,num_simulation,w,v,int_generator)				
		
		print 'Evaluation Done in ', str(time.time()-start),' seconds'


	


	# if eval_slant_full_train_data_poisson:
	# 	file=file_prefix_list[int(sys.argv[1])]
	# 	lamb=list([.1,.2,.3])[int(sys.argv[3])]# # list([.5,.7,.9,1,2])[int(sys.argv[2])]
	# 	list_of_time_span= np.array([.1,.2,.3,.4,.5])
	# 	num_simulation=20
	# 	int_generator='Poisson'
	# 	eval_using_slant_manual_tuning(file,lamb,list_of_time_span,num_simulation,int_generator)

	if merge_subset_sel_results :
		file_prefix_list.remove('trump_data')
		file_prefix_list.remove('MsmallTwitter')
		for file_prefix in file_prefix_list:
			merge_subset_selection_results( file_prefix, method = 'Robust_cherrypick', num_fraction = 5 , num_lambda = 5 ) 
	if merge_slant_result_files:
		number_of_w = 3 
		for file_prefix in file_prefix_list : 
			merge_slant_results( file_prefix , number_of_w )
	if merge_files_over_fractns : 
		for file_prefix in file_prefix_list :
			merge_over_fractns( file_prefix , start_index =  0 , end_index = 4  , method = 'Robust_cherrypick')
		
	if eval_slant:

		#!!!!!!!!!!!!!!!!!!!!!   LIST of lambdas and fraction has been updated. Please update them again !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

		
		list_of_lambda = [.01,.05,.1,.5,1] #[ 1e-02, 1e-01, .5, 1, 10 ]
		# list_of_lambda_cherrypick = [.5,1 ] # [.01, .1, .5, 1, 4, 6, 10]
		list_of_methods = ['cherrypick' , 'Robust_cherrypick'] 
		# list_of_fraction = np.array([.8]) # np.linspace(.6, 1, 5)
		file_index = int(sys.argv[1])
		method_index= int( sys.argv[2])
		# fraction_index = int(sys.argv[2])
		# lambda_index = int(sys.argv[3])
		
		print 'File:' + file_prefix_list[ file_index ] + ', Method:' + list_of_methods[method_index]
		# print 'Fraction : ' + str( list_of_fraction) # + ', LAMBDA: ' + str( list_of_lambda) 
		# time_index= 0 # int(sys.argv[5])
		
		for file_prefix in [file_prefix_list[ file_index ]] :
			if method_index == 0 :
				eval_using_slant_cherrypick(file_prefix , list_of_time_span, list_of_lambda = list_of_lambda , num_simulation = 10 )
			else:
				eval_using_slant_Robust_cherrypick(file_prefix,list_of_time_span, list_of_lambda=list_of_lambda,num_simulation = 10)
		print 'Evaluation Done'
	if eval_slant_tuning:
		list_of_lambda = [.01,.05,.1,.5,1] #[ 1e-02, 1e-01, .5, 1, 10 ]
		list_of_time_span=np.linspace(0,.5,6)
		# list_of_lambda_cherrypick = [.5,1 ] # [.01, .1, .5, 1, 4, 6, 10]
		list_of_methods = ['cherrypick' , 'Robust_cherrypick'] 
		# list_of_fraction = np.array([.8]) # np.linspace(.6, 1, 5)
		file_index = int(sys.argv[1])
		method_index= int( sys.argv[2])
		# fraction_index = int(sys.argv[2])
		# lambda_index = int(sys.argv[3])
		
		print 'File:' + file_prefix_list[ file_index ] + ', Method:' + list_of_methods[method_index]
		# print 'Fraction : ' + str( list_of_fraction) # + ', LAMBDA: ' + str( list_of_lambda) 
		# time_index= 0 # int(sys.argv[5])
		
		for file_prefix in [file_prefix_list[ file_index ]] :
			if method_index == 0 :
				eval_using_slant_cherrypick_tuning(file_prefix , list_of_time_span, list_of_lambda = list_of_lambda , num_simulation = 20 )
			else:
				eval_using_slant_Robust_cherrypick_tuning(file_prefix,list_of_time_span, list_of_lambda=list_of_lambda,num_simulation = 20)
		print 'Evaluation Done'
	if shuffle_result: 
		# convert a list of lambda over fraction to fraction over lambda list
		# for cherrypick subset selection result
		print 'shuffling results'
		file_index_set=[5,8] # [2,6,9] # [0,1,3,4,7,10]
		# print file_index_set.astype(int).shape
		for index in file_index_set:
			shuffle_results( file_prefix_list[index], num_frac = 5)
	if get_subset_of_set_selection_result:
		# from full result set, retreive result for only a subset
		# for cherrypick subset selection result
		print 'get subset of results'
		file_index_set=[5,8] # [2,6,9]# [0,1,3,4,7,10]
		f_index_set= [2,3]
		l_index_set=[2,3]
		# print file_index_set.astype(int).shape
		for index in file_index_set:
			get_subset_of_set_selection_results( file_prefix_list[index], f_index_set, l_index_set)
	if merge_result_over_fraction_lamda :
		# merging result of robust cherrypick
		file_index_set = [5,10]
		fraction_list = [2,3,4]
		lambda_list = [2,3]
		directory_to_load = '../result_robust_cherrypick/after_slant/splitted_result/'
		directory_to_save = '../result_robust_cherrypick/after_slant/subset_of_result/'
		for index in file_index_set:
			file_prefix = directory_to_load + file_prefix_list[index] + '.res.Robust_cherrypick.slant'
			file_to_save = directory_to_save + file_prefix_list[index] + '.res.Robust_cherrypick.slant'
			merge_over_lambda_fractions( fraction_list, lambda_list, file_prefix, file_to_save )
	if remove_suffix_flag:
		remove_suffix( '../Plots/Plots_with_lambda/Time_vs_MSE/Fig')
	if process_slant_msg_flag:
		process_slant_msg( '../result/MsmallTwitter.res.slant')
	
if __name__=='__main__':
	main()



# def plot_slant_results_nowcasting( list_of_files , list_of_methods ): 
# 	MSE = np.zeros(( len( list_of_files) , len( list_of_methods ) ) ) 
# 	FR = np.zeros(( len( list_of_files) , len( list_of_methods ) ) ) 
	
# 	data = {}# []
# 	for row, file_prefix in zip( range(len(list_of_files)) , list_of_files):
# 		for col , method in zip( range(len(list_of_methods)), list_of_methods ): 
# 			print method
# 			if method == 'slant':
# 				file_to_read = '../result/' + method + '/' + file_prefix + '.res.' + method + '.nowcasting'
# 				result = load_data( file_to_read )[0]
# 			else:
# 				file_to_read = '../result/' + method + '/' + file_prefix + '.res.' + method + '.slant.nowcasting'
# 				result = load_data( file_to_read )[0][0]
			
# 			data[method] = np.mean(result['predicted'], axis = 1)
# 			if method == 'slant':
# 				data['actual']=result['true_target']
# 			MSE[ row, col ] = result['MSE']
# 			FR[ row, col ] = result['FR']
# 		f = plt.figure()
# 		plt.plot( data['actual'], label='true opinion')
# 		plt.plot( data['slant'], label='slant')
# 		plt.plot( data['cherrypick'], label = 'cherrypick')
# 		plt.plot( data['Robust_cherrypick'], label='Robust_cherrypick')
# 		plt.legend( )
# 		plt.title('Dataset : ' + file_prefix + ' opinion for first 400-500 samples in test set ')
# 		plt.xlim((400,500))
# 		# print label
# 		# plt.show()
# 		plt.savefig( file_prefix + '_opinion for first 400-500 samples in test set .png')
# 		plt.clf()


		# for row, method in zip(data_to_plot,list_of_methods):
		# 	plt.plot( row, label = method )

	# print data['slant'].shape
	# print data['cherrypick'].shape
	# print data['Robust_cherrypick'].shape
	
	# print_for_kile( MSE, list_of_files, list_of_methods)
	# print_for_kile( FR, list_of_files, list_of_methods)

	#----------------------------------------------------------
	# for MSE_row , file_prefix in zip( MSE , list_of_files ):
	# 	f=  plt.figure()
	# 	plt.plot( MSE_row )
	# 	plt.title('MSE for dataset : '+file_prefix )
	# 	plt.savefig('MSE.'+file_prefix+'.png')
	# 	plt.clf()

	

	# for FR_row , file_prefix in zip( FR , list_of_files ):
	# 	f=  plt.figure()
	# 	plt.plot( FR_row )
	# 	plt.title('FR for dataset : '+file_prefix )
	# 	plt.savefig('FR.'+file_prefix+'.png')
	# 	plt.clf()
	#----------------------------------------------------------





			# print type(result)
			# plt.plot( result['predicted'], label=method)
			# if method=='slant':
				
			# 	plt.plot( result['true_target'], label='actual')
			#--------------------------------------------------------
			# if method=='slant':
			# 	s=plt.plot( result['predicted'], 'c')
			# 	t=plt.plot( result['true_target'], 'b')
			# if method == 'cherrypick':
			# 	cpk=plt.plot(result['predicted'], 'r' )
			# if method == 'Robust_cherrypick':
			# 	rcpk=plt.plot(result['predicted'], 'g' )
			#--------------------------------------------------------
			# print 'MSE' + str(result['MSE'])
			# print 'FR' + str(result['FR'])
			
			# data_to_plot.append( result['predicted'])
			# if method == 'Robust_cherrypick':
			# 	data_to_plot.append( result['true_target'])











	# label_list = [ 'time_span 0' , 'time span .1', 'time span .2' , 'time_span .3', 'time span .4', 'time span .5']
	# num_plot = num_time_span
	# title_txt = file_prefix 
	# xtitle = 'Fraction of endogenious msg ' # set such that fractions are shown here 
	# ytitle = 'MSE error'
	# image_title = file_prefix + '.MSE.cherrypick.png'
	# plot_result( MSE , label_list , num_plot, title_txt , xtitle, ytitle , image_title )


	# label_list = [ 'time_span 0' , 'time span .1', 'time span .2' , 'time_span .3', 'time span .4', 'time span .5']
	# num_plot = num_time_span
	# title_txt = file_prefix 
	# xtitle = 'Fraction of endogenious msg ' # set such that fractions are shown here 
	# ytitle = 'FR error'
	# image_title = file_prefix + '.FR.cherrypick.png'
	# plot_result(FR , label_list , num_plot, title_txt , xtitle, ytitle , image_title )



# def eval_using_slant_Robust_cherrypick( file_prefix, time_span_input_list,list_of_lambda = None,  num_simulation=1 ):
# 	# time_span_input_slant = # ***************************
# 	# print ' file name ' + file_prefix 
# 	method = 'Robust_cherrypick'
# 	path = '../Cherrypick_others/Data_opn_dyn_python/'
# 	file_suffix = '_10ALLXContainedOpinionX.obj'

# 	directory = '../result'
# 	if not os.path.exists(directory):
# 		os.makedirs(directory)
# 	file_to_read_obj = path + file_prefix + file_suffix 
# 	file_to_read_result = '../result/' + file_prefix + '.res.' + method
# 	file_to_write_result = '../result/' + file_prefix + '.res.' + method +'.slant'

# 	obj = load_data( file_to_read_obj) 
# 	result = load_data( file_to_read_result)
# 	full_train =  np.copy( obj.train )
	
# 	info_details = result[-1]
# 	# result  = [result[ fraction_index ]]
# 	result_list = []
# 	for res_obj in result[:-1] : 
# 		result_list_frac = []
# 		for lamb in list_of_lambda :
# 			# print 'LAMBDA: ' + str(lamb)
# 			if method == 'Robust_cherrypick':
# 				for res_obj_lamb in res_obj:
# 					# print 'type of result_object lamb' 
# 					# print res_obj_lamb

# 					if res_obj_lamb['lambda'] == lamb : 
# 						frac = res_obj_lamb['frac_end']
# 						print 'FRACtion: ' + str(frac)
# 						print 'LAMBDA: ' + str(res_obj_lamb['lambda'])
# 						end_msg = res_obj_lamb['data']
# 						break
# 			result_list_lamb = []
# 			# print ' FRACTION: ' + str(frac )
# 			# print obj.train.shape
# 			obj.train = full_train[ end_msg ]
# 			del end_msg
# 			start = time.time()
# 			slant_obj = slant( obj=obj, init_by='object' , data_type = 'real', tuning = True, tuning_param = [4,2,lamb] )
# 			print 'NUM_TRAIN ' + str( slant_obj.train.shape[0])
# 			slant_obj.estimate_param()
# 			slant_obj.set_train( full_train )
# 			print 'SLANT TRAIN TIME : ' + str( time.time() - start )
# 			# for each item of time span input list 
# 			print 'Num_TRAIN ' + str( slant_obj.train.shape[0])
			
# 			for time_span_input in time_span_input_list :
# 				start = time.time()
# 				result_obj = slant_obj.predict( num_simulation = num_simulation , time_span_input = time_span_input )
# 				result_obj['time_span_input'] = time_span_input 
# 				result_obj['lambda'] = lamb
# 				# result_obj['frac_end'] = frac 
# 				result_obj['name'] = file_prefix
# 				print 'Time Span:' + str( time_span_input ) + '		Prediction time :' + str( time.time() - start ) + ', Current TIME: ' + str( datetime.datetime.now() )
# 				result_list_lamb.append( result_obj )

# 			result_list_frac.append( result_list_lamb)
# 			save( result_list_frac, file_to_write_result + '.lamb.tmp')
# 		result_list.append( result_list_frac )
# 	result_list.append( info_details )
# 	save( result_list , file_to_write_result )
# if eval_slant_poisson:
# 		file_index = int(sys.argv[1])
# 		method_index= int(sys.argv[2])
# 		l_idx=int(sys.argv[3])  #  -1#
# 		list_of_lambda_full =[.1,.2] # [.5,.7,.9,1,2]#.5,.7,.9,1.1]#[.6,.7,.8,.9,1.1]# [.01,.05,.1,.5,1] #[ 1e-02, 1e-01, .5, 1, 10 ]
# 		if l_idx==-1:
# 			list_of_lambdas=list_of_lambda_full
# 		else:
# 			list_of_lambdas=[list_of_lambda_full[l_idx]]
# 		time_span_input_list=np.array([ 0,.1,.2,.3,.4,.5])
# 		list_of_fractions=[.8]
# 		num_simulation=20
# 		w=10
# 		v=10
# 		int_generator='Poisson'
# 		list_of_methods = ['cherrypick' , 'Robust_cherrypick','slant'] 
# 		print 'File:' + file_prefix_list[ file_index ] + ', Method:' + list_of_methods[method_index]
# 		print('lambda',list_of_lambdas)
# 		for file_prefix in [file_prefix_list[ file_index ]] :
# 			if method_index == 0 :
# 				eval_using_slant_cherrypick_manual_tuning( file_prefix, time_span_input_list,list_of_fractions,list_of_lambdas, num_simulation,w,v,int_generator)
# 			if method_index==1:
# 				eval_using_slant_Robust_cherrypick_manual_tuning(file_prefix, time_span_input_list,list_of_fractions,list_of_lambdas, num_simulation,w,v,int_generator)
# 			if method_index==2:
# 				eval_using_slant_manual_tuning(file_prefix,list_of_lambdas[0],time_span_input_list,num_simulation,w,v,int_generator)				
# 		print 'Evaluation Done'


# 	if eval_slant_for_manual_tuning:
# 		list_of_lambda_full =[.2, .5,.7,.9,2]#[1.3]#.5,.7,.9,1.1]#[.6,.7,.8,.9,1.1]# [.01,.05,.1,.5,1] #[ 1e-02, 1e-01, .5, 1, 10 ]
# 		#**********************************************************
# 		l_idx=int(sys.argv[3])
# 		#*********************************************************
# 		if l_idx==-1:
# 			list_of_lambdas=list_of_lambda_full
# 		else:
# 			list_of_lambdas=[list_of_lambda_full[l_idx]]
# 		# list_of_fractions=[.5,.6,.7,.9]
# 		list_of_fractions=[.8]#[list_of_fractions[int(sys.argv[3])]]
# 		num_simulation=20
# 		int_generator='Hawkes'
# 		w=10
# 		v=10
# 		time_span_input_list =np.array([0,.1,.2,.3,.4,.5]) #
# 		# list_of_lambda_cherrypick = [.5,1 ] # [.01, .1, .5, 1, 4, 6, 10]
# 		list_of_methods = ['cherrypick' , 'Robust_cherrypick','slant'] 
# 		# list_of_fraction = np.array([.8]) # np.linspace(.6, 1, 5)
# 		file_index = int(sys.argv[1])
# 		method_index= int(sys.argv[2])
# 		# fraction_index = int(sys.argv[2])
# 		# lambda_index = int(sys.argv[3])
		
# 		print 'File:' + file_prefix_list[ file_index ] + ', Method:' + list_of_methods[method_index]
# 		# print 'Fraction : ' + str( list_of_fraction) # + ', LAMBDA: ' + str( list_of_lambda) 
# 		# time_index= 0 # int(sys.argv[5])
		
# 		for file_prefix in [file_prefix_list[ file_index ]] :
# 			if method_index == 0 :
# 				eval_using_slant_cherrypick_manual_tuning( file_prefix, time_span_input_list,list_of_fractions,list_of_lambdas, num_simulation,w,v)
# 			if method_index == 1 :
# 				eval_using_slant_Robust_cherrypick_manual_tuning(file_prefix, time_span_input_list,list_of_fractions,list_of_lambdas, num_simulation,w,v)
# 			if method_index == 2 :
# 				eval_using_slant_manual_tuning(file_prefix,list_of_lambdas[0], time_span_input_list,num_simulation,int_generator)
# 		print 'Evaluation Done'
