import copy
import sys
import time
from Robust_Cherrypick import Robust_Cherrypick 
from cherrypick import cherrypick
from math import sqrt
from myutil import *
import matplotlib.pyplot as plt
from slant import slant
from numpy import linalg as LA 
import numpy as np
import numpy.random as rnd
from math import ceil
import networkx as nx
import pickle
from synthetic_data import synthetic_data
		
def save(obj,output_file):
	with open(output_file+'.pkl' , 'wb') as f:
		pickle.dump( obj , f , pickle.HIGHEST_PROTOCOL)
		
def generate_graph(file_graph,generate_str,num_node=None,num_nbr=None,num_node_start=None,num_rep=None,plot_file=None,max_intensity=None):
	data = synthetic_data()
	# generate_graph( self,generate_str, num_nbr = 0 , num_node_start=0, num_rep=0):
	G_nx=data.generate_graph(generate_str,num_node,num_nbr,num_node_start,num_rep)
	# nx.draw(G_nx)
	# plt.draw()
	# # plt.title('GRAPH',fontsize=50)
	# plt.savefig(plot_file)
	# # plt.show()
	# plt.clf()
	
	data.generate_parameters(max_intensity=max_intensity)       ###########################
	save(data,file_graph)
	
def generate_predictions(msg_file,slant_file,set_of_noise,num_simul,set_of_method,set_of_lambda,result_file,set_of_lambda_cpk=None):
	res={}
	for noise in set_of_noise:
		res['n'+str(noise)]={}
		for sim_no in range(num_simul):
			suff='n0'+str(noise)+'sim'+str(sim_no)
			data=load_data(msg_file+suff)
			res['n'+str(noise)]['s'+str(sim_no)]={}
			for method in set_of_method:
				parameter_file=slant_file+suff+'.'+method
				param=load_data(parameter_file)
				
				# print '****************************************8'
				# for key in param:
				# 	print key
				# print '********************************************'

				res['n'+str(noise)]['s'+str(sim_no)]['m'+method]={}
				for l in set_of_lambda:
					print 'noise,sim,method,l',noise,sim_no,method,l
					if str(l) in param:
						res_curr = prediction_module(param[str(l)],data)
						res['n'+str(noise)]['s'+str(sim_no)]['m'+method]['l'+str(l)]=res_curr
					else:
						print 'key not found'
					if method=='cpk':
						if set_of_lambda_cpk!=None:
							for l_cpk in set_of_lambda_cpk:
								print 'noise,sim,method,l',noise,sim_no,method,l
								res['n'+str(noise)]['s'+str(sim_no)]['m'+method]['l'+str(l_cpk)]=prediction_module(param[str(l_cpk)],data)
	save(res, result_file)

def show_predictions(noise, set_of_noise,sim_no,result_file,set_of_method,l,set_of_lambda_cpk=None):
	res=load_data(result_file)
	# noise 
	# sim_no
	list_of_res=[]
	list_of_label=[]
	for method in set_of_method:
		# print res['n'+str(noise)]['s'+str(sim_no)]['m'+method]['l'+str(l)]['predicted'].shape
		list_of_res.append(res['n'+str(noise)]['s'+str(sim_no)]['m'+method]['l'+str(l)]['predicted'])
		list_of_label.append(method+' L='+str(l))
		if method=='cpk':
			if set_of_lambda_cpk!=None:
				for l_cpk in set_of_lambda_cpk:
					list_of_res.append(res['n'+str(noise)]['s'+str(sim_no)]['m'+method]['l'+str(l_cpk)]['predicted'])
					list_of_label.append(method+' L='+str(l_cpk))
	true=res['n'+str(noise)]['s'+str(sim_no)]['m'+method]['l'+str(l)]['true_target']
	list_of_label.append('true')
	# print true.ravel().shape
	list_of_res.append(true.reshape(true.shape[0],1))
	# print type()
	return {'xvalues':set_of_noise,'yvalues':np.array(list_of_res),'labels':list_of_label}

def get_prediction_result(result_file,set_of_method,set_of_noise,num_simul,l, result_MSE_values,set_of_lambda_cpk=None):
	res=load_data(result_file)
	res_MSE=[]
	list_of_label=[]

	if set_of_lambda_cpk==None:
		set_of_lambda_cpk=[l]
	for method in set_of_method:
		if method =='cpk':
			for l_cpk in set_of_lambda_cpk:
				res_method=[]
				for noise in set_of_noise:
					res_noise=[]
					for sim_no in range(num_simul):
						# print type(res['n'+str(noise)]['s'+str(sim_no)]['m'+method]['l'+str(l)])
						res_curr=res['n'+str(noise)]['s'+str(sim_no)]['m'+method]['l'+str(l_cpk)]['MSE']
						res_noise.append(res_curr)
					res_method.append(np.mean(np.array(res_noise)))
				res_MSE.append(res_method)
				list_of_label.append(method)#+' L='+str(l_cpk))
		else:
			res_method=[]
			for noise in set_of_noise:
				res_noise=[]
				for sim_no in range(num_simul):
					# print type(res['n'+str(noise)]['s'+str(sim_no)]['m'+method]['l'+str(l)])
					res_curr=res['n'+str(noise)]['s'+str(sim_no)]['m'+method]['l'+str(l)]['MSE']
					res_noise.append(res_curr)
				res_method.append(np.mean(np.array(res_noise)))
			res_MSE.append(res_method)
			list_of_label.append(method)#+' L='+str(l))
	save({'xvalue':set_of_noise,'yvalues':np.array(res_MSE),'labels':list_of_label}, result_MSE_values)

def show_difference(data_true,data_estm, file_write):
	print LA.norm(data_true-data_estm)
	plt.plot(data_true)
	plt.plot(data_estm)
	plt.show()
	plt.savefig(file_write)

def compare_it(true, estimated, file_write,generate_plot):
	if generate_plot:
		show_difference(true.alpha, estimated.alpha, file_write+'_alpha.jpg')
		show_difference(true.mu, estimated.mu, file_write+'_mu.jpg')
		index_A = np.where(np.flatten(true.A)!=0)[0][0]
		index_B = np.where(np.flatten(true.B)!=0)[0][0]
		show_difference(np.flatten(true.A)[index_A], np.flatten(estimated.A)[index_A],file_write+'_A.jpg')
		show_difference(np.flatten(true.A)[index_B], np.flatten(estimated.A)[index_B]+'_B.jpg')

def run_RCPK(obj,lamb,frac_end,w):
	start = time.time()
	Cherrypick_obj = Robust_Cherrypick( obj = obj , init_by = 'object', lamb = lamb, w_slant=w ) 
	Cherrypick_obj.initialize_data_structures()
	w, active_set, norm_of_residual = Cherrypick_obj.robust_regression_via_hard_threshold( method = 'FC', max_itr = 50 , frac_end = frac_end) 
	# file_to_write=file_to_write_prefix+'w10f'+str(frac_end)+'l'+str(lamb)+'.res.Robust_cherrypick.full'
	total_time = time.time()-start
	result=Cherrypick_obj.save_active_set( norm_of_residual = norm_of_residual  , total_time = total_time, )
	return result['data']

def run_CPK(obj,lamb,frac_end,var,w):
	cherrypick_obj = cherrypick( obj = obj , init_by = 'object', param ={'lambda':lamb,'sigma_covariance':var},w=w) 
	cherrypick_obj.demarkate_process(frac_end=frac_end)
	# file_to_write=file_to_write_prefix+'w10f'+str(frac_end)+'l'+str(lamb)+'.res.cherrypick'+'.full'
	result_obj = cherrypick_obj.save_end_msg( frac_end = frac_end ) 
	return result_obj['data']

def get_recovery_fraction(msg_end_true, msg_end_estm):
	return np.count_nonzero(np.logical_and(msg_end_true,msg_end_estm))/float(np.count_nonzero(msg_end_true))
def estimate_parameter_for_subset_selection(data,info):
	res=load_data(info['result_set_selection_file'])
	parameters={'slant':{},'cpk':{},'rcpk':{}}
	for l in info['set_of_lambda']:
		parameters['slant'][str(l)]=run_slant(data,l)
		parameters['cpk'][str(l)]=run_slant(data,res['cpk'][str(l)],l)
		parameters['rcpk'][str(l)]=run_slant(data,res['rcpk'][str(l)],l)
	save(parameters,info['result_estimated_parameters_file'])
			
def get_l2_distance(parameters,true_parameter,key_minor):
	list_main=[]
	for key in ['slant','cpk','rcpk']:
		list_new=[]
		value = parameters[key]
		for (key_inner,val_inner) in value:
			list_new.append(LA.norm(val_inner[key_minor],true_parameter[key_minor]))
		list_main.append(list_new)
	return list_main

def plot_array(val,title_str):
	plt.plot(val[0],label='sl')
	plt.plot(val[0],label='cpk')
	plt.plot(val[0],label='rcpk')
	plt.legend()
	plt.title(title_str)
	plt.savefig('../result_synthetic_data/Plots/'+title_str+'.jpg')

def plot_parameter_recovery_results(data,info):
	parameters=load_data(info['result_estimated_parameters_file'])
	true_parameter={'A':data.A, 'B':data.B, 'alpha':data.alpha, 'mu':data.mu}
	plot_array(get_l2_distance(parameters,true_parameter,'alpha'),'alpha')
	plot_array(get_l2_distance(parameters,true_parameter,'mu'),'mu')
	plot_array(get_l2_distance(parameters,true_parameter,'A'),'A')
	plot_array(get_l2_distance(parameters,true_parameter,'B'),'B')
# def plot_fraction_of_recovery(file_data,file_res,file_plot,set_of_lambda,num_simul):
def get_fraction_of_recovery(file_msg_prefix,set_sel_file_prefix,recovery_res_file,l_rcpk,set_of_noise,num_simul,l_cpk):
	# num_lamb=len(set_of_lambda)
	# res=np.zeros((num_simul,num_lamb,2))
	# set_selection_res=load_data(file_res)
	# for key in set_selection_res:
	# 	print key
	res_mean=[[],[]]
	for noise in set_of_noise:
		res_cpk=[]
		res_rcpk=[]
	
		for sim_no in range(num_simul):
			suff='n0'+str(noise)+'sim'+str(sim_no)
			data=load_data(file_msg_prefix+suff)
			# curr_res=set_selection_res[sim_no]
			# for key in curr_res['rcpk']:
			# 	print key
			# for l,l_no in zip(set_of_lambda,range(num_lamb)):
			selected_subset=load_data(set_sel_file_prefix+suff+'.cpk')		
			res_cpk.append(get_recovery_fraction(data.msg_end_tr,selected_subset[str(l_cpk)]) )
			selected_subset=load_data(set_sel_file_prefix+suff+'.rcpk')		
			res_rcpk.append(get_recovery_fraction(data.msg_end_tr,selected_subset[str(l_rcpk)]) )		
		res_mean[0].append(np.mean(np.array(res_cpk)))
		res_mean[1].append(np.mean(np.array(res_rcpk)))
	save( {'xvalues':set_of_noise,'yvalues':np.array(res_mean)}, recovery_res_file)
	# print res_mean
	# return

def plot_fraction_of_recovery(recovery_res_file, file_plot):
	res=load_data(recovery_res_file)
	plt.plot(res['yvalues'][0],'r-',label='cpk')
	plt.plot(res['yvalues'][1],'b-',label='rcpk')
	# plt.plot(set_of_lambda,res_mean[:,0],label='rcpk')
	# plt.plot(set_of_lambda,res_mean[:,1],label='cpk')
	plt.legend()
	plt.title('Recovery of fraction', fontsize=20)
	plt.xlabel('variance of noise', fontsize=20)
	plt.ylabel('% of endogenious recovered',fontsize=20)
	plt.grid(True)
	plt.xticks(range(len(res['xvalues'])), res['xvalues'],fontsize=20)
	# plt.ylim([.75,.85])
	plt.yticks([.75,.8,.85,.9],fontsize=20)
	# plt.show()
	plt.savefig(file_plot)
	plt.show()
	# x_ind=1
	# y_ind=1
	# for l_ind in range(num_lamb):
	# 	plt.subplot(num_lamb,x_ind,y_ind)
	# 	plt.plot()
	# 	if 
	# # res_rcpk=[]
	# res_cpk=[]
	# res=load_data(info['result_set_selection_file'])
	# for l in info['set_of_lambda']:
	# 	res_rcpk.append(get_recovery_fraction(data.msg_end_tr,res['rcpk'][str(l)])) 
	# 	res_cpk.append(get_recovery_fraction(data.msg_end_tr,res['cpk'][str(l)]))
	# plt.plot(res_cpk,label='CPK')
	# plt.plot(res_rcpk,label='RCPK')
	# plt.title('Fraction of endogenious recovered')
	# plt.legend()
	# plt.show()
	# plt.savefig(info['plot_recovery_file']+'.jpg')

def recover_parameters(file_data,file_set_sel,file_estim_param,num_simul=1,l=0,l_sl=0):
	# res_dict={'alpha':{},'mu':{},'A':{},'B':{}}
	# for key in res_dict:
	# l_sl=3
	res_dict={'cpk':{},'slant':{},'rcpk':{}}
	res_set_selection=load_data(file_set_sel)
	for sim_no in range(num_simul):
		print 'sim no :',sim_no
		data= load_data(file_data+'_'+str(sim_no))
		train=np.copy(data.train)
		res_dict['slant'][sim_no]=run_slant(data,l=l_sl)
		res_dict['cpk'][sim_no]=run_slant(data,flag_subset=True,subset=res_set_selection[sim_no]['cpk'][str(l)],l=l_sl)
		data.train=train
		res_dict['rcpk'][sim_no]=run_slant(data,flag_subset=True,subset=res_set_selection[sim_no]['rcpk'][str(l)],l=l_sl)
	res_dict_mean=compute_mean(res_dict, num_simul)
	save(res_dict_mean,file_estim_param)
	return res_dict_mean

def compute_mean( res_dict, num_simul):
	final_dict={'alpha':{},'A':{},'mu':{},'B':{}}

	for key in res_dict:
		alpha=[]
		mu=[]
		A=[]
		B=[]
		for sim_no in range(num_simul):
			alpha.append(res_dict[key][sim_no]['alpha'])	
			mu.append(res_dict[key][sim_no]['mu'])
			A.append(res_dict[key][sim_no]['A'])
			B.append(res_dict[key][sim_no]['B'])
		if num_simul==1:
			print 'yes'
			final_dict['alpha'][key]=np.array(alpha)
			final_dict['mu'][key]=np.array(mu)
			final_dict['A'][key]=np.array(A)
			final_dict['B'][key]=np.array(B)
		else:
			final_dict['alpha'][key]=np.mean(np.array(alpha),axis=0)
			final_dict['mu'][key]=np.mean(np.array(mu),axis=0)
			final_dict['A'][key]=np.mean(np.array(A),axis=0)
			final_dict['B'][key]=np.mean(np.array(B),axis=0)
	return final_dict

def plot_it(res,true,title_str,file_save):
	# print res['slant'][0].shape
	# print true.flatten().shape
	# plot_it(mean_res[key],true_param[key],key)
	index=np.where(true.flatten()!=0)[0]
	# print np.count_nonzero(index)
	for key in res:
		# print key
		# print res[key][:10]
		plt.plot(res[key].flatten()[index],label=key)
	# plt.plot(range(10))
	# plt.show()
	# # plt.savefig('temp.jpg')
	# return
	# rnd_vector=0.02*rnd.uniform(-1,1,true.shape[0])
	plt.plot(true.flatten()[index],label='true')
	# plt.plot(rnd_vector,label='random')
	plt.title(title_str)
	plt.grid(True)
	plt.xlim([1,50])
	plt.ylim([-.2,.2])
	# plt.xticks(np.arange(50))
	# plt.xlabel()
	plt.legend()
	plt.savefig(file_save)
	plt.show()
	plt.clf()


 
def recover_parameters_norm_dist(file_data,file_set_sel,file_graph,file_estim_param_norm,num_simul=0,set_of_lambda=None):
	
	def get_diff(dict_a,dict_b):
		res=[]
		for key in ['alpha','mu','A','B']:
			res.append(LA.norm(dict_a[key]-dict_b[key]))
		return np.array(res)

	res_array=np.zeros((num_simul,len(set_of_lambda),3,4))
	res_set_selection=load_data(file_set_sel)
	data=load_data(file_graph)
	true_param={'alpha':data.alpha,'A':data.A,'mu':data.mu,'B':data.B}
	for sim_no in range(num_simul):
		data=load_data(file_data+'_'+str(sim_no))
		train=np.copy(data.train)
		l_ind=0
		for l in set_of_lambda:
			res_array[sim_no,l_ind,0]=get_diff(run_slant(data,l=l),true_param)
			res_array[sim_no,l_ind,1]=get_diff(run_slant(data,flag_subset=True,subset=res_set_selection[sim_no]['cpk'][str(l)],l=l),true_param)
			data.train=train
			res_array[sim_no,l_ind,2]=get_diff(run_slant(data,flag_subset=True,subset=res_set_selection[sim_no]['rcpk'][str(l)],l=l),true_param)
			l_ind+=1
			data.train=train
	res_array_mean=np.mean(res_array,axis=0)
	save(res_array_mean,file_estim_param_norm)


def show_predictions(noise, set_of_noise,sim_no,result_file,set_of_method,l,set_of_lambda_cpk=None):
	res=load_data(result_file)
	# noise 
	# sim_no
	list_of_res=[]
	list_of_label=[]
	for method in set_of_method:
		# print res['n'+str(noise)]['s'+str(sim_no)]['m'+method]['l'+str(l)]['predicted'].shape
		list_of_res.append(res['n'+str(noise)]['s'+str(sim_no)]['m'+method]['l'+str(l)]['predicted'])
		list_of_label.append(method+' L='+str(l))
		if method=='cpk':
			if set_of_lambda_cpk!=None:
				for l_cpk in set_of_lambda_cpk:
					list_of_res.append(res['n'+str(noise)]['s'+str(sim_no)]['m'+method]['l'+str(l_cpk)]['predicted'])
					list_of_label.append(method+' L='+str(l_cpk))
	true=res['n'+str(noise)]['s'+str(sim_no)]['m'+method]['l'+str(l)]['true_target']
	list_of_label.append('true')
	# print true.ravel().shape
	list_of_res.append(true.reshape(true.shape[0],1))
	# print type()
	return {'xvalues':set_of_noise,'yvalues':np.array(list_of_res),'labels':list_of_label}



def get_diff_over_method(dict_a, true_key):
	list_curr=[]
	index=np.where(true_key.flatten()!=0)[0]
	for key in ['slant','cpk','rcpk']:
		# print dict_a[key].shape
		# return
		dist=dict_a[key].flatten()[index]-true_key.flatten()[index]
		
		# dist=dist[0]
		# print dist.shape

		list_curr.append(LA.norm(dist))
		# print key, LA.norm(dict_a[key]-true_key)
	# rnd_vector=0.01*rnd.uniform(-1,1,true_key.shape[0])
	# list_curr.append(LA.norm(rnd_vector-true_key) )
	return list_curr

def plot_param_norm(file_estim_param_norm):
	res_array=load_data(file_estim_param_norm)
	# res_array=(num_l,3,4)
	ind=0
	for key in ['alpha','mu','A','B']:
		# for arr,label_str in res_array[]
		plt.plot(res_array[:,0,ind],label='slant')
		plt.plot(res_array[:,1,ind],label='cpk')
		plt.plot(res_array[:,2,ind],label='rcpk')
		plt.title(key)
		plt.legend()
		plt.show()
		plt.clf()
		ind+=1

def prediction_module(parameter,data):
	# param=load_data(parameter_file)
	slant_obj=slant(init_by='object', obj = data, tuning_param=[10,10,float('inf')])	
	slant_obj.set_parameter( parameter,flag_dict=True)
	res=slant_obj.predict(num_simulation=1, time_span_input=0)
	# res['msg_set']	=None
	return res 

def plot_result(file_result=None,file_plot=None,res=None):
	if file_result!=None:
		res=load_data(file_result)


	# plt.plot(res['yvalues'][0],'b-',label='slant')
	# plt.plot(res['yvalues'][1],'r-',label='cpk')
	# plt.plot(res['yvalues'][2],'g-',label='rcpk')
	# plt.plot(res['yvalues'][3],'y-',label='true')
	
	for a,label_it in zip(res['yvalues'],res['labels']):
		plt.plot(a, label=label_it)

	# plt.plot(set_of_lambda,res_mean[:,0],label='rcpk')
	# plt.plot(set_of_lambda,res_mean[:,1],label='cpk')
	plt.legend()
	plt.title('Prediction Error( MSE )', fontsize=20)
	plt.xlabel('Variance of Noise', fontsize=20)
	plt.ylabel('Prediction Error',fontsize=20)
	plt.grid(True)
	# plt.xlim([0,100])
	plt.xticks(range(len(res['xvalue'])), res['xvalue'],fontsize=20)
	# plt.ylim([.75,.85])
	plt.yticks([.14,.18,.22,.26],fontsize=20)
	# plt.show()
	if file_plot!=None:
		plt.savefig(file_plot)
	plt.show()

def check_slant_single(msg_file, set_of_lambda,w,v,param_file=None):

	data=load_data(msg_file)
	if param_file==None:
		param_file= msg_file+'.param_slant_all_lambda'
	param={}
	for l in set_of_lambda:
		start=time.time()

		param[l]=run_slant(data,l=l,w=w,v=v)

		print time.time()-start, 'seconds'
	# save(param,param_file)
	# return 

	# print param.keys()


	l=set_of_lambda[0]
	true_p={'A':data.A,'B':data.B,'mu':data.mu, 'alpha':data.alpha}
	# param=load_data(param_file)
	x1,x2=get_vec_v2(true_p['alpha'],param[l]['alpha'],true_p['A'],param[l]['A'], data.edges)

	# print np.count_nonzero(x1)
	# print np.count_nonzero(x2)

	# x1,x2=get_vec(true_p['alpha'],param[str(l)]['alpha'],true_p['A'],param[str(l)]['A'])
	# x1,x2=get_vec(true_p['mu'],param[l]['mu'],true_p['B'],param[l]['B'])
	
	# x1=true_p['A'].flatten()
	# x2=param[l]['A'].flatten()



	# print np.count_nonzero(x1)
	# print np.count_nonzero(x2)

	plt.plot(x1,label='true')
	plt.plot(x2,label='estimated')
	# plt.xlim([4000,4100])
	plt.xlim([700,1000])
	plt.ylim([-4,4])
	plt.legend()
	plt.title(msg_file+' l='+str(l))
	plt.show()
	return 




	key='alpha'
	true_p={'A':data.A,'B':data.B,'mu':data.mu, 'alpha':data.alpha}
	plt.plot(to_vec(true_p[key]),label='true')
	param=load_data(param_file)
	for l in set_of_lambda:
		plt.plot(to_vec(param[l][key]),label='l='+str(l))
	# plt.xlim([4000,5000])
	plt.legend()
	plt.show()

def plot_estimation_results(file_result,file_plot):
	
	# print 'I am here'
	res=load_data(file_result)

	# for key in res['labels']:
	# 	print key
	for a,label_it in zip(res['yvalues'],res['labels']):
		# print a, label_it
		# for a in res['yvalues']:
		# 	# print a
		plt.plot(a, label=label_it)
	# return 
	# plt.plot(res['yvalues'][0],'b-',label='slant')
	# plt.plot(res['yvalues'][1],'r-',label='cpk')
	# plt.plot(res['yvalues'][2],'g-',label='rcpk')
	# plt.plot(set_of_lambda,res_mean[:,0],label='rcpk')
	# plt.plot(set_of_lambda,res_mean[:,1],label='cpk')
	plt.legend()
	plt.title('Estimation error', fontsize=20)
	plt.xlabel('variance of noise', fontsize=20)
	plt.ylabel('Estimation error',fontsize=20)
	plt.grid(True)
	plt.xticks(range(len(res['xvalues'])), res['xvalues'],fontsize=20)
	# plt.ylim([.75,.85])
	plt.yticks([1.1,1.2,1.3,1.4,1.5],fontsize=20)
	# plt.show()
	# plt.savefig(file_plot)
	plt.show()
	
def generate_result_norm_param(file_graph,file_estim_param,result_file,l_sl,set_of_noise,num_simul,set_of_method, set_of_lambda_cpk):	
	def get_diff(dict_a,dict_b):
		res=[]
		norm_diff=0
		norm_val=0

		for key in ['alpha','A']:
			# res.append(LA.norm(dict_a[key]-dict_b[key])/LA.norm(dict_b[key]) )
			# plt.plot('')
			norm_diff+=np.power(LA.norm(dict_a[key]-dict_b[key]),2)
			norm_val+=np.power(LA.norm(dict_b[key]),2)
			# res.append(LA.norm(dict_a[key]-dict_b[key]))
		# return LA.norm(np.array(res))
		# return np.array(res)
		# return np.mean(np.array(res))
		return sqrt(norm_diff)/sqrt(norm_val)


	num_noise=len(set_of_noise)
	res_array=np.zeros((2+len(set_of_lambda_cpk),num_noise,num_simul))
	set_of_label=[' ' for i in range(res_array.shape[0])]
	data=load_data(file_graph)
	true_param={'alpha':data.alpha,'A':data.A,'mu':data.mu,'B':data.B}
	# set_of_labels=[]
	for noise,noise_no in zip(set_of_noise,range(num_noise) ):
		for sim_no in range(num_simul):
			suff='n0'+str(noise)+'sim'+str(sim_no)
			ind=0
			for method in set_of_method: 
				if method =='cpk':
					for l_cpk in set_of_lambda_cpk: 
						res_array[ind,noise_no,sim_no]=get_diff(load_data(file_estim_param+suff+'.'+method)[str(l_cpk)],true_param)	
						set_of_label[ind]=method # +' L='+str(l_cpk)
						ind+=1

				else:
					res_array[ind,noise_no,sim_no]=get_diff(load_data(file_estim_param+suff+'.'+method)[str(l_sl)],true_param)	
					set_of_label[ind]=method # +' L='+str(l_sl)
					ind+=1
			# res_array[0,noise_no,sim_no]=get_diff(load_data(file_estim_param+suff+'.slant')[str(l_sl)],true_param)
			# list_of_label.append(method+' L='+str(l_cpk))
			# res_array[1,noise_no,sim_no]=get_diff(load_data(file_estim_param+suff+'.cpk')[str(l_sl)],true_param)
			# res_array[2,noise_no,sim_no]=get_diff(load_data(file_estim_param+suff+'.rcpk')[str(l_sl)],true_param)
			
	res_array_mean=np.mean(res_array,axis=2)
	save({'xvalues':set_of_noise,'yvalues':res_array_mean,'labels':set_of_label},result_file)

def estimate_slant_params(file_msg,file_set_sel,file_estim_param,l_set_sel,set_of_l_sl,set_of_noise,num_simul):
	for noise in set_of_noise:
		for sim_no in range(num_simul):
			print noise, 'sim ',sim_no
			suff='n0'+str(noise)+'sim'+str(sim_no)
			slant={}
			cpk={}
			rcpk={}
			data=load_data(file_msg+suff)
			train=np.copy(data.train)
			cpk_subset=load_data(file_set_sel+suff+'.cpk')
			rcpk_subset=load_data(file_set_sel+suff+'.rcpk')
			for l in set_of_l_sl:
				slant[str(l)]=run_slant(data,l=l)
				cpk[str(l)]=run_slant(data,flag_subset=True,subset=cpk_subset[str(l_set_sel)],l=l)
				data.train=np.copy(train)
				rcpk[str(l)]=run_slant(data,flag_subset=True,subset=rcpk_subset[str(l_set_sel)],l=l)
				data.train=np.copy(train)
			save(slant, file_estim_param+suff+'.slant')
			save(cpk, file_estim_param+suff+'.cpk')
			save(rcpk, file_estim_param+suff+'.rcpk')

def generate_set_selection_files( msg_file,set_sel_file,set_of_lambda,frac_end,var,w ):
	# res_cpk={}
	res_rcpk={}
	start=time.time()
	data=load_data(msg_file)
	print 'sample size', data.train.shape[0]
	for l in set_of_lambda:
		# res_cpk[str(l)]=run_CPK(data,l,frac_end,var,w)	
		res_rcpk[str(l)]=run_RCPK(data,l,frac_end,w)
	print 'CPK and RCPK takes ', time.time()-start , 'seconds'
	# save(res_cpk, set_sel_file+'.cpk')
	save(res_rcpk, set_sel_file+'.rcpk')

def to_vec(X):
	if len(X.shape) == 1:
		return X
	res=np.array([])
	for x in X:
		res=np.concatenate((res,x[np.nonzero(x)]))
	return res

def to_vec_diff_sq(X,Z):
	sum=0
	for x,z in zip(X,Z):
		diff=x-z
		sum+=np.sum(diff[np.nonzero(x)]**2)
	return sum

def get_diff_single(dict_a,dict_b):
	res={}
	for key in dict_a:
		if key in ['alpha','mu']:
			res[key]=np.sum((dict_a[key]-dict_b[key])**2)
		else:
			res[key]=to_vec_diff_sq(dict_a[key],dict_b[key])#np.sum((to_vec(dict_a[key])-to_vec(dict_b[key]))**2)
	a=sqrt(res['alpha']+res['A'])/LA.norm(np.concatenate((dict_a['alpha'],to_vec(dict_a['A']))))
	b=sqrt(res['mu']+res['B'])/LA.norm(np.concatenate((dict_a['mu'],to_vec(dict_a['B']))))
	return np.array([a,b])

def get_vec(key1,key2,keymat1,keymat2):
	res1=key1
	res2=key2
	for x,z in zip(keymat1,keymat2):
		res1=np.concatenate((res1,x[np.nonzero(x)]))
		res2=np.concatenate((res2,z[np.nonzero(x)]))
	return res1, res2

def get_relative_error(dict_a,dict_b):
	def rel_error( base_vector, estimated_vector ):
		diff_vector = estimated_vector - base_vector
		return LA.norm( diff_vector)/LA.norm(base_vector)


	a_alpha,b_alpha=get_vec(dict_a['alpha'],dict_b['alpha'],dict_a['A'],dict_b['A'])
	a_mu,b_mu=get_vec(dict_a['mu'],dict_b['mu'],dict_a['B'],dict_b['B'])
	
	return np.array([ rel_error(a_alpha,b_alpha), rel_error(a_mu,b_mu) ])

def get_MSE_diff(dict_a,dict_b):
	def MSE(x):
		return np.average(x**2)
	a_alpha,b_alpha=get_vec(dict_a['alpha'],dict_b['alpha'],dict_a['A'],dict_b['A'])
	a_mu,b_mu=get_vec(dict_a['mu'],dict_b['mu'],dict_a['B'],dict_b['B'])
	return np.array([MSE(a_alpha-b_alpha), MSE(a_mu-b_mu)])

def run_slant(data,flag_subset=False,subset=None,l=None, max_iter=None, w=10,v=10 ):
	if flag_subset:
		# print data.train.shape 
		# print subset.shape
		data.train=data.train[subset]
		# print 'subset nonempty'
	# else:
		# print 'subset empty'
	start=time.time()
	slant_obj=slant(init_by='object', obj = data, tuning_param=[w,v,float('inf')])
	# slant_obj=slant(init_by='dict', obj = dict_obj, tuning_param=[10,10,float('inf')])
	slant_obj.estimate_param(lamb=l,max_iter=max_iter) # lambda can be passed here
	print 'slant takes ',time.time()-start,' seconds'
	return {'A':slant_obj.A,'B':slant_obj.B,'mu':slant_obj.mu, 'alpha':slant_obj.alpha}	

def compare(true_param, estim_param, field_name, set_of_lambda,opt='matrix_plot'):
	if opt=='l2':
		list_l2_distance=[]
		for l in set_of_lambda:
			# a=(true_param[field_name])
			# b=(estim_param[str(l)][field_name])
			# list_l2_distance.append(LA.norm(a-b))
			list_l2_distance.append(LA.norm( true_param[field_name]- estim_param[str(l)][field_name]) )
		plt.plot(list_l2_distance)
		plt.show()
		plt.savefig('temp.jpg')
	if opt=='vector_plot':
		# indices= np.where(true_param.flatten()!=0)[0]
		plt.plot(true_param[field_name])
		for l in set_of_lambda:
			plt.plot(estim_param[str(l)][field_name], label=str(l))
		plt.legend()
		# plt.xlim(1,20)
		plt.show()
		plt.savefig('../result_synthetic_dataset/plots/slant.jpg')
	if opt=='matrix_plot':
		indices= np.where(true_param[field_name].flatten()!=0)[0]
		plt.plot(true_param[field_name].flatten()[indices])
		for l in set_of_lambda:
			plt.plot(estim_param[str(l)][field_name].flatten()[indices], label=str(l))
		plt.legend()
		plt.xlim(200,400)
		plt.show()
		plt.savefig('../result_synthetic_dataset/plots/slant.jpg')

def generate_plot_file_single(result_file,plot_file,set_of_keys,set_of_lambda):
	res=load_data(result_file)
	num_key=len(set_of_keys)
	result_dict={'x-axis':set_of_keys}
	result_dict['labels']=[method+':l:'+str(l) for method in ['slant','cpk','rcpk'] for l in set_of_lambda ]
	num_rows=3*len(set_of_lambda)
	# for col in [0]:
	res_array=np.zeros((num_rows,num_key))
	ind=0
	for key in set_of_keys:
		res_array[:,ind]=res[str(key)][:,0]
		ind+=1
	# print res_array
	result_dict['y-axis']=res_array
	save(result_dict,plot_file)

def plot_single(plot_file,flag, dict_to_save=None, selected_index=None, num_msg_per_node=None, final_plot=None, set_of_lambda=None):
	res=load_data(plot_file)


	# print res['labels']
	if flag=='all':
		ind=0
		initial_line_w=1
		for x,l in zip(res['y-axis'],res['labels']):
			# if 'rcpk' not in l :
				line=plt.semilogy(x,label=l)
				plt.setp(line, linewidth=initial_line_w, linestyle='--')
				ind+=1
				if ind==len(set_of_lambda):
					initial_line_w+=1
					ind=0
		plt.legend()
		# plt.ylim([.6,1])
		# plt.yticks([.1,.2,.3,.4,.5,.6,.7,.8,.9])
		plt.show()
	if flag=='selected':
		labels=[]
		final_array=np.zeros((3,res['y-axis'].shape[1]))
		for ind,l,ind_final_res in zip(selected_index,['slant','cherrypick','robust cherrypick'],range(3)):
			plt.semilogy(res['y-axis'][ind],label=l)
			final_array[ind_final_res]=res['y-axis'][ind]
			labels.append(l)
		final_dict={'x-axis':num_msg_per_node,'y-axis':final_array,'labels':labels, 'lambda':set_of_lambda}
		save(final_dict, dict_to_save)
		plt.legend()
		plt.savefig(final_plot)
		plt.show()

def get_vec_v2(key1,key2,keymat1,keymat2,edges):
	z1=key1
	z2=key2
	for a,b,node_nbr in zip(keymat1,keymat2,edges):
		z1=np.concatenate((z1,a[np.nonzero(node_nbr)]))
		z2=np.concatenate((z2,b[np.nonzero(node_nbr)]))
	return z1,z2

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

def generate_result_norm_param_over_msg_per_node(msg_file,result_file,set_of_lambda):	
	# result_file=msg_file+'.combined_result'
	data=load_data(msg_file)
	true_param={'alpha':data.alpha,'A':data.A,'mu':data.mu,'B':data.B}
	result=load_data(result_file)
	# result={}

	res_array=np.zeros((3*len(set_of_lambda),2) )
	ind=0
	for method in ['.slant','.cpk','.rcpk']: 
		res=load_data(msg_file+method+'.param')
		print res.keys()
		# print res.keys()
		if not res:
			print 'file does not exists : method : ' , method , ' : msg : ', msg_file
		for l in set_of_lambda:
			# res_array[ind]=get_diff_single(true_param,res[str(l)]) 
			if res:
				res_array[ind]=get_MSE_diff(true_param,res[str(l)]) 
			else:
				res_array[ind]=0
			ind+=1
	num_msg=msg_file.split('msg_')[1].split('_data')[0]
	result[num_msg]=res_array
	save(result,result_file)

def estimate_slant_params_single(file_msg,set_of_lambda,w,v,l_cpk=None):
	# l_sub=0.5
	# print 'subset sel method only uses lambda',l_sub
	slant=load_data(file_msg+'.slant.param','ifexists')
	cpk=load_data(file_msg+'.cpk.param','ifexists')
	rcpk=load_data(file_msg+'.rcpk.param','ifexists')

	# print slant.keys()
	# print cpk.keys()
	# print rcpk.keys()
	# return 
	data=load_data(file_msg)
	train=np.copy(data.train)
	cpk_subset=None
	rcpk_subset=None
	if os.path.isfile(file_msg+'.cpk.pkl'):
		cpk_subset=load_data(file_msg+'.cpk')
	if os.path.isfile(file_msg+'.rcpk.pkl'):
		rcpk_subset=load_data(file_msg+'.rcpk')
	for l in set_of_lambda:
		if l_cpk==None:
			l_cpk=l
		slant[str(l)]=run_slant(data,l=l,w=w,v=v)
		if cpk_subset!=None:
			cpk[str(l)]=run_slant(data,flag_subset=True,subset=cpk_subset[str(l_cpk)],l=l,w=w,v=v)
			data.train=np.copy(train)
		
		if rcpk_subset!=None:
			rcpk[str(l)]=run_slant(data,flag_subset=True,subset=rcpk_subset[str(l)],l=l,w=w,v=v)
			data.train=np.copy(train)
		
	save(slant, file_msg+'.slant.param')
	save(cpk, file_msg+'.cpk.param')
	save(rcpk, file_msg+'.rcpk.param')


def plot_synthetic(file_data,file_prefix,file_plot):
	
	data=load_data(file_data)
	x= data['x-axis']
	y=data['y-axis']
	labels=data['labels']
	
	marker_seq=['P','o','*']#,'o','*','P']
	ls_seq=['--',':','-.']#,'--','-.','--',':']
	
	for y_row,l,ls,mk in zip(y,labels,ls_seq,marker_seq):
		line=plt.plot(y_row, label=l)
		plt.setp(line, linewidth=4, marker=mk, markersize=15,linestyle=ls)
	plt.grid(True)
	plt.legend()
	plt.xticks(range(5),x,rotation=0,fontsize=20.7,weight='bold')
	# print data.keys()
	# ytck=data['y-tick']
	ytck=np.arange(.7,1.01,.1)
	data['y-tick']=ytck
	plt.yticks(ytck,rotation=0,fontsize=20.7,weight='bold')
	plt.xlabel('Average msg per node',rotation=0,fontsize=20.7,weight='bold')
	plt.ylabel('MSE ',rotation=90,fontsize=20.7,weight='bold')
	# file_pref ix  Barabasi Alberta Hierarchical
	plt.title('Kronecker Homophily ',rotation=0,fontsize=20.7,weight='bold')
	plt.tight_layout()
	plt.savefig(file_plot)
	plt.show()
	save(data, file_data)

def check():
	y=np.array(range(10))*.1
	plt.semilogy(y)
	plt.grid(True)
	plt.yticks([.1,1,4,7])
	plt.show()

def main():
	# check()
	# return
	# msg_file='../Cherrypick_others/Data_opn_dyn_python/british_election_full'
	# data=load_data(msg_file)
	
	# print check_symmetric(data.edges, tol=1e-8)
	# return

	estimation_on_single_msg_stream=True
	generate_graph_flag=False# True #
	generate_msg_flag=False # True
	generate_single_msg_flag=False # True
	generate_set_selection_files_flag=False # True
	estimate_slant_params_flag=False # True
	param_recovery_norm_flag=False
	plot_param_norm_flag=False # True# False
	generate_predictions_flag=False # True
	init_kronecker=False
	check_slant_single_flag=False#True
	#***********NOT REQUIRED***************************************
	plot_fraction_flag=False # True
	param_recovery_flag=False # True
	plot_param_flag=False # True
	evaluate_synthetic_data_flag=False # True
	evaluate_intensity_param_flag=False # True
	check_slant_flag=False # True
	#----------------------------------------------------------------
	num_node = 500 # int(sys.argv[1]) # 10
	num_nbr=5
	num_node_start=3
	num_rep=2	
	split_frac=0.9
	index=2#3
	time_span=5
	num_simul=5
	w=1000
	v=10
	
	#----------------------------
	generate_str ='kronecker'#  'barabasi_albert' # 
	if generate_str=='kronecker':
		index=int(sys.argv[1])
		file_prefix=['kronecker_512', 'kroneckerCP512', 'kroneckerHeterophily512', 'kroneckerHier512', 'kroneckerHomophily512'][index]#[int(sys.argv[2])]#generate_str+'_'+str(num_node_start)+'_'+str(num_rep)+'_'+str(index)
	else:
		file_prefix=generate_str+'_'+str(num_node)+'_'+str(num_nbr)+'_'+str(index)
	#-------------------------------------
	path = '../result_synthetic_dataset/'
	file_common=path+'msg/'+file_prefix+'_data'
	file_msg_prefix=file_common
	#------------------------------------Estimation single data--------------------------------

	if estimation_on_single_msg_stream:


		# d = load_data('../result_synthetic_dataset_backup_25dec_23_44/'+file_prefix+'.final_plot')
		# print d['lambda']
		# return 

		#---
		# data=load_data('../result_synthetic_dataset/kroneckerCP512_msg_12500_data.slant.param')
		# # for key in data['0.5']:
		# # 	print key
		# print sys.getsizeof(data['0.5']['A'])/1024
		# return 

		#---
		# msg_file='../result_synthetic_dataset/'+file_prefix+'_msg_'+str(100000)+'_data'
		# set_of_lambda=[.5]
		# check_slant_single(msg_file, set_of_lambda,w,v)
		# return 
		# set_of_msg=[25,30,50,75,100]
		set_of_msg=[25, 50, 100, 200, 400 ]#[int(sys.argv[2])]]
		for max_msg in [500*i for i in set_of_msg]:#75 # [25,50,100,200,400,800]]:
			msg_file='../result_synthetic_dataset/'+file_prefix+'_msg_'+str(max_msg)+'_data'
			# print 'sample size ', max_msg
			set_of_lambda=[.2,.5]#[.1,.5,1] 
			# set_of_lambda=[.01,.05,.2,.3,.4,1.5,1.7,2]
			frac_end=0.8
			var=0.1
			msg_file='../result_synthetic_dataset/'+file_prefix+'_msg_'+str(max_msg)+'_data'

			# generate_set_selection_files( msg_file,msg_file,set_of_lambda,frac_end,var,w )
			l_cpk=.5
			set_of_method=['slant','cpk','rcpk']
			# estimate_slant_params_single(msg_file,set_of_lambda,w,v,l_cpk=l_cpk)
			# return 
			result_file='../result_synthetic_dataset/'+file_prefix+'_combined_result'
			# save({},result_file)
			# return 
			# generate_result_norm_param_over_msg_per_node(msg_file, result_file, set_of_lambda)
		# return 
		plot_file='../result_synthetic_dataset/'+file_prefix+'_plot'
		set_of_keys=[500*i for i in [25,50,100,200,400]]

		# generate_plot_file_single(result_file,plot_file,set_of_keys,set_of_lambda)
		# plot_single(plot_file,flag='all',set_of_lambda=set_of_lambda)
		selected_index=[0,3,4]
		set_of_lambda=[.2,.5,.2]
		dict_file='../result_synthetic_dataset/'+file_prefix+'.final_plot'
		final_plot='../result_synthetic_dataset/'+file_prefix+'.final_plot.jpg'
		# plot_single(plot_file,flag='selected',dict_to_save=dict_file,final_plot=final_plot, num_msg_per_node=set_of_msg, selected_index=selected_index, set_of_lambda=set_of_lambda)
		# change 1 : do not plot x axis 
		
		file_data='../result_synthetic_dataset/'+file_prefix+'.final_plot'
		file_plot='../result_synthetic_dataset/'+file_prefix+'.jpg'
		plot_synthetic(file_data,file_prefix,file_plot)

	if init_kronecker:

		#-------------checking
		# datafile='../result_synthetic_dataset/'+file_prefix+'_graph'
		# data=load_data(datafile)
		# print np.amax(data.edges)
		# print np.amin(data.edges)
		# print data.edges.shape
		#--------------
		data=synthetic_data()
		file_pre='../Cherrypick_others/Data_opn_dyn_python/Kronecker/' + file_prefix +'_'
		data.init_from_file(file_pre)
		save(data,'../result_synthetic_dataset/'+file_prefix+'_graph')

	if check_slant_single_flag:
		set_of_msg=[25,50,100,200] # ,300,400] #------------------------------------------
		set_of_lambda=[.5,.7,1,2]

		flag_run_slant=False 
		flag_generate_dict=False# True
		flag_generate_plot= True

		
		msg=400000
		data_file='../result_synthetic_dataset/'+file_prefix+'_msg_'+str(msg)+'_data'
		param_file='../result_synthetic_dataset/'+file_prefix+'_msg_'+str(msg)+'_data.slant.param'
		check_slant_single(data_file, set_of_lambda=[float(sys.argv[2])],w=w,v=v)#,param_file=param_file)
		return 

		if flag_run_slant:
			for msg in [500*i for i in set_of_msg]:#75 # [25,50,100,200,400,800]]:
				data_file='../result_synthetic_dataset/'+file_prefix+'_msg_'+str(msg)+'_data'
				data=load_data(data_file)
				res={}
				for lamb in set_of_lambda:
					res[str(lamb)]=run_slant(data, l=lamb, w=w,v=v )
				param_file=data_file+'.slant.param'
				save(res, param_file)


		#--------------------------PLOTTING----------------------------------
		# generate a dictionary for both measure
		if flag_generate_dict:
			measure='MSE' # 'rel_error'
			dict_file = '../result_synthetic_dataset/'+ file_prefix + '.slant.'+measure
			msg=25000
			data=load_data('../result_synthetic_dataset/'+file_prefix+'_msg_'+str(msg)+'_data')
			true_param={'alpha':data.alpha, 'A':data.A, 'mu':data.mu, 'B':data.B}
			res_dict={}
			for msg in [500*i for i in set_of_msg]:#75 # [25,50,100,200,400,800]]:
				res_dict[msg]={}
				param_file='../result_synthetic_dataset/'+file_prefix+'_msg_'+str(msg)+'_data.slant.param'
				res=load_data(param_file)
				for lamb in set_of_lambda:
					if measure == 'rel_error':
						res_dict[msg][lamb]=get_relative_error( true_param,res[str(lamb)])[0]
					if measure == 'MSE':
						res_dict[msg][lamb]=get_MSE_diff( true_param,res[str(lamb)])[0]
			save( res_dict , dict_file )

		# plotting
		if flag_generate_plot:
			measure='rel_error'#'MSE'
			dict_file = '../result_synthetic_dataset/'+ file_prefix + '.slant.'+measure
			res_dict=load_data(dict_file)
			plot_file='../result_synthetic_dataset/'+file_prefix+'.slant.'+measure+'.plot'
			plot_to_save='../result_synthetic_dataset/'+file_prefix+'.slant.'+measure+'.plot.jpg'
			plot_dict = {'x-axis':[500*i for i in set_of_msg]} 
			final_list=[]
			for lamb in set_of_lambda:
				res_list = [res_dict[500*i][lamb] for i in set_of_msg]
				final_list.append( res_list )
				plt.plot(res_list, label='lamb='+str(lamb))
			plt.legend()
			plt.show()
			plt.savefig( plot_to_save )

			plot_dict['y-axis']=np.array(final_list)
			save( plot_dict, plot_file)

	#********************************GENERATE GRAPH ********************************************
	if generate_graph_flag:
		plot_file=path + 'plots/' + file_prefix + '_graph.jpg'
		max_intensity=5
		file_graph=path+file_prefix+'_graph'
		generate_graph(file_graph,generate_str,num_node,num_nbr,num_node_start,num_rep,plot_file,max_intensity)
		# return
	
	if generate_single_msg_flag:
		
		# split, delete if only generation is required
		max_msg=400000
		output_file=path+file_prefix+'_msg_'+str(max_msg)+'_data'
		data=load_data(output_file)
		for n in [25,50,100,200,400]:#,200,400,800]:
			msg=int(n*500)
			data_new=copy.deepcopy(data)
			data_new.train=data_new.train[:msg]
			data_new.msg_end_tr=data_new.msg_end_tr[:msg]
			file_to_save=path+file_prefix+'_msg_'+str(msg)+'_data'
			save(data_new,file_to_save)
			del data_new
		return

		# generation
		data=load_data(path+file_prefix+'_graph')
		data.generate_parameters() 	
		time_span=100000
		max_msg=int(sys.argv[2])
		split_frac=1
		output_file=path+file_prefix+'_msg_'+str(max_msg)+'_data'
		plot_file=path+file_prefix+'_data.jpg'
		data.generate_msg_single(plot_file,time_span,p_exo=.2,var=.1,flag_check_num_msg = True,max_msg=max_msg,w=w,v=v)
		data.split_data(split_frac)
		print data.train.shape[0]
		save(data,output_file)
		data.clear_msg()

	if generate_msg_flag:
		data=load_data(path+file_prefix+'_graph')
		output_file=path+'msg/'+file_prefix+'_data'
		time_span=5
		num_simul=5
		set_of_noise=[.1,.2,.3,.4,.5]
		for noise_param in set_of_noise:
			for sim_no in range(num_simul):
				suff='n0'+str(noise_param)+'sim'+str(sim_no)
				plot_file=path+'plots/'+file_prefix+suff+'_data.jpg'
				data.generate_msg(plot_file,time_span, option='both_endo_exo',p_exo=.2,var=.1, noise_param=noise_param)
				# msg[sim_no]=np.copy(data.msg)
				data.split_data(split_frac)
				save(data,output_file+'n0'+str(noise_param)+'sim'+str(sim_no))
				data.clear_msg()

	# if estimation
	#********************************************************************************

	#*************************************ESTIMATION**********************************

	if generate_predictions_flag:
		set_of_noise=[.1,.2,.3,.4,.5]
		num_simul=5
		set_of_lambda=[.5,.7,1,3]
		set_of_lambda_cpk=[1.2]
		set_of_method=['slant','cpk','rcpk']
		msg_file=path+'msg/'+file_prefix+'_data'
		slant_file=path+'slant_files/'+file_prefix
		result_file=path+file_prefix+'_opinion_prediction'
		l=.5
		result_MSE_values=path+file_prefix+'_l'+str(l)+'_prediction_MSE'
		file_plot=path+'plots/'+file_prefix+'_l'+str(l)+'_MSE.jpg'
		# generate_predictions(msg_file,slant_file,set_of_noise,num_simul,set_of_method,set_of_lambda,result_file,set_of_lambda_cpk)
		# get_prediction_result(result_file,set_of_method,set_of_noise,num_simul,l,result_MSE_values,set_of_lambda_cpk)
		plot_result(result_MSE_values,file_plot)	
		# plot_res=show_predictions(.1, set_of_noise,0,result_file,set_of_method,l)
		# plot_result(res=plot_res)


	if generate_set_selection_files_flag:
		set_of_noise=[.1,.2,.3,.4,.5]
		num_simul=5
		set_of_lambda=[1.2]
		frac_end=0.8
		set_sel_file_prefix=path+'set_sel_files/'+file_prefix
		var=0.1
		for noise in set_of_noise:
			for sim_no in range(num_simul):
				print 'noise ',noise, ' simulation ',sim_no
				msg_file=file_msg_prefix+'n0'+str(noise)+'sim'+str(sim_no)
				set_sel_file=set_sel_file_prefix+'n0'+str(noise)+'sim'+str(sim_no)
				generate_set_selection_files( msg_file,set_sel_file,set_of_lambda,frac_end,var,w )
		
	if estimate_slant_params_flag:
		file_estim_param=path+'slant_files/'+file_prefix
		file_set_sel=path+'set_sel_files/'+file_prefix
		file_msg=path+'msg/'+file_prefix+'_data'
		file_graph=path+file_prefix+'_graph'
		file_result=path+file_prefix+'_estim_error'
		file_plot=path+'plots/'+file_prefix+'_estim_error_cpkall.jpg'
		l_set_sel=.5
		set_of_l_sl=[1.2,1.3,1.5]
		set_of_noise=[.1,.2,.3,.4,.5]
		set_of_method=['slant','cpk','rcpk']
		set_of_lambda_cpk=[1.2]#,1.3,1.5]
		num_simul=5
		# estimate_slant_params(file_msg,file_set_sel,file_estim_param,l_set_sel,set_of_l_sl,set_of_noise,num_simul)
		l_sl=.5
		# generate_result_norm_param(file_graph,file_estim_param,file_result,l_sl,set_of_noise,num_simul,set_of_method,set_of_lambda_cpk)
		plot_estimation_results(file_result,file_plot)


	if param_recovery_norm_flag:
		
		file_estim_param_norm=path+file_prefix+'_estim_parameter'
		file_set_sel=path+file_prefix+'_set_selection_result'
		file_data=file_common
		file_graph=path+file_prefix+'_graph'
		recover_parameters_norm_dist(file_data,file_set_sel,file_graph,file_estim_param_norm,num_simul=num_simul,set_of_lambda=set_of_lambda)
	
	if plot_param_norm_flag:
		file_estim_param_norm=path+file_prefix+'_estim_parameter'
		plot_param_norm(file_estim_param_norm)

	#******************************NOT REQUIRED***********************************************		
	
if __name__=="__main__":
	main()
