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
	
	data.generate_parameters(max_intensity=max_intensity)
	save(data,file_graph)
		
def save(obj,output_file):
	with open(output_file+'.pkl' , 'wb') as f:
		pickle.dump( obj , f , pickle.HIGHEST_PROTOCOL)

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

def run_RCPK(obj,lamb,frac_end):
	start = time.time()
	Cherrypick_obj = Robust_Cherrypick( obj = obj , init_by = 'object', lamb = lamb ) 
	Cherrypick_obj.initialize_data_structures()
	w, active_set, norm_of_residual = Cherrypick_obj.robust_regression_via_hard_threshold( method = 'FC', max_itr = 50 , frac_end = frac_end) 
	# file_to_write=file_to_write_prefix+'w10f'+str(frac_end)+'l'+str(lamb)+'.res.Robust_cherrypick.full'
	total_time = time.time()-start
	result=Cherrypick_obj.save_active_set( norm_of_residual = norm_of_residual  , total_time = total_time, )
	return result['data']

def run_CPK(obj,lamb,frac_end,var):
	cherrypick_obj = cherrypick( obj = obj , init_by = 'object', param ={'lambda':lamb,'sigma_covariance':var}) 
	cherrypick_obj.demarkate_process(frac_end=frac_end)
	# file_to_write=file_to_write_prefix+'w10f'+str(frac_end)+'l'+str(lamb)+'.res.cherrypick'+'.full'
	result_obj = cherrypick_obj.save_end_msg( frac_end = frac_end ) 
	return result_obj['data']

def get_recovery_fraction(msg_end_true, msg_end_estm):
	return np.count_nonzero(np.logical_and(msg_end_true,msg_end_estm))/float(np.count_nonzero(msg_end_true))
		
def generate_set_selection_files( msg_file,set_sel_file,set_of_lambda,frac_end,var ):
	res_cpk={}
	res_rcpk={}

	data=load_data(msg_file)
	for l in set_of_lambda:
		res_cpk[str(l)]=run_CPK(data,l,frac_end,var)
		res_rcpk[str(l)]=run_RCPK(data,l,frac_end)
	save(res_cpk, set_sel_file+'.cpk')
	save(res_rcpk, set_sel_file+'.rcpk')

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
	
	 
	
def run_slant(data,flag_subset=False,subset=None,l=None, max_iter=None):
	if flag_subset:
		# print data.train.shape 
		# print subset.shape
		data.train=data.train[subset]
		# print 'subset nonempty'
	# else:
		# print 'subset empty'
	slant_obj=slant(init_by='object', obj = data, tuning_param=[10,10,float('inf')])
	# slant_obj=slant(init_by='dict', obj = dict_obj, tuning_param=[10,10,float('inf')])
	slant_obj.estimate_param(lamb=l,max_iter=max_iter) # lambda can be passed here
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

# def plot_fraction_of_recovery(file_data,file_res,file_plot,set_of_lambda,num_simul):
def get_fraction_of_recovery(file_msg_prefix,set_sel_file_prefix,recovery_res_file,l,set_of_noise,num_simul):
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
			res_cpk.append(get_recovery_fraction(data.msg_end_tr,selected_subset[str(l)]) )
			selected_subset=load_data(set_sel_file_prefix+suff+'.rcpk')		
			res_rcpk.append(get_recovery_fraction(data.msg_end_tr,selected_subset[str(l)]) )		
		res_mean[0].append(np.mean(np.array(res_cpk)))
		res_mean[1].append(np.mean(np.array(res_rcpk)))
	save( {'xvalues':set_of_noise,'yvalues':np.array(res_mean)}, recovery_res_file)
	# print res_mean
	# return

def plot_fraction() :
	pass
	# plt.plot(res[:,0,0],label='rcpk')
	# plt.plot(res[:,0,1],label='cpk')
	# plt.plot(set_of_lambda,res_mean[:,0],label='rcpk')
	# plt.plot(set_of_lambda,res_mean[:,1],label='cpk')
	# plt.legend()
	# plt.xlabel('lambda')
	# plt.ylabel('percentage of endogenious recovered')
	# plt.grid(True)
	# plt.ylim([.75,.85])
	# # plt.yticks
	# # plt.show()
	# plt.savefig(file_plot)
	# plt.show()
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


def main():
	# define parameters
	# train_fraction = .9
	# frac_sparse = .6
	generate_graph_flag= False #
	generate_msg_flag=False # True
	generate_set_selection_files_flag=False # True
	param_recovery_norm_flag=False
	plot_param_norm_flag=False#True# False
	#***********NOT REQUIRED***************************************
	plot_fraction_flag= True#False#True
	param_recovery_flag=False # True
	plot_param_flag=False # True
	evaluate_synthetic_data_flag=False#True
	evaluate_intensity_param_flag=False#True
	check_slant_flag=False # True
	#----------------------------------------------------------------
	num_node = 500# int(sys.argv[1]) # 10
	num_nbr=5
	num_node_start=3
	num_rep=5
		
	split_frac=0.9
	index=1
	time_span=5
	num_simul=5
	#----------------------------
	generate_str = 'barabasi_albert' # 'kronecker' # 
	if generate_str=='kronecker':
		file_prefix=generate_str+'_'+str(num_node_start)+'_'+str(num_rep)+'_'+str(index)
	else:
		file_prefix=generate_str+'_'+str(num_node)+'_'+str(num_nbr)+'_'+str(index)
	#-------------------------------------
	path = '../result_synthetic_dataset/'
	
	
	file_common=path+'msg/'+file_prefix+'_data'
	file_msg_prefix=file_common
	#********************************GENERATE GRAPH ********************************************
	if generate_graph_flag:
		plot_file=path + 'plots/' + file_prefix + '_graph.jpg'
		max_intensity=5
		file_graph=path+file_prefix+'_graph'
		generate_graph(file_graph,generate_str,num_node,num_nbr,num_node_start,num_rep,plot_file,max_intensity)
	
	if generate_msg_flag:
		data=load_data(path+file_prefix+'_graph')
		output_file=path+'msg/'+file_prefix+'_data'
		time_span=5
		set_of_noise=[.1,.2,.3,.4,.5]
		for noise_param in set_of_noise:
			for sim_no in range(num_simul):
				# print sim_no

				plot_file=path+'plots/'+file_prefix+'_data_'+str(sim_no)+'.jpg'
				data.generate_msg(plot_file,time_span, option='both_endo_exo',p_exo=.2,var=.1, noise_param=noise_param) # modify
				# msg[sim_no]=np.copy(data.msg)
				data.split_data(split_frac)
				save(data,output_file+'n0'+str(noise_param)+'sim'+str(sim_no))
				data.clear_msg()

	#********************************************************************************

	#*************************************ESTIMATION**********************************

	if generate_set_selection_files_flag:
		set_of_noise=[.1,.2,.3,.4,.5]
		num_simul=5
		set_of_lambda=[.5,.7,1,3]
		frac_end=0.8
		set_sel_file_prefix=path+'set_sel_files/'+file_prefix
		var=0.1
		for noise in set_of_noise:
			for sim_no in range(num_simul):
				print 'noise ',noise, ' simulation ',sim_no
				msg_file=file_msg_prefix+'n0'+str(noise)+'sim'+str(sim_no)
				set_sel_file=set_sel_file_prefix+'n0'+str(noise)+'sim'+str(sim_no)
				generate_set_selection_files( msg_file,set_sel_file,set_of_lambda,frac_end,var )
				# set_selection_res[sim_no]=generate_set_selection_files( msg_file,set_sel_file,set_of_lambda,frac_end,var )
				# print sim_no
				# save( set_selection_res, result_set_selection_file)
	
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
	if plot_param_flag:
		file_graph=path+file_prefix+'_graph'
		file_param=path+file_prefix+'_estim_parameter'
		data_graph=load_data(file_graph)
		true_param={'alpha':data_graph.alpha,'A':data_graph.A,'mu':data_graph.mu,'B':data_graph.B}
		mean_res=load_data('mean')
		key='alpha'
		plot_it(mean_res[key],true_param[key],key)


	if param_recovery_flag:
		file_graph=path+file_prefix+'_graph'
		
		# param=load_data(file_estim_param)
		# for key in param['rcpk']:
		# 	print key 
		# return 

		data=load_data(file_graph)
		true_param={'alpha':data.alpha,'A':data.A,'mu':data.mu,'B':data.B}
		num_simul=1
		l =.5#.5#1
		l_sl=.5#.5#3
		file_estim_param=path+file_prefix+'_estim_parameter'+'_l'+str(l)+'_sim'+str(num_simul)
		file_set_sel=path+file_prefix+'_set_selection_result'
		key='A'
		file_param=path+file_prefix+'_'+key #+'_sim100'
		plot_param_exact=path+'plots/'+file_prefix+'_'+key+'_exact_values.jpg'
		plot_param=path+'plots/'+file_prefix+'_'+key+'.jpg'#'_sim100'+

		res_dict_mean=recover_parameters(file_msg_prefix,file_set_sel,file_estim_param,num_simul,l=l,l_sl=l_sl)
		# return
		# res_dict_mean=load_data(file_estim_param)
		# plot_it(res_dict_mean[key],true_param[key],key,plot_param_exact)
		# return
		dist_norm = get_diff_over_method(res_dict_mean[key],true_param[key])
		# plt.bar(dist_norm,'-*',linewidth=3)
		plt.bar(np.arange(len(dist_norm)),dist_norm,width=.25)
		plt.xticks(np.arange(len(dist_norm)),('slant','cpk','rcpk'), fontsize=50)#,'random') )
		# plt.grid(True)
		plt.yticks((8,8.3,8.6), fontsize=50 )
		# plt.title('norm of ||'+key+'_estim-'+key+'_true||', fontsize=50)
		plt.title('Estimation error of '+key, fontsize=50)
		plt.ylim([8,8.6])
		# plt.ylim([.12,.16])
		plt.savefig(plot_param)
		plt.show()
		save(dist_norm,file_param)
	

	if evaluate_intensity_param_flag:
		key='B'
		num_simul=1
		file_data=file_common
		file_graph=path+file_prefix+'_graph'
		plot_param_exact=path+'plots/'+file_prefix+'_'+key+'_exact.jpg'
		data_original=load_data(file_graph)
		true_param={'mu':data_original.mu,'B':data_original.B}

		res_dict={'slant':{}}
		# res_set_selection=load_data(file_set_sel)
		for sim_no in range(num_simul):
			data= load_data(file_data+'_'+str(sim_no))
			res_dict['slant'][sim_no]=run_slant(data, max_iter=10000)
			
		res_dict_mean=compute_mean(res_dict, num_simul)	


		# save( dict_res , file+'_slant')
		plot_it({'slant':res_dict_mean[key]['slant']},true_param[key],key,plot_param_exact)

	if plot_fraction_flag:
		# set_of_lambda=[.5,.7,1,2,3]
		lamb=1 #[.5,.6,.7,.8,.9,1]
		set_sel_file_prefix=path+'set_sel_files/'+file_prefix
		set_of_noise=[.1,.2,.3,.4,.5]
		num_simul=5
		plot_file=path+'plots/'+file_prefix+'set_recovery.jpg'
		recovery_res_file=path+file_prefix+'set_recovery'

		# plot_recovery=path+'plots/'+file_prefix+'_set_sel_recovery_2.jpg'
		# result_set_selection_file=path+file_prefix+'_set_selection_result'
		# plot_fraction_of_recovery(file_common,result_set_selection_file,plot_recovery,set_of_lambda,num_simul)
		get_fraction_of_recovery(file_msg_prefix,set_sel_file_prefix,recovery_res_file,lamb,set_of_noise,num_simul)

	# if plot_fraction_flag_old:
	# 	# set_of_lambda=[.5,.7,1,2,3]
	# 	set_of_lambda=[1]#[.5,.6,.7,.8,.9,1]
	# 	plot_recovery=path+'plots/'+file_prefix+'_set_sel_recovery_2.jpg'
	# 	result_set_selection_file=path+file_prefix+'_set_selection_result'
	# 	plot_fraction_of_recovery(file_common,result_set_selection_file,plot_recovery,set_of_lambda,num_simul)
		
	
	#********************************************************************************************************
	#****************************************************NOT REQUIRED NOW ********************
	if evaluate_synthetic_data_flag:
		num_simul=1
		file='../result_synthetic_dataset/'+generate_str+'_'+str(num_node)+'_'+str(index)
		data=load_data(file)
		slant_obj=slant(init_by='object', obj = data, tuning_param=[10,10,float('inf')])
		# slant_obj=slant(init_by='dict', obj = dict_obj, tuning_param=[10,10,float('inf')])
		dict_res={}
		for l in set_of_lambda:
			print l 
			dict_res[str(l)]=slant_obj.estimate_param(lamb=l) # lambda can be passed here
		save( dict_res , file+'_slant')




	if check_slant_flag:

		file='../result_synthetic_dataset/'+generate_str+'_'+str(num_node)+'_'+str(index)
		data=load_data(file)
		true_parameter={'A':data.A, 'B':data.B, 'alpha':data.alpha, 'mu':data.mu}
		dict_res=load_data(file+'_slant')
		compare(true_parameter, dict_res, 'B',set_of_lambda)

	#****************DONE*****************************************************************************
			
		# print "Recovery statistics : fraction of endogenious msg recovered by our method "
		# print "Cherrypick"
		# print np.count_nonzero(np.logical_and(data.msg_end_tr,msg_end_cpk))[0]/float(np.count_nonzero(data.msg_end_tr)[0])
		# print "Robust Cherrypick"
		# print np.count_nonzero(np.logical_and(data.msg_end_tr,msg_end_cpk)  )[0]/float(np.count_nonzero(data.msg_end_tr)[0])
		#---------------------------------------------------------------------------------------------------------------------






	# if flag_generate_graph_only:
	# 	data = create_synthetic_data( num_node, frac_sparse  )
	# 	data.generate_graph()
	# 	output_file = "synthetic_graph_node_5.obj"
	# 	save(data,output_file)


	# print data.edges
	#a = [0,1,2,3,4,5]
	#rnd.shuffle(a)
	#print type(ceil(1.5))
	
	# print "----------------------------------------"
	# print data.nodes
	# print "----------------------------------------"
	# print data.edges
	#print data.nodes
	#print data.num_node
	#print data.msg
	
	# for i in range(20):
	# 	print rnd.randint(1,5)
	# print data.msg
	# data.split_data(train_fraction)

if __name__=="__main__":
	main()

	# def generate_graph(self):
	# 	# self.edges={}
	# 	# shuffled_items = list(self.nodes)
	# 	# for i in self.nodes:
	# 	# 	#self.edges[i]=list(rnd.choices(self.num_node, int( self.frac_sparse * self.num_node ) ))
	# 	# 	rnd.shuffle(shuffled_items)
	# 	# 	#print shuffled_items
	# 	# 	self.edges[i]=list(shuffled_items[:int( self.frac_sparse * self.num_node )])
	# 	# 	if i in self.edges[i]:
	# 	# 		self.edges[i].remove(i)
	# 	self.edges=np.zeros((self.num_node,self.num_node))
	# 	# num_nbr = int( self.frac_sparse * self.num_node)
	# 	shuffled_items = np.array(self.nodes)
	# 	for i in self.nodes:
	# 		nbr = rnd.randint(self.min_nbr, self.max_nbr )
	# 		if np.count_nonzero(self.edges[i]) < nbr:
	# 			nbr_remain = nbr - np.count_nonzero(self.edges[i,:i])
	# 			sample = np.arange(i+1,self.num_node)
	# 			nbr_indices = np.array(random_sample(sample, nbr_remain))
				
	# 			# print "____________________________"
	# 			# print current_neighbours
	# 			# for v in current_neighbours :
	# 			# 	self.edges[i][v] = 1
	# 			# 	print self.edges[i]
	# 			self.edges[i,nbr_indices]=1
	# 			self.edges[nbr_indices,i]=1
	# 			# print self.edges[i] 
	# 	# np.fill_diagonal( self.edges, 0 )


################## pld fraction plot code segment ###########################################


# def plot_fraction_of_recovery(file_data,file_res,file_plot,set_of_lambda,num_simul):
# 	# plot_fraction_of_recovery(file_msg_prefix,set_sel_file_prefix,recovery_file,lamb,set_of_noise,num_simul)
# 	num_lamb=len(set_of_lambda)
# 	res=np.zeros((num_simul,num_lamb,2))
# 	set_selection_res=load_data(file_res)
# 	# for key in set_selection_res:
# 	# 	print key
# 	for sim_no in range(num_simul):
# 		data=load_data(file_data+'_'+str(sim_no))
# 		curr_res=set_selection_res[sim_no]
# 		# for key in curr_res['rcpk']:
# 		# 	print key
# 		for l,l_no in zip(set_of_lambda,range(num_lamb)):
# 			res[sim_no,l_no,0]=get_recovery_fraction(data.msg_end_tr,curr_res['rcpk'][str(l)])
# 			res[sim_no,l_no,1]=get_recovery_fraction(data.msg_end_tr,curr_res['cpk'][str(l)])		
# 	res_mean=np.mean(res,axis=0)