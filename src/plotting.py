from shutil import copyfile
from shutil import copy 
# from scipy.interpolate import CubicSpline
import datetime
import os 
import sys
import matplotlib 
# matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy.random as rnd
import time
from slant import slant 
from data_preprocess import *
from myutil import *

def find_min_idx(x):
    k = x.argmin()
    ncol = x.shape[1]
    return k/ncol, k%ncol
def get_min( A, tune_nowcasting  = None ):
	# print shape(A)
	num_dim = len( A.shape )
	num_time_span = 6
	v = np.zeros(num_time_span)
	if tune_nowcasting : 
		if num_dim == 2 :
			return A[A[:,0].argmin(),:]
		if num_dim==3:
			i,j = find_min_idx(A[:,:,0])
			return A[i,j,:]


	for i in range(num_time_span):
		if num_dim == 3:
			v[i]=np.min(np.min( A[:,:,i], axis = 0 ), axis = 0 )
		if num_dim == 2 : 
			v[i]=np.min( A[:,i], axis = 0 )
	return v
def plot_MSE( list_of_MSE, file_prefix, file_to_write):
	list_of_error=[]
	for error,param in list_of_MSE: # check
		list_of_error.append( error)
	ind = np.argmin(np.array(list_of_error))
	param=list_of_MSE[ind][1]
	w=param['w']
	v=param['v']
	lamb=param['l']
	f=plt.figure()
	plt.plot(list_of_error)
	plt.text( int( len(list_of_error)/3 ),list_of_error[0], 'w='+ str(w)+',v='+str(v)+',lambda='+str(lamb))	
	plt.xlabel('various combinations')
	plt.ylabel('MSE')
	plt.grid()
	# plt.legend('w='+ str(w)+',v='+str(v)+',lambda='+str(lamb))
	plt.title(file_prefix+':slant tuning results for \nw:[1,2,4,10] \nlambda:[.0001,.0005,.001,.005,.01,.5,1]\n For best MSE, w='+str(w)+',l='+str(lamb))
	plt.savefig( file_to_write )
	plt.clf()

def split(file):
	obj=load_data(file)
	folder='/'.join(file.split('/')[:-1])
	file_prefix=file.split('/')[-1].split('.res.')[0]
	file_suffix='.res.'+file.split('/')[-1].split('.res.')[1]
	
	for sub_obj in obj[:-1]:
		for sub_sub_obj in sub_obj:
			frac=sub_sub_obj['frac_end']
			lamb=sub_sub_obj['lambda']
			file_to_write=folder+'/'+file_prefix+'w10'+'f'+str(frac)+'l'+str(lamb)+file_suffix
			save(sub_sub_obj,file_to_write)

def plot_final_tuned_image(file_prefix,list_of_time_span,file_to_save):
	res=[]
	index_dict=load_data('index_of_lambda_tuned/'+file_prefix+'.slant')
	for time_span_input in list_of_time_span:
		lamb=index_dict[time_span_input]
		file_to_read=f_prefix+'l'+str(lamb)+'t'+str(time_span_input)+'.res.slant'
		res.append(load_data(file_to_read)['MSE'])
	# plot res
	line=plt.semilogy(list_of_time_span, res, label='slant')
	# plt.plot(list_of_time_span[:idx], arr_final)
	plt.setp(line, linewidth=4, color='r', linestyle='--')

	plt.ylim(ymin=0,ymax=.06)
	plt.xlim(xmin=0,xmax=1.0)
	plt.grid(True)
	plt.xlabel('Time span')
	plt.ylabel('MSE')
	plt.title(file_prefix)
	plt.savefig(file_to_save)
	plt.show()
	plt.clf()

def separate_msg(input_directory,output_directory):
	for file in os.listdir(input_directory):
		obj=load_data(input_directory+file[:-4])
		if 'msg_set' in obj:
			file_write_msg=output_directory+file[:-4]+'.msg'
			file_write_obj=output_directory+file[:-4]
			msg=dict(obj['msg_set'])
			del obj['msg_set']
			save(msg, file_write_msg)
			save(obj,file_write_obj)

def ma(y, window):
	avg_mask = np.ones(window) / window
	y_ma=np.convolve(y, avg_mask, 'same')
	y_ma[0]=y[0]
	# if 0 in list(y):
	# 	idx=np.where(y==0)[0][0]
	# 	if idx>0:
	# 		y_ma[idx-1]=y[idx-1]
	y_ma[-1]=y[-1]
	return y_ma

def compute_FR_old( result, threshold):
	set_of_predictions= result['predicted']
	mean_prediction = np.mean( set_of_predictions, axis = 1 ) 
	true_values = result['true_target'] 
	mean_opn=np.mean(true_values)
	mean_prediction -= mean_opn
	true_values -= mean_opn
	num_test=true_values.shape[0]
	flag_array=np.zeros(num_test,dtype=bool)
	#****
	# plt.plot(true_values)
	# plt.plot(mean_prediction)
	# plt.show()
	#***
	# count=0
	for target,idx in zip(true_values,range(num_test)):
		if abs(target ) > threshold: # define threshold  # check abs 
			# if abs(predicted) > threshold:
			flag_array[idx]=True
			# count+=1
			# print 'count'
			# print count
	return get_FR(mean_prediction[flag_array],true_values[flag_array])

def get_value(file_to_read,measure,threshold=0):
	if os.path.isfile(file_to_read+'.pkl'):
		res=load_data(file_to_read)
		if measure=='MSE':
			return res['MSE']
		else:
			return compute_FR(res,threshold)
	else:
		return 0

def get_polarity(s,thres):
	polarity_s=np.zeros(s.shape[0])
	polarity_s[np.where(s>thres)[0]]=1
	polarity_s[np.where(s<-thres)[0]]=-1
	return polarity_s

def compute_FR_part(mean_prediction, true_values, threshold):

	num_test=true_values.shape[0]
	polar_pred=get_polarity(mean_prediction,threshold)
	polar_true=get_polarity(true_values,threshold)
	return float(np.count_nonzero(polar_true-polar_pred))/num_test

def compute_FR( result, threshold):
	# print result.keys()
	set_of_predictions= result['predicted']
	mean_prediction = np.mean( set_of_predictions, axis = 1 ) 
	true_values = result['true_target'] 
	num_test=true_values.shape[0]
	polar_pred=get_polarity(mean_prediction,threshold)
	polar_true=get_polarity(true_values,threshold)
	return float(np.count_nonzero(polar_true-polar_pred))/num_test

def read_result_from_file( num_l, num_t, list_of_lambda,list_of_time_span, file_to_read_prefix,method, measure, threshold=0):
	MSE=np.zeros((num_l,num_t))
	for l_index in range(num_l):
		for t_index in range(num_t):
			l=list_of_lambda[l_index]
			t=list_of_time_span[t_index]
			if method == 'slant':
				file_to_read=file_to_read_prefix+'w0v0l'+str(l)+'t'+str(t)+'.res.slant'
			if method=='cpk':
				file_to_read=file_to_read_prefix+'w10v10f0.8lc'+str(l)+'ls'+str(l)+'t'+str(t)+'.res.cherrypick.slant'
			if method=='rcpk':
				file_to_read=file_to_read_prefix+'w10v10f0.8ls'+str(l)+'t'+str(t)+'.res.Robust_cherrypick.slant'
			if os.path.isfile(file_to_read+'.pkl'):
				# print 'yes'
				res=load_data(file_to_read)
				if measure=='FR':
					# print 'compute FR'
					MSE[l_index, t_index]=compute_FR(res,threshold) # ['MSE']
					# print compute_FR(res,threshold)
				if measure=='MSE':
					MSE[l_index, t_index]=res['MSE']
			else:
				print 'Error: not found *****************************'
				print file_to_read
				print 'method' + method + ' l='+str(l)+' t='+str(t)
	return MSE

def print_subset_sel_parameter( file_prefix_list, file_prefix ):
	# print "method 0 for cpk 1 for rcpk . curr method : "+str(method)
	for file in file_prefix_list:
		file_to_read = file_prefix+file+'_desc'
		print "____________________________________________________"
		print file 
		print "____________________________________________________"
		if os.path.isfile(file_to_read+'.pkl'):
			index=load_data(file_to_read)
			# print type(index)
			print 'cpk' + str(index['cpk'][2]) + '--------- rcpk : ' + str(index['rcpk'][2]) + ' ------- thr :  ' + str(index['threshold'])
			# for key,value in index.iteritems():
			# 	print key,value
				# if key==0.2:
				# 	print str(key) + ':cpk : ------ ' +  str(value[1]) + ': rcpk : ------- ' + str(value[2])
			# if method==0:
			# 	param = index['cpk'][2]
			# else:
			# 	param = index['rcpk'][2]
			# print "File : " + file + "--------------------------   " + str(param)
		else:
			print "File: " + file

def plot_variation_fraction_old(file_to_read_prefix, list_of_lambda, l_3, l_5, list_of_fraction, file_to_write, file_to_write_desc,measure,time_span,threshold=0):
	list_of_fraction_plot=map(int,100*np.array(list_of_fraction))
	# print list_of_fraction_plot
	file_to_read_pre=file_to_read_prefix.split('w10v10')[0]+'w10v10'
	file_to_read_post=file_to_read_prefix.split('w10v10')[1]
	file_prefix=file_to_read_prefix.split('w10v10')[0].split('/')[-1]
	method=file_to_read_prefix.split('.res.')[1].split('.slant')[0]
	num_l=len(list_of_lambda)
	num_f=len(list_of_fraction)
	Error=np.zeros((num_l,num_f))
	for l_idx in range(num_l):
		for f_idx in range(num_f):
			if method=='Robust_cherrypick':
				file_to_read=file_to_read_pre+'f'+str(list_of_fraction[f_idx])+'ls'+str(list_of_lambda[l_idx])+file_to_read_post 
			else:
				file_to_read=file_to_read_pre+'f'+str(list_of_fraction[f_idx])+'lc'+str(list_of_lambda[l_idx])+'ls'+str(list_of_lambda[l_idx])+file_to_read_post 
			# print file_to_read
			if not os.path.isfile(file_to_read+'.pkl'): 
				if list_of_fraction[f_idx]==.8 and list_of_lambda[l_idx]==l_3:
					file_to_read='../result_performance_forecasting/'+file_prefix+'/'+file_prefix+'w10v10'+file_to_read.split('w10v10')[-1]
					if not os.path.isfile(file_to_read+'.pkl'):
						file_to_read='../result_performance_forecasting/'+file_prefix+'/'+file_prefix+'w10v10'+'f'+str(list_of_fraction[f_idx])+'lc'+str(int(list_of_lambda[l_idx]))+'ls'+str(int(list_of_lambda[l_idx]))+file_to_read_post 
				if list_of_fraction[f_idx]==1.0 and list_of_lambda[l_idx]==l_5:
					file_to_read='../result_performance_forecasting/'+file_prefix+'/'+file_prefix+'w0v0l'+str(l_5)+'t'+str(time_span)+'.res.slant'
					if not os.path.isfile(file_to_read+'.pkl'):
						file_to_read='../result_performance_forecasting/'+file_prefix+'/'+file_prefix+'w0v0l'+str(int(l_5) )+'t'+str(time_span)+'.res.slant'
			if os.path.isfile(file_to_read+'.pkl'):
				Error[l_idx,f_idx]=get_value(file_to_read,measure,threshold)
			else:
				print 'Not found: '+ file_to_read
	print 'l(3,5)='+str(l_3)+','+str(l_5)
	flag_sel_all=0
	if flag_sel_all==1:
		for l_idx in range(num_l):
			# plt.plot(2,2,l_idx+1)
			plt.semilogy(list_of_fraction_plot,Error[l_idx], label='l='+str(list_of_lambda[l_idx])) # print slant end results here
			plt.ylim(ymin=.065,ymax=.075)	
			# plt.grid(True)
			# plt.xlabel('Fraction')
			plt.ylabel(measure)
		plt.title(file_prefix)
		plt.legend()
		plt.savefig(file_to_write+'_all.jpg')
		plt.show()
		plt.clf()
		return
	l_tuned=l_3*np.ones(num_f)
	# l_tuned[1]=.5
	l_tuned[-1]=np.array([l_5])
	save(l_tuned,file_to_write_desc)
	final_val=[]
	for l,f_idx in zip(l_tuned,range(num_f)):
		final_val.append(Error[list_of_lambda.index(l),f_idx])
	line=plt.semilogy(list_of_fraction_plot, final_val, label=method)
	plt.setp(line, linewidth=3,linestyle='-',marker='o', markersize=10)
	ymin=0.03
	ymid=.035
	ymax=.04
	# plt.ylim(ymin=ymin,ymax=ymax)	
	plt.xlim(xmin=50,xmax=100)
	plt.yticks([ymax,ymid,ymin])
	plt.grid(True)
	plt.legend()
	# plt.xticks(ll,l7,rotation=0,fontsize=22.7,weight='bold')
	
	plt.xlabel('Fraction')
	plt.ylabel(measure)
	plt.title(file_prefix)
	plt.savefig(file_to_write+'.jpg')
	# plt.savefig('../Plots/slant_tuned/'+measure+'/'+file_prefix+'_slant_tuned_0'+ext+'.jpg')
	plt.show()
	plt.clf()

def plot_final_results( file,file_to_load, file_to_plot):
	res = load_data(file_to_load)
	# res['label']=['slant Hawkes','cherrypick Hawkes','Robust cherrypick Hawkes','cherrypick Poisson', 'Robust cherrypick Poisson','slant Poisson' ]
	# dict_res={'x-axis':list_of_time_span, 'y-axis':{'MSE':list_res_MSE,'FR':list_res_FR}, 'threshold':threshold, 'labels':labels}	
	f=plt.figure()
	marker_seq=['P','o','*','o','*','P']
	ls_seq=[':','-.','--','-.','--',':']
	x=res['x-axis']
	y=res['y-axis']['MSE']
	labels=res['labels']
	for data,label,mk,ls in zip(y,labels,marker_seq,ls_seq):
			# plt.plot( ma(data[:6],3), label=label)
			line=plt.plot(ma(data,3), label=label)
			plt.setp(line, linewidth=4,linestyle=ls,marker=mk, markersize=10)
	plt.xticks(np.arange(6),np.array([0,.1,.2,.3,.4,.5]),rotation=0,fontsize=22.7,weight='bold')
	# plt.grid()
	# plt.legend()

	# plt.ylim(ymin=ymin,ymax=ymax)	
	# # plt.xlim(xmax=0.5)
	plt.grid(True)
	plt.legend()
	# plt.xticks(ll,l7,rotation=0,fontsize=22.7,weight='bold')
	# yt=plt.yticks
	# plt.yticks([])
	ytick_val=np.arange(.066,.079,.004)
	res['ytick_val']=ytick_val
	# plt.yticks(res['ytick_val'],rotation=0,fontsize=20.7,weight='bold')
	# plt.yticks(res['ytick_val'],rotation=0,fontsize=15.7,weight='bold')


	# plt.ylim([0,1])
	# 
	plt.yticks(ytick_val,rotation=0,fontsize=20.7,weight='bold')
	# plt.yticks('manual')
	# res['ytick_val']=ytick_val
	# plt.yticks([ymax,ymid,ymin])
	plt.xlabel('Time span',rotation=0,fontsize=20.7,weight='bold')
	# plt.ylabel(file, fontsize=22.7,weight='bold')
	plt.title(file,rotation=0,fontsize=20.7,weight='bold')

	plt.tight_layout()
	plt.savefig(file_to_plot)
	# plt.savefig(file_to_plot)
	plt.show()
	plt.clf()
	save(res, file_to_load)
		
def plot_slant_result_with_lambda_manual_tuning_v2(file_to_read_prefix, list_of_lambda, list_of_time_span, file_to_write,window,measure,threshold,file_to_save_plot):
	num_l=len(list_of_lambda)
	num_t=len(list_of_time_span)
	num_method=3

	file_prefix=file_to_read_prefix.split('w0v0')[0].split('/')[-1]
	MSE=np.zeros((num_l,num_t,num_method))
	MSE[:,:,0]=read_result_from_file( num_l, num_t, list_of_lambda,list_of_time_span, file_to_read_prefix,'slant', measure,threshold)	
	# return
	MSE[:,:,1]=read_result_from_file( num_l, num_t, list_of_lambda,list_of_time_span, file_to_read_prefix,'cpk', measure,threshold)
	MSE[:,:,2]=read_result_from_file( num_l, num_t, list_of_lambda,list_of_time_span, file_to_read_prefix,'rcpk', measure,threshold)
	# print MSE.shape
	for l_index in range(num_l):
		for i in range(num_method):
			# print MSE[l_index,:,i].shape
			# print MSE[l_index,:,i].shape
			MSE[l_index,:,i]=ma(MSE[l_index,:,i],window) 
	#*****************************************************************************************************************
	flag_show_all=0
	# [.1,.2,.3,.4,.5,.6,.7,.8,1.3,1.5,1.7,2]
	list_of_l_sl=[.7]#,1.1]#,.7,.9,1.1]
	list_of_l_cpk=[1.1]#,1.1]#.7,.8,.9,1.1]#[1.3,1.5,1.7,2]
	list_of_l_rcpk=[2]#,.7,.8,.9,1.1,1.3]#[1.3,1.5,1.7,2]#[.1,.3,.5,1.3,1.5,2]
	# print 'sl--------------------cpk------rcpk'
	# print str(MSE[list_of_l_sl[0],0,0])+'---'+str(MSE[list_of_l_cpk[0],0,1])+'---'+str(MSE[list_of_l_rcpk[0],0,2])
	if flag_show_all==1:
		f=plt.figure()
		for l_index in range(num_l):
			l=list_of_lambda[l_index]
			if l in list_of_l_sl:
				line_sl=plt.semilogy(list_of_time_span, MSE[l_index,:,0], label='slant,l='+str(list_of_lambda[l_index]))
				plt.setp(line_sl, linewidth=1, linestyle='--')
				print 'slant:lamb:'+str(l)+':FR:'+str(MSE[l_index,0,0])
			if l in list_of_l_cpk:					
				line_cpk=plt.semilogy(list_of_time_span, MSE[l_index,:,1], label='cpk,l='+str(list_of_lambda[l_index]))
				plt.setp(line_cpk, linewidth=2, linestyle='-.')
				print 'cpk:lamb:'+str(l)+':FR:'+str(MSE[l_index,0,1])

			if l in list_of_l_rcpk:
				line_rcpk=plt.semilogy(list_of_time_span, MSE[l_index,:,2], label='rcpk,l='+str(list_of_lambda[l_index]))
				plt.setp(line_rcpk, linewidth=3, linestyle='-')
				print 'rcpk:lamb:'+str(l)+':FR:'+str(MSE[l_index,0,2])
		plt.legend()
		plt.ylabel(measure)
		plt.xlabel('Time')
		plt.grid(True)
		plt.ylim(ymin=0.1,ymax=0.2)
		# plt.xlim(xmax=.5)
		plt.title(file_prefix)
		plt.savefig(file_to_save_plot+'_all.jpg')
		# plt.savefig('../Plots/slant_tuned/'+measure+'/'+file_prefix+'_slant_cpk_all_0.jpg')
		plt.show()
		plt.clf()
		return
	#**************************************************************************************
	idx=6
	l_tuned_sl=1.1*np.ones(idx)
	l_tuned_sl[:6]=np.array([1.1,.9,.8,.7,.6,.5])
	l_tuned_rcpk=0.6*np.ones(idx)
	l_tuned_rcpk[0]=np.array([1.1])
	# [.7,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1]
	l_tuned_cpk=2*np.ones(idx)
	# l_tuned_cpk[:1]=np.array([1.1])#.9
	#*********
	if os.path.isfile(file_to_write):
		os.remove(file_to_write)
	index_dict={}
	index_dict['slant']=l_tuned_sl
	index_dict['cpk']=l_tuned_cpk
	index_dict['rcpk']=l_tuned_rcpk
	index_dict['idx']=idx
	# for tm,lamb_sl,lamb_cpk,lamb_rcpk in zip(list_of_time_span,l_tuned_sl,l_tuned_cpk,l_tuned_rcpk):
	# 	index_dict[tm]=[lamb_sl,lamb_cpk,lamb_rcpk]  # list_of_lambda[idx]
	#***************************************************************************************
	arr_final=np.zeros((3,idx) )
	for t_index in range(idx):
		arr_final[0,t_index]=MSE[ list_of_lambda.index(l_tuned_sl[t_index]) ,t_index,0]
		arr_final[1,t_index]=MSE[ list_of_lambda.index(l_tuned_cpk[t_index]) ,t_index,1]
		arr_final[2,t_index]=MSE[ list_of_lambda.index(l_tuned_rcpk[t_index]) ,t_index,2]
	# for l_index in range(3,num_l,1):
	print arr_final[:,0]
	list_of_label =['slant','cpk','rcpk']
	linewidth=3
	for i,l_string in zip(range(3),list_of_label):
		line=plt.semilogy(list_of_time_span[:idx], arr_final[i,:idx], label=list_of_label[i])
		plt.setp(line, linewidth=linewidth,linestyle='-',marker='o', markersize=10)
		linewidth+=1
	ymin=.04
	ymid=.05
	ymax=.06
	plt.ylim(ymin=ymin,ymax=ymax)	
	# plt.xlim(xmax=0.5)
	plt.grid(True)
	plt.legend()
	# plt.xticks(ll,l7,rotation=0,fontsize=22.7,weight='bold')
	
	plt.yticks([ymax,ymid,ymin])
	plt.xlabel('Time span')
	plt.ylabel(measure)
	plt.title(file_prefix)
	plt.savefig(file_to_save_plot+'_final.jpg')
	# plt.savefig('../Plots/slant_tuned/'+measure+'/'+file_prefix+'_slant_tuned_0'+ext+'.jpg')
	plt.show()
	plt.clf()
	index_dict['threshold']=threshold
	index_dict['ymin']=ymin
	index_dict['ymax']=ymax
	index_dict['ymid']=ymid
	save(index_dict,file_to_write)

def read_from_file( file_to_read_prefix, l, t, method,measure):
	if l==1.0 or l==2.0:
		l=int(l)
	if method==0:
		file_to_read=file_to_read_prefix+'w0v0l'+str(l)+'t'+str(t)+'.res.slant'
	if method==1:
		file_to_read=file_to_read_prefix+'w10v10f0.8lc'+str(l)+'ls'+str(l)+'t'+str(t)+'.res.cherrypick.slant'
	if method==2:
		file_to_read=file_to_read_prefix+'w10v10f0.8ls'+str(l)+'t'+str(t)+'.res.Robust_cherrypick.slant'
	if os.path.isfile(file_to_read+'.pkl'):
		res=load_data(file_to_read)
		if measure=='FR':
			return compute_FR(res,threshold) 
		if measure=='MSE':
			return res['MSE']
	else:
		print 'Error: not found *****************************'
		print file_to_read
		print 'method' + str(method) + ' l='+str(l)+' t='+str(t)
		return 0
	
def save_result_forecasting_performance(file_to_read_prefix, list_of_time_span, file_to_read_desc,measure,threshold, result_file):
	index_dict=load_data(file_to_read_desc) 
	#****
	result_dict=load_data
	threshold=index_dict['threshold']
	#***
	file_prefix=file_to_read_prefix.split('/')[-1]
	idx=index_dict['idx']
	l_tuned_sl=index_dict['slant']
	l_tuned_rcpk=index_dict['rcpk']
	l_tuned_cpk=index_dict['cpk']
	list_of_label =['slant','cpk','rcpk']
	arr_final=np.zeros((3,idx) )
	for t_index in range(idx):
		arr_final[0,t_index]=read_from_file( file_to_read_prefix, l_tuned_sl[t_index],list_of_time_span[t_index],0,measure) #MSE[ list_of_lambda.index(l_tuned_sl[t_index]) ,t_index,0]
		arr_final[1,t_index]=read_from_file( file_to_read_prefix, l_tuned_cpk[t_index],list_of_time_span[t_index],1,measure) #MSE[ list_of_lambda.index(l_tuned_sl[t_index]) ,t_index,0]
		arr_final[2,t_index]=read_from_file( file_to_read_prefix, l_tuned_rcpk[t_index],list_of_time_span[t_index],2,measure) #MSE[ list_of_lambda.index(l_tuned_sl[t_index]) ,t_index,0]
	print arr_final
	result_dict={'index':index_dict, 'x-axis': list_of_time_span[:idx], 'y-axis':{'MSE':arr_final}, 'label':list_of_label}
	save( result_dict, result_file) 

def save_variation_fraction(list_of_fraction,method,time_span, desc_read,threshold, file_write,file_read_prefix, file_read_suff,measure_to_plot):
	n_fraction=len(list_of_fraction)
	desc_array=load_data(desc_read)
	MSE=np.zeros(n_fraction)
	FR=np.zeros(n_fraction)
	for lamb,pos,frac in zip(desc_array,range(n_fraction),list_of_fraction):
		if lamb==1.0 or lamb==2.0:
			lamb=int(lamb)
		if method=='cherrypick':
			file_read=file_read_prefix+'f'+str(frac)+'lc'+str(lamb)+'ls'+str(lamb)+file_read_suff 
		else:
			file_read=file_read_prefix+'f'+str(frac)+'ls'+str(lamb)+file_read_suff 

		res=load_data(file_read)
		# print type(res)
		MSE[pos]=res['MSE']
		FR[pos]=compute_FR( res, threshold) # get_FR(res,threshold) # res['FR']
	if measure_to_plot==0:
		plt.plot(MSE)
		print MSE
	else:
		plt.ylim([.1,.3])
		plt.plot(FR)
		print FR
	plt.savefig('buffer.jpg')
	output={'y-axis':{'MSE':MSE,'FR':FR},'lambda':desc_array,'threshold':threshold,'x-axis':list_of_fraction}
	save(output,file_write)
		
def plot_variation_fraction( file,file_to_load, file_to_plot):
	
	measure=list(['MSE','FR'])[int(sys.argv[3])]
	res_dict=load_data(file_to_load)
	f=plt.figure()
	line=plt.plot(ma(res_dict['y-axis'][measure][1:],3))
	# line=plt.plot(res_dict['y-axis'][measure][1:])
	plt.setp(line, linewidth=4,linestyle='-',marker='o', markersize=10)
	plt.xticks(range(5),res_dict['x-axis'][1:],rotation=0,fontsize=22.7,weight='bold')
	plt.grid()
	plt.xlabel(file)
	# plt.legend()

	# plt.ylim(ymin=ymin,ymax=ymax)	
	# # plt.xlim(xmax=0.5)
	# plt.grid(True)
	# plt.legend()
	# plt.xticks(ll,l7,rotation=0,fontsize=22.7,weight='bold')
	# yt=plt.yticks
	# plt.yticks([])
	# ytick_val=np.arange(.15,.21,.01)
	plt.yticks(res_dict['ytick_val'],rotation=0,fontsize=15.7,weight='bold')
	# plt.yticks(ytick_val,rotation=0,fontsize=15.7,weight='bold')
	# plt.yticks('manual')
	# res_dict['ytick_val']=ytick_val
	# plt.yticks([ymax,ymid,ymin])
	# plt.xlabel('Time span')
	# plt.ylabel('MSE', fontsize=22.7,weight='bold')
	plt.title('Cherrypick \n MSE',rotation=0,fontsize=15.7,weight='bold')
	plt.tight_layout()
	plt.savefig(file_to_plot+'_'+measure+'.jpg')
	plt.show()
	plt.clf()
	save(res_dict, file_to_load)

def save_all_result_forecasting_poisson(file_prefix, list_of_lambda, list_of_time_span, list_of_method, file_to_save ):
	result_dict={'x-axis':list_of_time_span, 'y-axis':{}, 'list_of_lambda':list_of_lambda}

	for method in list_of_method:
		res_list=[]
		for lamb in list_of_lambda:
			lamb_list=[]
			for time in list_of_time_span:
				if method=='cherrypick':
					result_file_read=file_prefix+'lc'+str(lamb)+'ls'+str(lamb)+'t'+str(time)+'.res.cherrypick.slant.Poisson'
				else:
					result_file_read=file_prefix+'ls'+str(lamb)+'t'+str(time)+'.res.Robust_cherrypick.slant.Poisson'
				if os.path.isfile(result_file_read+'.pkl'):
					lamb_list.append(load_data(result_file_read)['MSE'])
				else:
					print 'Not found', file_prefix, '\t\t', 'lamb:  ', lamb, 'time:  ', time, ' method:  ', method
					lamb_list.append(0)
			res_list.append(lamb_list)
		result_dict['y-axis'][method]=np.array(res_list)
	save(result_dict,file_to_save)

def save_slant_poisson(file_prefix, list_of_lambda, list_of_time_span, file_to_save ):
	result_dict=load_data(file_to_save)
	method='slant'	

	res_list=[]
	for lamb in list_of_lambda:
		lamb_list=[]
		for time in list_of_time_span:
			result_file_read=file_prefix+'l'+str(lamb)+'t'+str(time)+'.res.slant.Poisson'

			if os.path.isfile(result_file_read+'.pkl'):
				lamb_list.append(load_data(result_file_read)['MSE'])
			else:
				print 'Not found', file_prefix, '\t\t', 'lamb:  ', lamb, 'time:  ', time, ' method:  ', method
				lamb_list.append(0)
		res_list.append(lamb_list)

	# result_dict['y-axis'][method]=np.array(res_list)
	# save(result_dict,file_to_save)
	

	# saved_data=np.copy(result_dict['y-axis'][method])
	# result_dict['y-axis'][method]=np.concatenate((np.array(res_list),saved_data))
	# # result_dict['list_of_lambda']=[.2].append(result_dict['list_of_lambda'])
	# result_dict['list_of_lambda'].insert(0,.1)
	# print result_dict['y-axis'][method]
	# print result_dict['list_of_lambda']
	save(result_dict,file_to_save)

def plot_and_select_lambda_slant( file_incomplete, file_Poisson_slant, file_to_save):
	res_incomplete=load_data(file_incomplete)
	res_Poisson_slant=load_data(file_Poisson_slant)
	
	if int(sys.argv[2])==0: # plot 
		f=plt.figure()
		res_incomplete['label']=['slant','cherrypick','Robust_cherrypick','cherrypick Poisson','Robust_cherrypick Poisson']
		x=res_incomplete['x-axis'][:6]
		for data,label in zip( res_incomplete['y-axis']['MSE'],res_incomplete['label']):
			# plt.plot( ma(data[:6],3), label=label)
			line=plt.plot(ma(data[:6],3), label=label)
			plt.setp(line, linewidth=4,linestyle='-',marker='o', markersize=10)

		# for method in list_of_method:
		
		for data,label in zip(res_Poisson_slant['y-axis']['slant'],res_Poisson_slant['list_of_lambda']):
			data=np.concatenate(([res_incomplete['y-axis']['MSE'][0][0]],data))
			plt.plot( ma(data,3), label= 'slant Poisson,l:' + str(label))
		# plt.ylim([.037,.041])
		plt.legend()
		plt.show()
		# plt.savefig('buffer.jpg')
	else:
		l_sl_no = input('l slant no   ')
		l_sl=input('l slant  ')
		# print type(lamb)
		res_incomplete['label']=['slant','cherrypick','Robust_cherrypick','cherrypick Poisson','Robust_cherrypick Poisson','slant Poisson']
		# res_incomplete['label'].append(['Poisson slant'])
		data=np.zeros((6,6) )
		data[:5]=res_incomplete['y-axis']['MSE'][:,:6]
		data[5,0]=res_incomplete['y-axis']['MSE'][0][0]
		data[5,1:]=res_Poisson_slant['y-axis']['slant'][l_sl_no]
		res_incomplete['y-axis']['MSE']=data
		res_incomplete['poisson_lambda']['slant']=l_sl
		save(res_incomplete, file_to_save)

def plot_and_select_lambda_slant_complete( file_complete, file_to_save):
	res_incomplete=load_data(file_incomplete)
	# res_Poisson_slant=load_data(file_Poisson_slant)
	
	if int(sys.argv[2])==0: # plot 
		f=plt.figure()
		res_incomplete['label']=['slant','cherrypick','Robust_cherrypick','cherrypick Poisson','Robust_cherrypick Poisson','slant Poisson']
		x=res_incomplete['x-axis'][:6]
		for data,label in zip( res_incomplete['y-axis']['MSE'],res_incomplete['label']):
			# plt.plot( ma(data[:6],3), label=label)
			line=plt.plot(ma(data[:6],3), label=label)
			plt.setp(line, linewidth=4,linestyle='-',marker='o', markersize=10)

		# for method in list_of_method:
		
		for data,label in zip(res_Poisson_slant['y-axis']['slant'],res_Poisson_slant['list_of_lambda']):
			data=np.concatenate(([res_incomplete['y-axis']['MSE'][0][0]],data))
			plt.plot( ma(data,3), label= 'slant Poisson,l:' + str(label))
		# plt.ylim([.037,.041])
		plt.legend()
		plt.show()
		# plt.savefig('buffer.jpg')
	# else:
	# 	l_sl_no = input('l slant no   ')
	# 	l_sl=input('l slant  ')
	# 	# print type(lamb)
	# 	res_incomplete['label']=['slant','cherrypick','Robust_cherrypick','cherrypick Poisson','Robust_cherrypick Poisson',]
	# 	# res_incomplete['label'].append(['Poisson slant'])
	# 	data=np.zeros((6,6) )
	# 	data[:5]=res_incomplete['y-axis']['MSE'][:,:6]
	# 	data[5,0]=res_incomplete['y-axis']['MSE'][0][0]
	# 	data[5,1:]=res_Poisson_slant['y-axis']['slant'][l_sl_no]
	# 	res_incomplete['y-axis']['MSE']=data
	# 	res_incomplete['poisson_lambda']['slant']=l_sl
	# 	save(res_incomplete, file_to_save)

def read_lambda(dict_lamb, keyword, list_of_methods,list_of_time_span):
	def alias(str1):
		if str1=='Robust_cherrypick':
			return 'rcpk'
		if str1=='cherrypick':
			return 'cpk'
		return str1

	if 'idx' in dict_lamb:
		print dict_lamb['idx']
	if 'threshold' in dict_lamb:
		print dict_lamb['threshold']
	num_point=len(list_of_time_span)

	res_dict_lamb={'slant':{},'cherrypick':{},'Robust_cherrypick':{}}
	for method in list_of_methods:
		if keyword=='poisson':
			for time in list_of_time_span:	
				res_dict_lamb[method][time]=dict_lamb['poisson_lambda'][method]
		if keyword=='hawkes':
			for time, lamb_at_that_time in zip(list_of_time_span,dict_lamb[alias(method)][:num_point]):
				res_dict_lamb[method][time]=lamb_at_that_time
	return res_dict_lamb

def plot_result_forecasting_FR(file_prefix, file_to_save, flag_save,threshold, frac, list_of_process, list_of_methods,list_of_time_span, hawkes_lambda_file, poisson_lambda_file):
	hawkes_lambda=read_lambda(load_data(hawkes_lambda_file),'hawkes',list_of_methods, list_of_time_span)
	poisson_lambda=read_lambda(load_data(poisson_lambda_file),'poisson',list_of_methods, list_of_time_span)
	dict_lamb={'hawkes':hawkes_lambda,'poisson':poisson_lambda}

	labels=[ process+'_'+method  for process in list_of_process for method in list_of_methods ]
	# print labels
	list_res=[]
	for process in list_of_process:
		for method in list_of_methods:
			list_res_each_plot=[]
			for time_span in list_of_time_span:

				# print process, method, time_span
				lamb=dict_lamb[process][method][time_span]
				if lamb==1.0 or lamb==2.0:
					lamb=int(lamb)
				if process=='poisson' and time_span!=0.0:
					file_read_prefix = '../result_forecasting_Poisson/'+file_prefix+'w10v10'
					if method =='Robust_cherrypick':
						file_read = file_read_prefix+'f0.8ls'+str(lamb)+'t'+str(time_span)+'.res.'+method+'.slant.Poisson'
					if method =='cherrypick':#,'Robust_cherrypick']:
						file_read = file_read_prefix+'f0.8lc'+str(lamb)+'ls'+str(lamb)+'t'+str(time_span)+'.res.'+method+'.slant.Poisson'
					if method=='slant':
						file_read = file_read_prefix+'l'+str(lamb)+'t'+str(time_span)+'.res.slant.Poisson'
					
			
				if process=='hawkes' or time_span==0.0:
					if process=='poisson':
						lamb=dict_lamb['hawkes'][method][time_span]
						if lamb==1.0 or lamb==2.0:
							lamb=int(lamb)
					file_read_prefix='../result_performance_forecasting/'+file_prefix+'/' + file_prefix 
					if method=='slant':
						file_read=file_read_prefix+'w0v0l'+str(lamb)+'t'+str(time_span)+'.res.slant'
					if method=='cherrypick':
						file_read=file_read_prefix+'w10v10f0.8lc'+str(lamb)+'ls'+str(lamb)+'t'+str(time_span)+'.res.cherrypick.slant'
					if method=='Robust_cherrypick':
						file_read=file_read_prefix+'w10v10f0.8ls'+str(lamb)+'t'+str(time_span)+'.res.Robust_cherrypick.slant'
				# print file_read
				res=load_data(file_read,'ifexists')
				if not res:
					print 'Not found: ', file_read
				# MSE[pos]=res['MSE']
				res_FR=compute_FR( res, threshold) 
				list_res_each_plot.append(res_FR)
			list_res.append( list_res_each_plot)

	
	if flag_save:
		dict_res={'x-axis':list_of_time_span, 'y-axis':np.array(list_res), 'threshold':threshold, 'labels':labels}
		save(dict_res,file_to_save)
	else:
		plot_matrix(list_res, labels)
	return np.array(list_res)

def plot_matrix(data, labels, name):
	for row,l in zip(data,labels):
		plt.plot(ma(row,3),label=l)
		# plt.semilogy(ma(row,3),label=l)
	# plt.ylim([0.12,.15])
	plt.legend()
	plt.tight_layout()
	plt.savefig(name)
	plt.show()  

def plot_and_select_lambda( file_Hawkes, file_Poisson, list_of_method, file_to_save, flag_save):
	res_Hawkes=load_data(file_Hawkes)
	res_Poisson=load_data(file_Poisson)
	
	if not flag_save: # plot 
		f=plt.figure()

		x=res_Hawkes['x-axis'][:6]
		for data,label in zip( res_Hawkes['y-axis']['MSE'],res_Hawkes['label']):
			# plt.plot( ma(data[:6],3), label=label)
			line=plt.plot(ma(data[:6],3), label=label)
			plt.setp(line, linewidth=4,linestyle='-',marker='o', markersize=10)

		for method in list_of_method:
			for data,label in zip(res_Poisson['y-axis'][method],res_Poisson['list_of_lambda']):
				if method == 'cherrypick':
					data=np.concatenate(([res_Hawkes['y-axis']['MSE'][1][0]],data))
				else:
					data=np.concatenate(([res_Hawkes['y-axis']['MSE'][2][0]],data))
				plt.plot( ma(data,3), label=method + ',l:' + str(label))
		# plt.ylim([.037,.041])
		plt.legend()
		plt.show()
	else:
		l_cpk_no = input('l cpk no   ')
		l_cpk=input('l cpk  ')
		l_rcpk_no=input('l rcpk no   ')
		l_rcpk=input('l rcpk')
		res_Hawkes['ymin']=input('ymin ')
		res_Hawkes['ymax']=input('ymax ')
		# print type(lamb)
		res_Hawkes['label'].append(['cherrypick','Robust_cherrypick'])
		data=np.zeros((5,6) )
		data[:3]=res_Hawkes['y-axis']['MSE'][:,:6]
		data[3,0]=res_Hawkes['y-axis']['MSE'][1][0]
		data[4,0]=res_Hawkes['y-axis']['MSE'][2][0]
		data[3,1:]=res_Poisson['y-axis']['cherrypick'][l_cpk_no]
		data[4,1:]=res_Poisson['y-axis']['Robust_cherrypick'][l_rcpk_no]
		res_Hawkes['y-axis']['MSE']=data
		res_Hawkes['poisson_lambda']={'cherrypick':l_cpk,'Robust_cherrypick':l_rcpk}
		save(res_Hawkes, file_to_save)
	
def plot_matrix_all_result(file_prefix, data_list, labels, n_lamb, name):
	fig,axes=plt.subplots(1,2)
	ls_seq=[':','-.','--']
	ls_seq.extend([ i  for i in [':','-.','--'] for n in range(n_lamb)])
	lw_seq=[1,1,1]
	lw_seq.extend([ i  for i in range(2,5,1) for n in range(n_lamb)])
	# print len(ls_seq)
	# print len(lw_seq)
	plt.title(file_prefix)

	# for ind,data in zip(range(2),data_list):
	for i,data in zip([0,1],data_list):		
		# print i
		for row,l,lw,ls in zip(data,labels,lw_seq,ls_seq):
			axes[i].plot(ma(row,3),label=l,linewidth=lw,linestyle=ls,marker='o', markersize=4)
		# plt.plot(ma(row,3),label=l, linewidth=lw,linestyle=ls,marker='o', markersize=10)
	# 	# axes[ind].setp(line, linewidth=lw,linestyle=ls,marker='o', markersize=10)
		# axes[0].legend()
	# axes[0].set_ylim([.02,.025])#([.04,.07])#([.05,.1])
	# axes[1].set_ylim([.17,.19])#([.1,.25])#([.1,.25])

	# axes[0].plot(range(6))
	# axes[1].plot(range(5))
	# plt.tight_layout()
	plt.savefig(name)
	plt.show()  

def plot_final_results_FR(file_prefix,file_to_load,file_to_plot,measure):

	# dict_res={'x-axis':list_of_time_span, 'y-axis':{'MSE':list_res_MSE,'FR':list_res_FR}, 'threshold':threshold, 'labels':labels}
	res=load_data(file_to_load)
	x=res['x-axis']
	y=res['y-axis'][measure] # remove if you use old version
	labels=res['labels']
	seq=[0,1,2,4,5,3]
	marker_seq=['P','o','*','o','*','P']
	ls_seq=[':','-.','--','-.','--',':']
	f,ax=plt.subplots()

	for ind,mk,ls in zip(seq,marker_seq,ls_seq):
			line=plt.plot(ma(y[ind],3), label=labels[ind])
			plt.setp(line, linewidth=4,linestyle=ls,marker=mk, markersize=10)
	plt.xticks(np.arange(6),x,rotation=0,fontsize=22.7,weight='bold')
	plt.grid()
	plt.legend()

	# plt.yticks(res['ytick_val'],rotation=0,fontsize=15.7,weight='bold')
	# plt.ytick.remove()
	# ax.set_yticks(ytick_val,rotation=0,fontsize=15.7,weight='bold')
	# ax.tick_params(labelleft=False,labelright=False)    
	# labels = [item.get_text() for item in ax.get_yticklabels()]
	# empty_string_labels = ['']*len(labels)
	# ax.set_yticklabels(empty_string_labels)
	ytick_val=[.01*i for i in range(10,16,1)]#np.arange(.06,.1,.01)
	
	# ytick_val=res['ytick_val']
	# res['ytick_val']=ytick_val
	plt.yticks(ytick_val,rotation=0,fontsize=20.7,weight='bold')
	plt.xlabel('Time span',rotation=0,fontsize=20.7,weight='bold')
	# plt.ylabel('FR', fontsize=26.7,weight='bold')
	plt.title(file_prefix,rotation=0,fontsize=20.7,weight='bold')
	plt.tight_layout()
	plt.savefig(file_to_plot)
	# plt.savefig(file_to_plot)
	plt.show()
	plt.clf()
	save(res, file_to_load)					

def get_entries_var_frac(result_dict,lambda_dict,list_of_methods,list_of_fractions):
	num_x_val=len(list_of_fractions)
	res_array=np.zeros((2,num_x_val))
	idx=0
	legend_array=[]
	for method in list_of_methods:
		legend_array.append(method)
		for f_idx in range(num_x_val): 
			# print result_dict[process][method].keys()
			# print '*********************************'
			res_array[idx,f_idx]=result_dict[method][str(lambda_dict[method][f_idx])][f_idx]
		idx+=1
	return res_array,legend_array

def plot_variation_fraction_new( file_prefix,file_lambda,flag_plot_all,measure,fig_save,file_save,lambda_dict,file_save_sel):

	list_of_fraction=[.5,.6,.7,.8,.9]
	# list_of_lambda=[.2,1.1]#[.7]#[.2,.4]#[0.9,2]#[.7,.08]
	# [.9,2]
	# [.2,.5,.7,.9,2]
	list_of_method=['cherrypick','Robust_cherrypick']
	
	
	res_lambda=load_data(file_lambda)['lambda']
	sl_lambda=res_lambda['hawkes']['slant'][2]
	cpk_lambda=res_lambda['hawkes']['cherrypick'][2]
	rcpk_lambda=res_lambda['hawkes']['Robust_cherrypick'][2]
	# rcpk_lambda=.4
	list_of_lambda=[cpk_lambda,rcpk_lambda]
	print 'lambdas', list_of_lambda
	# list_of_lambda[1]=.4
	lambda_dict={'cherrypick':[cpk_lambda]*6,'Robust_cherrypick':[rcpk_lambda]*6}
	if flag_plot_all:
		final_acc=get_file_name( file_prefix, 'slant' , sl_lambda, 0.2, 'hawkes', threshold=1 )[1]			
		res_dict={}
		fig,axes=plt.subplots()
		for method in list_of_method:
			res_dict[method]={}
			for l in list_of_lambda:
				list_acc=[ get_file_name(file_prefix,method,l,0.2,'hawkes',1,'f'+str(f))[1] for f in list_of_fraction]
				list_acc.append(final_acc)
				res_dict[method][str(l)]=list_acc
				if (method =='cherrypick' and l==cpk_lambda) or (method =='Robust_cherrypick' and l==rcpk_lambda):
					lw=6
					ls='--'
				else:
					lw=4
					ls='-'
				axes.plot( list_acc , label=str(l)+method,linewidth=lw,linestyle=ls )
		save(res_dict, file_save)
		axes.set_yscale('log')
		plt.title(file_prefix+' variation of fraction')
		plt.legend()
		plt.tight_layout()
		plt.xlabel('fraction')
		plt.ylabel('MSE')
		plt.savefig(fig_save)
		plt.show()
	else:	
		result_dict=load_data(file_save)
		list_of_fraction.append(1)
		res_array,legend_array=get_entries_var_frac(result_dict,lambda_dict,list_of_method,list_of_fraction)
		save({'x-axis':list_of_fraction,'y-axis':{'MSE':res_array},'labels':legend_array, 'lambda':lambda_dict},file_save_sel)

def plot_result_sanitize_test(  ) :
	# print "start"
	frac = '0.8'
	frac_tr = 0.9
	list_of_file=  ['barca','british_election','GTwitter','jaya_verdict', 'JuvTwitter' ,'MlargeTwitter','MsmallTwitter','real_vs_ju_703','trump_data','Twitter','VTwitter'] 
	file_index=int(sys.argv[2])
	method_index=int(sys.argv[1])
	list_of_lambda_both = np.array([[0.9,2,0.9,1.1,1.1,.7,2],[.9,.9,1.3,0.2,.7,1.1,.6]])
	# list_of_lambda_both = 0.9*np.ones((2,7))
	list_of_method = ['cherrypick','Robust_cherrypick']
	file_to_save='../result_sanitize_test/f0.8t0.2_MSE_FR'
	# result_dict={}
	# for method in list_of_method:
	# 	result_dict[method]={}
	# 	for file in list_of_file:
	# 		result_dict[method][file]={}
	# save(result_dict,file_to_save)
	# return
	result_dict=load_data(file_to_save)

	
	# result_dict={}
	for method, list_of_lambda_method in zip([list_of_method[method_index]], [list_of_lambda_both[method_index]] ):
		# result_dict[method]={}
		# print method
		# print method
		for file,lamb in zip([list_of_file[file_index] ], [list_of_lambda_method[file_index] ]) :
			# print file, lamb 
			# return 
			if lamb == 2 :
				lamb = int(2)
			# result_dict[method][file]={}
			file_to_read_full = '../result_subset_selection/'+ file +'w10f'+frac+'l'+str(lamb)+'.res.'+method+'.full'  
			flag=True
			if not os.path.isfile(file_to_read_full+'.pkl'):
				print "Not found : Full : "+file_to_read_full
				flag=False

			if method=='cherrypick':
				file_suffix = file +'w10v10f'+frac+'lc'+str(lamb)+'ls'+str(lamb)+'t0.2.res.'+method+'.slant'  
			if method=='Robust_cherrypick':
				file_suffix = file +'w10v10f'+frac+'ls'+str(lamb)+'t0.2.res.'+method+'.slant'
			
			# file_suffix = file+'w10v10f'+frac+'ls'+str(lamb)+'t0.2.res.'+method+'.slant'
			file_to_read_part = '../result_subset_selection_slant/'+ file_suffix
			
			if not os.path.isfile(file_to_read_part+'.pkl'):
				
				file_to_read_part='../result_performance_forecasting/'+file+'/'+file_suffix	
				if not os.path.isfile( file_to_read_part +'.pkl'):
					print "Not found : part : "+file_to_read_part
					flag=False
			if flag :
				result_full = load_data(file_to_read_full)
				result_part = load_data(file_to_read_part)
				n_data = result_full['data'].shape[0]
				n_tr = int( n_data*frac_tr)
				flag_on_test = result_full['data'][n_tr:]
				prediction_endo = np.mean(result_part['predicted'], axis = 1)[flag_on_test]
				target_endo = result_part['true_target'][flag_on_test]
				result_dict[method][file]['MSE_san']=get_MSE(prediction_endo,target_endo)
				result_dict[method][file]['MSE']= result_part['MSE']
				print file,method
				print result_dict[method][file]['MSE'],'\t\t', result_dict[method][file]['MSE_san']
				response=0
				while response==0:
					threshold=input('Enter threshold')
					FR_after_san = compute_FR_part(prediction_endo,target_endo,threshold)
					FR = compute_FR(result_part,threshold)
					print 'file, threshold, FR, FR after sanitization'
					print FR ,'\t\t', FR_after_san, '\n\n\n'
					response = input('1: save , 0: Recurse\n')
					# print 'you have entered ---',response
					# print type(response)
					if response==1:
						result_dict[method][file]['threshold']=threshold
						result_dict[method][file]['FR']=FR
						result_dict[method][file]['FR_san']=FR_after_san
						
					

			else:
				result_dict[method][file]['MSE']=0
				result_dict[method][file]['MSE_san']=0
				# return
	save( result_dict, file_to_save) 

def plot_variation_fraction_combined( file,file_to_load_cpk,file_to_load_rcpk, file_to_save,file_to_plot,measure,skip_ytick):
	
	res_dict=load_data(file_to_save,'ifexists')
	if not res_dict:
		res_dict['cherrypick']=load_data(file_to_load_cpk)
		res_dict['Robust_cherrypick']=load_data(file_to_load_rcpk)
	##################################################################################
	# print res_dict['cherrypick']['ytick_val']

	# print res_dict['Robust_cherrypick']['ytick_val']
	
	# print res_dict['cherrypick']['threshold']

	# print res_dict['Robust_cherrypick']['threshold']
	########################################
	# fig, ax=plt.subplots()
	# ax.plot(x,y)
	# ax.set_yscale("log")
	#####################################################
	# sub_ticks = [.001*i for i in range(25,56,5)] # fill these midpoints
	# format = "%.3f" # standard float string formatting
	# ax.set_ylim([.025,.055])
	# ax=format_ticks(ax, sub_ticks)
	# plt.show()
	####################################################################################
	fig,ax=plt.subplots()
	for method,ls in zip(['cherrypick','Robust_cherrypick'],['-.','--']):
		line=ax.plot(ma(res_dict[method]['y-axis'][measure][1:],3),label=method)
		print ma(res_dict[method]['y-axis'][measure][1:],3)
		# line=plt.plot(res_dict['y-axis'][measure][1:])
		plt.setp(line, linewidth=4,linestyle=ls,marker='o', markersize=10)
	ax.set_yscale('log')
	plt.xticks(range(5),res_dict['cherrypick']['x-axis'][1:],rotation=0,fontsize=20.7,weight='bold')
	plt.grid()
	plt.xlabel('Fraction of endogenous messages',rotation=0,fontsize=20.7,weight='bold')
	plt.ylabel(measure,rotation=90,fontsize=20.7,weight='bold')
	if not skip_ytick:
		ytick_val =[.01*i for i in [10,15,20,25,30]]
		# ytick_val =[.01*i for i in [8,10,12,14,16]]
		# ytick_val =[.01*i for i in [16,20,24,28]]
		# ytick_val =[.01*i for i in [14,16,18,20]]
		# ytick_val =[.01*i for i in [8,9,10,12,14]]
		# ytick_val =[.001*i for i in [22,24,26,28,30]]
		# ytick_val =[.001*i for i in [40,60,80,100,150]]
		# ytick_val =[.001*i for i in [68,70,72,74,76]]
		# ytick_val =[.001*i for i in [48,50,52,54,58]]# [.001*i for i in range(35,45,2)] # fill these midpoints
		# ytick_val =[.001*i for i in [35,38,40,42,45]]# [.001*i for i in range(35,45,2)] # fill these midpoints
		format = "%.3f" # standard float string formatting
		ax.set_ylim([.10,.50])
		ax=format_ticks(ax, ytick_val)
		if 'ytick_val' not in res_dict:
			res_dict['ytick_val']={}
		res_dict['ytick_val'][measure]=ytick_val
	plt.legend()
	plt.title(file,rotation=0,fontsize=20.7,weight='bold')
	plt.tight_layout()
	plt.savefig(file_to_plot+'.jpg')
	plt.show()
	plt.clf()
	save(res_dict, file_to_save)

def plot_final_results_FR(file_prefix,file_to_load,file_to_plot,measure):

	# dict_res={'x-axis':list_of_time_span, 'y-axis':{'MSE':list_res_MSE,'FR':list_res_FR}, 'threshold':threshold, 'labels':labels}
	res=load_data(file_to_load)
	x=res['x-axis']
	y=res['y-axis'][measure] # remove if you use old version
	labels=res['labels']
	seq=[0,1,2,4,5,3]
	marker_seq=['P','o','*','o','*','P']
	ls_seq=[':','-.','--','-.','--',':']
	f,ax=plt.subplots()

	for ind,mk,ls in zip(seq,marker_seq,ls_seq):
		line=plt.plot(ma(y[ind],3), label=labels[ind])
		plt.setp(line, linewidth=4,linestyle=ls,marker=mk, markersize=10)
	# ax.set_yscale('log')
	plt.xticks(np.arange(6),x,rotation=0,fontsize=22.7,weight='bold')
	plt.grid()
	plt.legend()

	# plt.yticks(res['ytick_val'],rotation=0,fontsize=15.7,weight='bold')
	# plt.ytick.remove()
	# ax.set_yticks(ytick_val,rotation=0,fontsize=15.7,weight='bold')
	# ax.tick_params(labelleft=False,labelright=False)    
	# labels = [item.get_text() for item in ax.get_yticklabels()]
	# empty_string_labels = ['']*len(labels)
	# ax.set_yticklabels(empty_string_labels)
	ytick_val=[.01*i for i in range(10,16,1)]#np.arange(.06,.1,.01)
	
	# ytick_val=res['ytick_val']
	# res['ytick_val']=ytick_val
	plt.yticks(ytick_val,rotation=0,fontsize=20.7,weight='bold')
	plt.xlabel('Time span',rotation=0,fontsize=20.7,weight='bold')
	# plt.ylabel('FR', fontsize=26.7,weight='bold')
	plt.title(file_prefix,rotation=0,fontsize=20.7,weight='bold')
	plt.tight_layout()
	plt.savefig(file_to_plot)
	# plt.savefig(file_to_plot)
	plt.show()
	plt.clf()
	save(res, file_to_load)

def plot_result_forecasting_FR(file_prefix, file_to_save, flag_save,threshold, list_of_process, list_of_methods,list_of_time_span, hawkes_lambda_file, set_of_lambda=None,selected_lambda=None):
	hawkes_lambda=read_lambda(load_data(hawkes_lambda_file),'hawkes',list_of_methods, list_of_time_span)
	# poisson_lambda=read_lambda(load_data(poisson_lambda_file),'poisson',list_of_methods, list_of_time_span)
	dict_lamb={'hawkes':hawkes_lambda}#,'poisson':poisson_lambda}
	labels=[]
	list_res_MSE=[]
	list_res_FR=[]
	for process in list_of_process:
		for method in list_of_methods:
			if process=='hawkes':
				labels.append(process+'_'+method)
				lamb=[dict_lamb[process][method][time_span] for time_span in list_of_time_span]
				list_FR,list_MSE=compute_list_MSE_FR(file_prefix, process, method, lamb, list_of_time_span,threshold)
				list_res_FR.append(list_FR)
				list_res_MSE.append(list_MSE)
			if process=='poisson':
				# print 'keys',dict_lamb['hawkes'][method].keys()
				lamb_t0=dict_lamb['hawkes'][method][0.0]
				if not flag_save:
					for lamb in set_of_lambda:
						# print process+'_'+method+'_l:'+str(lamb)
						labels.append(process+'_'+method+'_l:'+str(lamb))
						list_FR,list_MSE=compute_list_MSE_FR(file_prefix, process, method, [lamb], list_of_time_span,threshold, lamb_t0)
						list_res_FR.append(list_FR)
						list_res_MSE.append(list_MSE)
				else:
					lamb=selected_lambda[process+'_'+method]
					labels.append(process+'_'+method)
					list_FR,list_MSE=compute_list_MSE_FR(file_prefix, process, method, lamb, list_of_time_span,threshold, lamb_t0)
					list_res_FR.append(list_FR)
					list_res_MSE.append(list_MSE)
	# print list_res_FR
	if flag_save:
		dict_res={'x-axis':list_of_time_span, 'y-axis':{'MSE':list_res_MSE,'FR':list_res_FR}, 'threshold':threshold, 'labels':labels, 'poisson_lambda':selected_lambda}
		save(dict_res,file_to_save)
		plot_matrix(list_res_MSE,labels,file_prefix+'_mse_modified.jpg')
		plot_matrix(list_res_FR,labels,file_prefix+'_FR_modified.jpg')
	else:
		# print len(list_res_MSE)
		# print len(labels)
		plot_matrix_all_result(file_prefix,[list_res_MSE,list_res_FR],labels,len(set_of_lambda), file_prefix+'_all_modified.jpg')	

def compute_list_MSE_FR(file_prefix, process, method, lamb, list_of_time_span, threshold,lamb_t0=None):
	# print 'prev',lamb
	if len(lamb)==1:
		lamb=lamb*len(list_of_time_span)
	# print 'later',lamb
	list_res_each_plot_FR=[]
	list_res_each_plot_MSE=[]
	for time_span,l in zip(list_of_time_span,lamb):
		if time_span==0.0 and process == 'poisson':
			res_FR,res_MSE=get_file_name(file_prefix,method,lamb_t0,time_span,'hawkes',threshold)
		else:
			res_FR,res_MSE=get_file_name(file_prefix,method,l,time_span,process,threshold)
			
		list_res_each_plot_MSE.append(res_MSE)
		list_res_each_plot_FR.append(res_FR)
	return list_res_each_plot_FR,list_res_each_plot_MSE

def compute_list_MSE_FR_asymmetric_slant(file_prefix, process, method, lamb, list_of_time_span, threshold,lamb_t0=None):

	if len(lamb)==1:
		lamb=lamb*len(list_of_time_span)
	list_res_each_plot_FR=[]
	list_res_each_plot_MSE=[]
	for time_span,l in zip(list_of_time_span,lamb):
		res_FR,res_MSE=get_file_name(file_prefix,method,l,time_span,process,threshold)			
		list_res_each_plot_MSE.append(res_MSE)
		list_res_each_plot_FR.append(res_FR)
	return list_res_each_plot_FR,list_res_each_plot_MSE
	
def get_entries(result_dict,lambda_dict,list_of_process,list_of_methods,list_of_time_span,num_x_val):
	res_array=np.zeros((6,num_x_val) ) # def num_x_val
	idx=0
	legend_array=[]
	for process in list_of_process:
		for method in list_of_methods:
			legend_array.append(process+'_'+method)
			for t,t_idx in zip(list_of_time_span, range(num_x_val)): 
				print '*********************************'
				print result_dict[process][method].keys()
				print '*********************************'
				res_array[idx,t_idx]=result_dict[process][method][str(lambda_dict[process][method][t_idx])][str(t)]
			idx+=1
	return res_array,legend_array

def plot_result_forecasting_MSE_asymmetric_slant(file_prefix, file_to_save, list_of_process, list_of_methods,list_of_time_span, set_of_lambda, plot_to_save, lambda_dict=None, file_to_save_selected=None,flag_plot_all=None):
	if flag_plot_all:
		labels=[]
		list_res_MSE=[]
		# list_res_FR=[]
		result_dict={'hawkes':{},'poisson':{}}
		for process in list_of_process:
			for method in list_of_methods:
				if method not in result_dict[process]:
					result_dict[process][method]={}
				for lamb in set_of_lambda:
					labels.append(process+'_'+method+'_'+str(lamb))
					list_FR,list_MSE=compute_list_MSE_FR_asymmetric_slant(file_prefix, process, method, [lamb], list_of_time_span,threshold=1)
					list_res_MSE.append(list_MSE)
					result_dict[process][method][str(lamb)]={}
					for time,elm in zip(list_of_time_span,list_MSE):
						result_dict[process][method][str(lamb)][str(time)]= elm
		save(result_dict,file_to_save)
		file_title=file_prefix+sys.argv[7]
		plot_matrix_all_result_asymmetric_slant(file_title,list_res_MSE,labels,len(set_of_lambda),plot_to_save)			
	

	else:
		result_dict=load_data(file_to_save)
		res_array,legend_array=get_entries(result_dict,lambda_dict,list_of_process,list_of_methods,list_of_time_span,6)
		save({'x-axis':list_of_time_span,'y-axis':{'MSE':res_array},'labels':legend_array, 'lambda':lambda_dict},file_to_save_selected)
		
def plot_matrix_all_result_asymmetric_slant(file_title, data, labels, n_lamb, name):
	fig,axes=plt.subplots()
	ls_seq=[ i  for i in [':','-.','--'] for n in range(n_lamb)]*2
	lw_seq=[ i  for i in range(2,5,1) for n in range(n_lamb)]*2
	# print len(ls_seq)
	# print len(lw_seq)
	plt.title(file_title)

	
	output_plot_p=sys.argv[5]

	output_plot_h=sys.argv[6]
	
	for row,l,lw,ls in zip(data,labels,lw_seq,ls_seq):
		if 'poisson' in l:
			if 'c' in output_plot_p and 'cherrypick' in l and 'Ro' not in l  :
				axes.plot(row,label=l,linewidth=lw,linestyle=ls,marker='o', markersize=6)
			if 'r' in output_plot_p  and 'Ro' in l : 
				axes.plot(row,label=l,linewidth=lw,linestyle=ls,marker='o', markersize=6)
			if 's' in output_plot_p  and 'sl' in l : 
				axes.plot(row,label=l,linewidth=lw,linestyle=ls,marker='o', markersize=6)
		if 'hawkes' in l : 
			if 'c' in output_plot_h and 'cherrypick' in l and 'Ro' not in l  :
				axes.plot(row,label=l,linewidth=lw,linestyle=ls,marker='^', markersize=4)
			if 'r' in output_plot_h  and 'Ro' in l  : 
				axes.plot(row,label=l,linewidth=lw,linestyle=ls,marker='^', markersize=4)
			if 's' in output_plot_h  and 'sl' in l  : 
				axes.plot(row,label=l,linewidth=lw,linestyle=ls,marker='^', markersize=4)
		
	axes.set_yscale('log')
	# plt.yticks([.06,.07,.8,1])
	plt.tight_layout()
	
	plt.legend()
	plt.savefig(name[:-4]+sys.argv[7]+'.jpg')
	plt.show()  
	plt.clf()

def plot_final_result_performance( file_prefix, file_to_load,file_to_plot,measure,skip_ytick,specify_ytick,threshold=None):
	
	res=load_data(file_to_load)
	x=res['x-axis']
	y=res['y-axis'][measure]
	y[3:,0]=y[:3,0]
	labels=res['labels']
	marker_seq=['P','o','*','P','o','*']
	ls_seq=[':','-.','--',':','-.','--']

	f,ax=plt.subplots()
	for data,l,mk,ls in zip(y,labels,marker_seq,ls_seq):
		line=plt.plot(ma(data,3), label=l)
		# if 'hawkes' in l : 
		# 	markersize=5
		# if 'poisson' in l: 
		# 	markersize=1
		plt.setp(line, linewidth=4,linestyle=ls,marker=mk, markersize=10)

	ax.set_yscale('log')
	# plt.ylim([.08,.09])
	plt.xticks(range(x.shape[0]),x,rotation=0,fontsize=20.7,weight='bold')
	if not skip_ytick:
		if specify_ytick:
			ytick_val =[.01*i for i in [9,10,12,14,16]]
			# ytick_val =[.1*i for i in [2,2.1,2.2,2.3,2.4,2.5]]
			res['ytick_val'][measure]=ytick_val
			save(res,file_to_load)
		else:
			ytick_val =res['ytick_val'][measure]
		format = "%.2f"
		# ax.set_ylim([.10,.50])
		ax=format_ticks(ax, ytick_val,format)
		# if 'ytick_val' not in res:
		# 	res['ytick_val']={}
		
		# if measure=='FR':
		# 	pass # threshold
		

	plt.xlabel('Time span',rotation=0,fontsize=20.7,weight='bold')
	plt.ylabel(measure,rotation=90,fontsize=20.7,weight='bold')
	plt.grid()
	plt.legend()
	# if measure=='FR':
	# 	plt.title(file_prefix+'_threshold_'+str(threshold),rotation=0,fontsize=20.7,weight='bold')
	# else:
	plt.title(file_prefix,rotation=0,fontsize=20.7,weight='bold')
	plt.tight_layout()
	plt.savefig(file_to_plot)
	plt.show()
	plt.clf()

def plot_result_forecasting_FR_asymmetric_slant(file_prefix):
	
	lambda_file='../result_performance_forecasting_pkl/'+file_prefix+'.selected'
	lambda_dict=load_data(lambda_file,'ifexists')['lambda']

	list_of_process=['hawkes','poisson']
	list_of_method=['slant','cherrypick','Robust_cherrypick']
	list_of_time_span=np.array([0.0,.1,.2,.3,.4,.5])

	threshold=float(sys.argv[2])
	w_v_string=sys.argv[3]

	result_list=[]
	for process in list_of_process:
		for method in list_of_method:
			set_of_lambda=lambda_dict[process][method]
			res=[ get_file_name(file_prefix,method,lamb,time,process,threshold,'f0.8',w_v_string)[0] for lamb,time in zip(set_of_lambda,list_of_time_span)]	
			result_list.append(res)
	
	file_save='../result_performance_forecasting_pkl/'+file_prefix+'.selected'
	res=load_data(file_save)
	res['y-axis']['FR']=np.array(result_list)
	res['threshold']=threshold
	save(res, file_save)
	
	skip_ytick=int(sys.argv[4])
	specify_ytick=int(sys.argv[5])
	file_plot='../Plots/Plots_forecasting/'+file_prefix+'_FR.jpg'	
	plot_final_result_performance( file_prefix, file_save ,file_plot,'FR',skip_ytick,specify_ytick,threshold)

def get_file_name(file_prefix,method,lamb,time_span,keyword,threshold,frac=None,w_v_string=None):
	# if lamb==1.0 or lamb==2.0:
	# 	lamb=int(lamb)
	if not frac:
		frac='f0.8'
	if not w_v_string:
		w_v_string=sys.argv[7]
	for f in [0.5,0.6,0.7,0.9]:
		# print lamb
		if str(f) in frac and ('barca' not in file_prefix and 'real' not in file_prefix):
			# print 'doing'
			# print lamb
			lamb=float(lamb)
			# print lamb
			# print '*******************'

	file_read_prefix='../result_subset_selection_slant/'+file_prefix+w_v_string
	if method=='slant':
		file_read=file_read_prefix+'l'+str(lamb)+'t'+str(time_span)+'.res.slant'
	if method=='cherrypick':
		file_read=file_read_prefix+frac+'lc'+str(lamb)+'ls'+str(lamb)+'t'+str(time_span)+'.res.cherrypick.slant'
	if method=='Robust_cherrypick':
		file_read=file_read_prefix+frac+'ls'+str(lamb)+'t'+str(time_span)+'.res.Robust_cherrypick.slant'
	if keyword=='poisson':
		file_read = file_read+'.Poisson'	

	#--------------------------------------------------------------------
	#-----------------------FOR OLD SETTING------------------------------
	#--------------------------------------------------------------------
	# if keyword=='hawkes':
	# 	file_read_prefix='../result_performance_forecasting/'+file_prefix+'/' + file_prefix 
	# 	if method=='slant':
	# 		file_read=file_read_prefix+'w10v10l'+str(lamb)+'t'+str(time_span)+'.res.slant'
	# 	if method=='cherrypick':
	# 		file_read=file_read_prefix+'w10v10f0.8lc'+str(lamb)+'ls'+str(lamb)+'t'+str(time_span)+'.res.cherrypick.slant'
	# 	if method=='Robust_cherrypick':
	# 		file_read=file_read_prefix+'w10v10f0.8ls'+str(lamb)+'t'+str(time_span)+'.res.Robust_cherrypick.slant'
	# if keyword == 'poisson':
	# 	file_read_prefix = '../result_forecasting_Poisson/'+file_prefix+'w10v10'
	# 	if method =='Robust_cherrypick':
	# 		file_read = file_read_prefix+'f0.8ls'+str(lamb)+'t'+str(time_span)+'.res.'+method+'.slant.Poisson'
	# 	if method =='cherrypick':#,'Robust_cherrypick']:
	# 		file_read = file_read_prefix+'f0.8lc'+str(lamb)+'ls'+str(lamb)+'t'+str(time_span)+'.res.'+method+'.slant.Poisson'
	# 	if method=='slant':
	# 		file_read = file_read_prefix+'l'+str(lamb)+'t'+str(time_span)+'.res.slant.Poisson'
	res=load_data(file_read,'ifexists')
	if not res:
		print 'Not found: ', file_read
		return 0,0
	res_MSE=res['MSE']
	res_FR=compute_FR( res, threshold) 
	return res_FR,res_MSE

def format_ticks(ax, sub_ticks,format):
	# format = "%.2f" 
	ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
	ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter(format))
	ax.yaxis.set_minor_formatter(matplotlib.ticker.ScalarFormatter())
	ax.yaxis.set_minor_formatter(matplotlib.ticker.FormatStrFormatter(format))
	
	
	ax.set_yticks(sub_ticks)
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(20)
		tick.label.set_fontweight('bold')
	for tick in ax.yaxis.get_minor_ticks():
		tick.label.set_fontsize(20)
		tick.label.set_fontweight('bold')
	return ax 

def plot_variation_fraction_after_asymm_slant( file_prefix,file_to_save,file_to_plot,measure,skip_ytick,specify_ytick):
	
	res_dict=load_data(file_to_save,'ifexists')
	# print '*****************************'
	# print res_dict.keys()
	# print '*****************************'
	num_x_val=len(res_dict['x-axis'])
	fig,ax=plt.subplots()
	for data,method,ls in zip(res_dict['y-axis'][measure],res_dict['labels'],['-.','--','-.','--']):
		if ('Robust'  in method):# and 'poisson' in method) :
			data= ma(data,3)
		# if 'Ro' not in method:
		line=ax.plot( data,label=method)
		# print ma(res_dict[method]['y-axis'][measure][1:],3)
		# line=plt.plot(res_dict['y-axis'][measure][1:])
		plt.setp(line, linewidth=4,linestyle=ls,marker='o', markersize=10)
	ax.set_yscale('log', nonposy="clip")
	plt.xticks(range(num_x_val),res_dict['x-axis'],rotation=0,fontsize=20.7,weight='bold')
	plt.grid()
	plt.xlabel('Fraction of endogenous messages',rotation=0,fontsize=20.7,weight='bold')
	plt.ylabel(measure,rotation=90,fontsize=20.7,weight='bold')
	if not skip_ytick:
		# ytick_val =[.001*i for i in [50,55,60,65,70]] # british
		# ytick_val =[.001*i for i in [35,40,45,50,55]] # barca 
		if specify_ytick:
			# pass
			# tmp=load_data('../result_variation_fraction_pkl_backup_6feb2019/'+file_prefix+'.selected')
			# ytick_val=tmp['ytick_val'][measure]
			# ytick_val.append(.1)
			ytick_val =[.1*i for i in [.8,1,1.2,1.5,2]] # Juv
			# print ytick_val
			if 'ytick_val' not in res_dict:
				res_dict['ytick_val']={}
			res_dict['ytick_val'][measure]=ytick_val
		else:
			ytick_val =res_dict['ytick_val'][measure]

		format = "%.2f" 
		# ax.set_ylim([.10,.50])
		ax=format_ticks(ax, ytick_val,format)
		plt.minorticks_off()

		
	plt.legend()
	plt.title(file_prefix,rotation=0,fontsize=20.7,weight='bold')
	plt.tight_layout()
	plt.savefig(file_to_plot)
	plt.show()
	plt.clf()
	save(res_dict, file_to_save)

def plot_variation_fraction_new_FR( file_prefix):

	file_lambda='../result_performance_forecasting_pkl/'+file_prefix+'.selected'
	res_lambda=load_data(file_lambda)['lambda']

	threshold=float(sys.argv[3])
	threshold_file='../result_performance_forecasting_pkl/'+file_prefix+'.selected'
	threshold=load_data(threshold_file,'ifexists')['threshold']
	# print threshold
	# return
	w_v_string=sys.argv[4]

	res_list=[]
	list_of_method=['cherrypick','Robust_cherrypick']
	list_of_process=['hawkes','poisson']
	list_of_fraction=[.5,.6,.7,.8,.9]
	for process in list_of_process:
		sl_lambda=res_lambda[process]['slant'][2]
		final_acc=get_file_name( file_prefix,'slant',sl_lambda, 0.2,process,threshold,None,w_v_string)[0]				
		for method in list_of_method:	
			l=res_lambda[process][method][2]
			acc_list=[ get_file_name(file_prefix,method,l,0.2,process,threshold,'f'+str(f),w_v_string)[0] for f in list_of_fraction]
			acc_list.append(final_acc)
			res_list.append(acc_list)
			
	
	file_save='../result_variation_fraction_pkl/'+file_prefix+'.selected'
	res=load_data(file_save)
	res['y-axis']['FR']=np.array(res_list)
	save(res,file_save)

	skip_ytick=int(sys.argv[5])
	specify_ytick=int(sys.argv[6])
	file_plot='../Plots/Plots_variation_fraction/'+file_prefix+'_combined_FR.jpg'
	plot_variation_fraction_after_asymm_slant( file_prefix,file_save,file_plot,'FR',skip_ytick,specify_ytick)

def plot_variation_fraction_new_MSE( file_prefix):

	
	file_lambda='../result_performance_forecasting_pkl/'+file_prefix+'.selected'
	res_lambda=load_data(file_lambda)['lambda']

	threshold=float(sys.argv[3])
	w_v_string=sys.argv[4]

	res_list=[]
	labels=[]
	list_of_method=['cherrypick','Robust_cherrypick']
	list_of_process=['hawkes','poisson']
	list_of_fraction=[.5,.6,.7,.8,.9]
	for process in list_of_process:
		sl_lambda=res_lambda[process]['slant'][2]
		final_acc=get_file_name( file_prefix,'slant',sl_lambda, 0.2,process,threshold,None,w_v_string)[1]				
		for method in list_of_method:	
			l=res_lambda[process][method][2]
			acc_list=[ get_file_name(file_prefix,method,l,0.2,process,threshold,'f'+str(f),w_v_string)[1] for f in list_of_fraction]
			acc_list.append(final_acc)
			res_list.append(acc_list)
			labels.append(process+'_'+method)
			# print process, method, l
	list_of_fraction.append(1)
	# print np.array(res_list).shape
	file_save='../result_variation_fraction_pkl/'+file_prefix+'.selected'
	save({'x-axis':list_of_fraction,'y-axis':{'MSE':np.array(res_list)},'labels':labels},file_save)

	skip_ytick=int(sys.argv[5])
	specify_ytick=int(sys.argv[6])
	file_plot='../Plots/Plots_variation_fraction/'+file_prefix+'_combined_MSE.jpg'
	plot_variation_fraction_after_asymm_slant( file_prefix,file_save,file_plot,'MSE',skip_ytick,specify_ytick)

def plot_forecasting_prediction( set_of_files, set_of_labels, plot_title, plot_to_save):
	for file,l in zip(set_of_files, set_of_labels):
		# print file, l
		res=load_data(file,'ifexists')
		if res:
			plt.plot(res['predicted'],label=str(l))	
	plt.plot(res['true_target'],label='true')
	plt.legend()
	plt.title(plot_title)
	plt.tight_layout()
	plt.savefig(plot_to_save)
	plt.show()
	
def get_acc_binned(prediction,target):
		# def map_class(v):
		# 	return ((v+1)/2)*21+1
		# pred_cls = map(map_class, prediction)
		# true_cls = map(map_class, self.test[:,2])
		corr_pred=0
		for p,t in zip(prediction,target):
			if np.absolute(t-p)< .6 :
				corr_pred+=1
		return 1-float(corr_pred)/prediction.shape[0]

def plot_result_sanitize_test_v2_non_opinion():
	
	frac_tr = 0.9

	
	file_subset_full = '../general_form/result_subset_selection/so_tr_binned_fullwin100f0.95l0.0001.res.cherrypick' #*
	subset_full = load_data(file_subset_full)

	result = load_data('../general_form/result_subset_selection_slant/so_tr_binnedwin100f0.95lc0.0001ls0.0001t0.0.res.cherrypick.slant.Poisson' )

	n_data = subset_full['data'].shape[0]
	n_tr = int( n_data*frac_tr)
	n_tr_actual=result['true_target'].shape[0]

	print n_data, n_tr, n_tr_actual

	flag_on_test = subset_full['data'][n_tr:]
	prediction_endo = result['predicted'][flag_on_test]
	target_endo = result['true_target'][flag_on_test]

	print 'san',get_acc_binned(prediction_endo,target_endo)
	print result['FR']

	# print 'MSE',result_dict[method][file_prefix]['MSE'],'\t\tMSEsan', result_dict[method][file_prefix]['MSE_san']

def plot_result_sanitize_test_v2():
	
	frac_tr = 0.9

	list_of_file=['barca','british_election','GTwitter','jaya_verdict', 'JuvTwitter' ,'MlargeTwitter','MsmallTwitter','real_vs_ju_703','trump_data','Twitter','VTwitter']
	file_prefix=list_of_file[int(sys.argv[1])]

	list_of_method = ['cherrypick','Robust_cherrypick']
	method=list_of_method[int(sys.argv[2])]

	print file_prefix,method

	w_v_str=sys.argv[3]
	w_str=w_v_str.split('v')[0]

	# file_to_save='../result_sanitize_test/f0.8t0.2_MSE_FR'
	# result_dict=load_data(file_to_save)
	# result_dict={}
	# for method in list_of_method:
	# 	result_dict[method]={}
	# 	for file in list_of_file:
	# 		result_dict[method][file]={}
	# save(result_dict,file_to_save)
	# return	


	file_lambda='../result_performance_forecasting_pkl/'+file_prefix+'.selected'
	lamb=0.4#load_data(file_lambda)['lambda']['hawkes'][method][2]
	
	file_subset_full = '../result_subset_selection_full/'+ file_prefix+w_str+'f0.8l'+str(float(lamb))+'.res.'+method
	subset_full = load_data(file_subset_full)

	lamb=load_data(file_lambda)['lambda']['hawkes'][method][0]
	if method=='cherrypick':
		file_suffix = file_prefix +w_v_str+'f0.8lc'+str(lamb)+'ls'+str(lamb)+'t0.0.res.'+method+'.slant'  
	if method=='Robust_cherrypick':
		file_suffix = file_prefix +w_v_str+'f0.8ls'+str(lamb)+'t0.0.res.'+method+'.slant'
	result = load_data('../result_subset_selection_slant/'+ file_suffix)

	n_data = subset_full['data'].shape[0]
	n_tr = int( n_data*frac_tr)
	n_tr_actual=result['true_target'].shape[0]

	print n_data, n_tr, n_tr_actual

	flag_on_test = subset_full['data'][n_tr:]
	prediction_endo = np.mean(result['predicted'], axis = 1)[flag_on_test]
	target_endo = result['true_target'][flag_on_test]

	def impr(a,b):
		return float(a-b)/a
	print 'full',result['MSE']
	print 'san',get_MSE(prediction_endo,target_endo)
	print impr(result['MSE'],get_MSE(prediction_endo,target_endo))
	
	return 

	result_dict[method][file_prefix]['MSE_san']=get_MSE(prediction_endo,target_endo)
	result_dict[method][file_prefix]['MSE']= result['MSE']

	print 'MSE',result_dict[method][file_prefix]['MSE'],'\t\tMSEsan', result_dict[method][file_prefix]['MSE_san']


	return 


	threshold_file='../result_performance_forecasting_pkl/'+file_prefix+'.selected'
	threshold=load_data(threshold_file,'ifexists')['threshold']
	result_dict[method][file_prefix]['FR_san'] = compute_FR_part(prediction_endo,target_endo,threshold)
	result_dict[method][file_prefix]['FR'] = compute_FR(result,threshold)

	print 'FR',result_dict[method][file_prefix]['FR'],'\t\tFRsan', result_dict[method][file_prefix]['FR_san']
	
	data=[result_dict[method][file_prefix]['MSE'],result_dict[method][file_prefix]['MSE_san'],result_dict[method][file_prefix]['FR'],result_dict[method][file_prefix]['FR_san'],]
	plt.plot(data)
	plt.show()
	# response=0
	# while response==0:
	# 	threshold=input('Enter threshold')
	# 	FR_after_san = compute_FR_part(prediction_endo,target_endo,threshold)
	# 	FR = compute_FR(result,threshold)
	# 	print 'file, threshold, FR, FR after sanitization'
	# 	print FR ,'\t\t', FR_after_san, '\n\n\n'
	# 	response = input('1: save , 0: Recurse\n')
	# 	# print 'you have entered ---',response
	# 	# print type(response)
	# 	if response==1:
	# 		result_dict[method][file]['threshold']=threshold
	# 		result_dict[method][file]['FR']=FR
	# 		result_dict[method][file]['FR_san']=FR_after_san
		
	save( result_dict, file_to_save) 

def impr(a,b):
		return float(a-b)/a

def main():

	list_of_windows = np.array([.4]) # np.linspace(0.05,1,20)
	time_span_input_list = np.linspace(0,.5,6) 
	file_prefix_list = ['barca','british_election','GTwitter','jaya_verdict', 'JuvTwitter' ,'MlargeTwitter','MsmallTwitter','real_vs_ju_703','trump_data','Twitter','VTwitter','food','reddit']
	save_result_variation_fraction= False # True
	plot_forecasting_performance_wt_poisson=False # True
	save_result_forecasting_poisson=False # True 
	plot_result_forecasting_FR_flag=False # True
	plot_forecasting_all_new_flag=False # True
	plot_forecasting_after_asymmetric_slant=False # True
	plot_result_variation_fraction=False # True # True
	plot_result_sanitize_test_flag=False # 
	plot_opinions=False # True


	
	if plot_opinions:

		file_prefix=file_prefix_list[11]
		set_of_file=[]
		set_of_labels=[]
		set_of_w=[10] # [1000,500,100,50,10,1]
		set_of_v=[10]  # [1000,500,100,50,10,1]
		set_of_lamb=[1000,100,10,5,2,1,.5,.1,.05,.01,.001,.0001]
		for w in set_of_w:
			for v in set_of_v:
				for lamb in set_of_lamb:
					file_name = '../general_form/result_subset_selection_slant/'+file_prefix+'w'+str(w)+'v'+str(v)+'l'+str(l)+'t0.res.slant.pkl'
					set_of_files.append(file_name)
					set_of_labels.append('w-'+str(w)+',v-'+str(v)+',lamb='+str(lamb))
		plot_title='slant'
		plot_to_save='../Plots/Plots_nowcasting/slant.jpg'
		plot_forecasting_prediction( set_of_files, set_of_labels, plot_title, plot_to_save)

	if plot_result_sanitize_test_flag:
		plot_result_sanitize_test_v2()
		# plot_result_sanitize_test_v2_	non_opinion()

	if plot_result_variation_fraction:
		file_prefix_list=['barca','british_election','GTwitter','jaya_verdict', 'JuvTwitter' ,'MlargeTwitter','MsmallTwitter','real_vs_ju_703','trump_data','Twitter','VTwitter']
		file_prefix=file_prefix_list[int(sys.argv[1])]
		measure=list(['MSE','FR'])[int(sys.argv[2])]
		if measure == 'MSE':
			plot_variation_fraction_new_MSE( file_prefix)
		else:
			plot_variation_fraction_new_FR( file_prefix)
		return

		###############
		file_lambda='../result_performance_forecasting_pkl/'+file_prefix+'.selected'
		fig_save='../Plots/Plots_variation_fraction/'+file_prefix+'.all.MSE.jpg'
		file_plot='../Plots/Plots_variation_fraction/'+file_prefix+'_combined_'+measure+'.jpg'
		file_save='../result_variation_fraction_pkl/'+file_prefix+'_t0.2_combined'
		lambda_dict={'cherrypick':[],'Robust_cherrypick':[]}
		file_save_sel='../result_variation_fraction_pkl/'+file_prefix+'.selected'
		################
		plot_variation_fraction_new( file_prefix,file_lambda,flag_plot_all,measure,fig_save,file_save,lambda_dict,file_save_sel)
		if not flag_plot_all:
			plot_variation_fraction_after_asymm_slant( file_prefix,file_save_sel,file_plot,measure,skip_ytick)

	if plot_forecasting_after_asymmetric_slant:
		file_prefix_list=['barca','british_election','GTwitter','jaya_verdict', 'JuvTwitter' ,'MlargeTwitter','MsmallTwitter','real_vs_ju_703','trump_data','Twitter','VTwitter']
		file_prefix=file_prefix_list[int(sys.argv[1])]
		plot_result_forecasting_FR_asymmetric_slant(file_prefix)
		return 
		##################   Pring all lambdas            #######################
		# for file_prefix in file_prefix_list:
		# 	if 'trump' not in file_prefix and 'Mlarge' not in file_prefix:
		# 		lamb=load_data('../result_performance_forecasting_pkl/'+file_prefix+'.selected','ifexists')['lambda']
		# 		# print lamb.keys()
		# 		print file_prefix,lamb['poisson']['cherrypick'][2],lamb['poisson']['Robust_cherrypick'][2]
		# return
		##########################################
		flag_plot_all=int(sys.argv[2])
		measure=['MSE','FR'][int(sys.argv[3])]
		skip_ytick=int(sys.argv[4])
		##############################
		# list_of_lambda =[.2,.5,.7,.9,2]#[.1,.5,1]#
		# list_of_lambda =[.1,.2,.4,.5,.6,.9,1.0]
		# list_of_lambda.extend([0.08,0.1,0.5,1.0,1.3,1.5])
		# list_of_lambda.extend([0.08,0.1,0.5,1.0,1.3,1.5])# GTwitter
		# list_of_lambda =[.1,.5,1.0]#
		# list_of_lambda=[.2,.5,.7,.9,2]
		# list_of_lambda.extend([.6,.9,1.1,1.5,1.6,1.8,2.2,2.5])
		# list_of_lambda=[.4,]
		# list_of_lambda.extend([.1,.5,1.0,1.5,2.5])
		# list_of_lambda.extend([.4,.6])# jaya
		# list_of_lambda.extend([.08,.09,.1,1.5,2.5])# Juv
		# list_of_lambda.extend([.05,.1,.15,1.1,1.5])	 # Msmall
		# list_of_lambda.extend([2.2,2.5,3,0.05,1.5])# real
		# list_of_lambda.extend([.05,.1,.15,1.1,1.5,2.2,2.5,3])# real
		# list_of_lambda=[1.0,1.5,.4,.9,.2,.5,1.8,1.3,.55]#GTwitter
		# list_of_lambda=[2.5,.2,.7,.08,.2,2,.3]#JuvTwitter
		# list_of_lambda=[.4,2,.7,.9,.5,.2,.6,1.3,1.5]#jaya 
		# list_of_lambda=[.9,.4,.2,.6,1.1]#Msmall
		# list_of_lambda=[.5,.2,.7,.9,2] # barca
		list_of_lambda=[.2,.5,.7,1.0,1.5] # british
		# list_of_lambda=[2.2,2.5,.4,1.1,.2,2.0,.7,.6,2.8,1.2] # VTwitter
		list_of_time_span=np.array([0.0,.1,.2,.3,.4,.5]) #  ,.6,.7,.8,.9,1.0])
		list_of_method=['slant','cherrypick','Robust_cherrypick']
		list_of_process=['hawkes','poisson']		
		#########################################
		# l={'hawkes':{},'poisson':{}}
		# l['hawkes']['slant']=[.7]+[.5] * 5 
		# l['hawkes']['cherrypick']=[1.5]+[1.0]*5
		# l['hawkes']['Robust_cherrypick']=[1.5]+[1.0]*5
		# l['poisson']['slant']=[.2] * 6 
		# l['poisson']['cherrypick']=[.7] * 6 
		# l['poisson']['Robust_cherrypick']=[.7] * 6

		file_lambda='../result_performance_forecasting_pkl/'+file_prefix+'.selected'
		l=load_data(file_lambda)['lambda']
		print l
	
		#########################################
		file_to_save='../result_performance_forecasting_pkl/'+file_prefix+'.all.'+measure
		plot_to_save='../Plots/Plots_forecasting/'+file_prefix+'.all.'+measure+'.jpg'
		file_to_save_selected='../result_performance_forecasting_pkl/'+file_prefix+'.selected'
		# plot_result_forecasting_MSE_asymmetric_slant(file_prefix,file_to_save,list_of_process,list_of_method,list_of_time_span,list_of_lambda,plot_to_save,l, file_to_save_selected,flag_plot_all)
		file_to_plot_selected='../Plots/Plots_forecasting/'+file_prefix+'_'+measure+'.jpg'	
		# plot_final_results_FR(file_prefix,file_to_save_selected,file_to_plot_selected,'FR')
		if not flag_plot_all:
			plot_final_result_performance(file_prefix,file_to_save_selected,file_to_plot_selected,measure,skip_ytick)


	if plot_forecasting_all_new_flag:

		# for file_prefix in file_prefix_list:
		# 	f=plt.figure()		
		# 	plt.plot(range(10))
		# 	plt.ylabel(file_prefix,rotation=90,fontsize=22.7,weight='bold')
		# 	plt.savefig(file_prefix+'_name.jpg')
		# return 
		file_prefix=file_prefix_list[int(sys.argv[1])]
		file_to_save='../result_performance_forecasting_pkl/'+file_prefix+'.modified'
		list_of_lambda =[.5,.7,1,1.3,2]#[.6,.8,.9,1.1]
		list_of_time_span=np.array([0.0,.1,.2,.3,.4,.5]) #  ,.6,.7,.8,.9,1.0])
		list_of_method=['slant','cherrypick','Robust_cherrypick']
		list_of_process=['hawkes','poisson']
		threshold=float(sys.argv[2])
		flag_save=int(sys.argv[3]) # 1 for save 0 for plot
		hawkes_lambda_file='../Plots/slant_tuned/FR/'+file_prefix+'_desc'
		# plot_result_forecasting_FR(file_prefix,file_to_save,flag_save,threshold,list_of_process,list_of_method,list_of_time_span,hawkes_lambda_file,set_of_lambda=list_of_lambda)

		l={}
		l['poisson_slant']=[2]#[.5]#[.5]#[2]#[2]#[.5]
		l['poisson_cherrypick']=[2]#[.5]#[2]#[2]#[2]#[.7]
		l['poisson_Robust_cherrypick']=[.7]#[.5]#[2]#[.7]#[.5]#[1]
		# plot_result_forecasting_FR(file_prefix,file_to_save,flag_save,threshold,list_of_process,list_of_method,list_of_time_span,hawkes_lambda_file,selected_lambda=l)		
		file_to_plot='../Plots/Plots_forecasting/'+file_prefix+'.jpg'
		# plot_final_results_FR( file_prefix,file_to_save, file_to_plot,'MSE')	
		file_to_plot='../Plots/Plots_forecasting/'+file_prefix+'_FR.jpg'	
		plot_final_results_FR(file_prefix,file_to_save,file_to_plot,'FR')
	
	if save_result_forecasting_poisson:

		# for cherrypick robust cherrypick 
		file_prefix='../result_forecasting_Poisson/'+file_prefix_list[int(sys.argv[1])]+'w10v10f0.8' 
		list_of_lambda =[.5,.7,.9,1,2]#[.6,.8,.9,1.1]
		list_of_time_span =np.array([.1,.2,.3,.4,.5]) #  ,.6,.7,.8,.9,1.0])
		list_of_method=['cherrypick','Robust_cherrypick']
		file_to_save='../result_performance_forecasting_pkl/'+file_prefix_list[int(sys.argv[1])]+'.poisson'
		# save_all_result_forecasting_poisson(file_prefix, list_of_lambda, list_of_time_span, list_of_method, file_to_save )

		file_Hawkes='../result_performance_forecasting_pkl/'+file_prefix_list[int(sys.argv[1])]
		file_Poisson=file_Hawkes+'.poisson'
		file_to_save=file_Poisson+'.incomplete'
		flag_save=int(sys.argv[2]) # 1 for save 0 for plot
		plot_and_select_lambda(file_Hawkes, file_Poisson, list_of_method, file_to_save, flag_save)
		file_to_plot='../Plots/Plots_forecasting/'+ file_prefix_list[int(sys.argv[1])]+'_MSE.jpg'
		file_to_load='../result_performance_forecasting_pkl/'+file_prefix_list[int(sys.argv[1])]+'.poisson.incomplete'
		# plot_final_results( file_prefix_list[int(sys.argv[1])],file_to_load, file_to_plot)






		# file_prefix='../result_forecasting_Poisson/'+file_prefix_list[int(sys.argv[1])]+'w10v10' 
		# list_of_lambda =[.1] # [.5,.7,.9,1,2]#[.6,.8,.9,1.1]
		# list_of_time_span =np.array([.1,.2,.3,.4,.5]) #  ,.6,.7,.8,.9,1.0])
		

		# file_to_save='../result_performance_forecasting_pkl/'+file_prefix_list[int(sys.argv[1])]+'.poisson'
		

		# save_slant_poisson(file_prefix, list_of_lambda, list_of_time_span, file_to_save )

		# file_incomplete='../result_performance_forecasting_pkl/'+file_prefix_list[int(sys.argv[1])]+'.poisson.incomplete'
		# file_Poisson_slant='../result_performance_forecasting_pkl/'+file_prefix_list[int(sys.argv[1])]+'.poisson'
		# file_to_save='../result_performance_forecasting_pkl/'+file_prefix_list[int(sys.argv[1])]+'.poisson.complete'
		# plot_and_select_lambda_slant(file_incomplete, file_Poisson_slant, file_to_save)
		# file_to_plot='../Plots/Plots_forecasting/'+ file_prefix_list[int(sys.argv[1])]+'_MSE.jpg'
		# file_to_load='../result_performance_forecasting_pkl/'+file_prefix_list[int(sys.argv[1])]+'.poisson.complete'
		# plot_final_results( file_prefix_list[int(sys.argv[1])],file_to_load, file_to_plot)
		


		# for cherrypick robust cherrypick 
		file_prefix='../result_forecasting_Poisson/'+file_prefix_list[int(sys.argv[1])]+'w10v10f0.8' 
		list_of_lambda =[.5,.7,.9,1,2]#[.6,.8,.9,1.1]
		list_of_time_span =np.array([.1,.2,.3,.4,.5]) #  ,.6,.7,.8,.9,1.0])
		list_of_method=['cherrypick','Robust_cherrypick']
		file_to_save='../result_performance_forecasting_pkl/'+file_prefix_list[int(sys.argv[1])]+'.poisson'
		# save_all_result_forecasting_poisson(file_prefix, list_of_lambda, list_of_time_span, list_of_method, file_to_save )

		file_Hawkes='../result_performance_forecasting_pkl/'+file_prefix_list[int(sys.argv[1])]
		file_Poisson=file_Hawkes+'.poisson'
		file_to_save=file_Poisson+'.incomplete'
		flag_save=int(sys.argv[2]) # 1 for save 0 for plot
		plot_and_select_lambda(file_Hawkes, file_Poisson, list_of_method, file_to_save, flag_save)
		file_to_plot='../Plots/Plots_forecasting/'+ file_prefix_list[int(sys.argv[1])]+'_MSE.jpg'
		file_to_load='../result_performance_forecasting_pkl/'+file_prefix_list[int(sys.argv[1])]+'.poisson.incomplete'
		# plot_final_results( file_prefix_list[int(sys.argv[1])],file_to_load, file_to_plot)

		# save_slant_poisson()		
		# measure='MSE'
		# threshold=.37
		# file_prefix=file_prefix_list[int(sys.argv[1])]
		# list_of_lambda_poisson=[.5, 1]
		# list_of_time_span=[ 0 , .1, .2, .3 , .4, .5 ]
		# file_to_read_prefix = '../result_performance_forecasting/'+file_prefix+'/' + file_prefix 
		# file_to_read_desc='../Plots/slant_tuned/FR/'+file_prefix+'_desc'
		# # file_to_save_plot='../Plots/slant_tuned/'+measure+'/'+file_prefix
		# result_file='../result_performance_forecasting_pkl/'+file_prefix
		# save_result_forecasting_performance(file_to_read_prefix, list_of_time_span, file_to_read_desc,measure,threshold,result_file)

		# for file in file_prefix_list:
		# 	file_to_update = '../result_performance_forecasting_pkl/'+file+'.poisson'
		# 	res=load_data(file_to_update)
		# 	res['list_of_lambda']=[.5,.7,.9,1,2]
		# 	save(res,file_to_update)
	
		if plot_result_forecasting_FR_flag:
			# file_prefix=file_prefix_list[int(sys.argv[1])]
			file_to_save='../result_performance_forecasting_pkl/'+file_prefix+'.poisson.complete.FR'
			# flag_save=int(sys.argv[3])
			# threshold=float(sys.argv[2])
			# file_prefix=file_prefix_list[int(sys.argv[1])]
			frac=0.8
			list_of_process=['hawkes','poisson']
			list_of_methods=['slant','cherrypick','Robust_cherrypick']
			list_of_time_span=[0.0,0.1,0.2,0.3,0.4,0.5]
			hawkes_lambda_file='../Plots/slant_tuned/FR/'+file_prefix+'_desc'
			poisson_lambda_file='../result_performance_forecasting_pkl/'+file_prefix+'.poisson.complete'
			#------------
			print file_prefix
			print 'Hawkes cpk',load_data(hawkes_lambda_file)['cpk']
			print 'Hawkes rcpk',load_data(hawkes_lambda_file)['rcpk']
			
			# print 'Poisson'
			print 'P slant', load_data(poisson_lambda_file)['poisson_lambda']['slant']
			# print 'P rcpk',load_data(poisson_lambda_file)['poisson_lambda']['Robust_cherrypick']
			print 'P cpk',load_data(poisson_lambda_file)['poisson_lambda']['cherrypick']
			print 'P rcpk',load_data(poisson_lambda_file)['poisson_lambda']['Robust_cherrypick']
			# return 


		
		# plot_result_forecasting_FR(file_prefix, file_to_save, flag_save,threshold, frac, list_of_process, list_of_methods,list_of_time_span, hawkes_lambda_file, poisson_lambda_file)
		# file_to_load=file_to_save
		file_to_plot='../Plots/Plots_forecasting/'+file_prefix+'_FR.jpg'
		# plot_final_results_FR(file_prefix,file_to_load,file_to_plot)

	

	if plot_forecasting_performance_wt_poisson: # forecasting performance
		measure='MSE'
		threshold=.37
		file_prefix=file_prefix_list[int(sys.argv[1])]
		# list_of_lambda=[.5, 1]
		# list_of_time_span=[ 0 , .1, .2, .3 , .4, .5 ]
		# plot_slant_result_with_lambda_manual_tuning(file_prefix, list_of_lambda, list_of_time_span)
		window=3
		list_of_time_span =np.array([0.0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
		list_of_lambda =[.5,.6,.7,.8,.9,1.1,2]#[.6,.8,.9,1.1]
		file_to_read_prefix = '../result_performance_forecasting/'+file_prefix+'/' + file_prefix 
		file_to_read_desc='../Plots/slant_tuned/FR/'+file_prefix+'_desc'
		file_to_save_plot='../Plots/slant_tuned/'+measure+'/'+file_prefix
		result_file='../result_performance_forecasting_pkl/'+file_prefix
		save_result_forecasting_performance(file_to_read_prefix, list_of_time_span, file_to_read_desc,measure,threshold,result_file)


	if plot_manual_tuning: # forecasting performance
		measure='MSE'
		threshold=.37
		file_prefix=file_prefix_list[0]
		# list_of_lambda=[.5, 1]
		# list_of_time_span=[ 0 , .1, .2, .3 , .4, .5 ]
		# plot_slant_result_with_lambda_manual_tuning(file_prefix, list_of_lambda, list_of_time_span)
		window=3
		list_of_time_span =np.array([0.0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
		list_of_lambda =[.5,.6,.7,.8,.9,1.1,2]#[.6,.8,.9,1.1]
		file_to_read_prefix = '../result_performance_forecasting/'+file_prefix+'/' + file_prefix 
		file_to_write='../Plots/slant_tuned/'+measure+'/'+file_prefix+'_desc'
		file_to_save_plot='../Plots/slant_tuned/'+measure+'/'+file_prefix
		plot_slant_result_with_lambda_manual_tuning_v2(file_to_read_prefix, list_of_lambda, list_of_time_span, file_to_write,window,measure,threshold,file_to_save_plot)

	
	if save_result_variation_fraction:

		list_of_fraction=[.5,.6,.7,.8,.9,1.0]
		list_of_method=['cherrypick','Robust_cherrypick']
		file_prefix=file_prefix_list[int(sys.argv[1])]
		method=list_of_method[int(sys.argv[2])]
		threshold=float(sys.argv[3])
		measure_to_plot=int(sys.argv[4])
		time_span=.2
		measure='MSE'
		desc_read='../Plots/variation_of_fraction/'+file_prefix+'_'+method+'_'+'t'+str(time_span)+'_'+measure+'_desc'
		file_write='../result_variation_fraction_pkl/'+file_prefix+'_t'+str(time_span)+'_'+method # create dir
		time_span=0.2
		file_read_prefix='../result_variation_fraction/'+file_prefix+'w10v10'
		file_read_suff='t'+str(time_span)+'.res.'+method+'.slant'
		save_variation_fraction(list_of_fraction,method,time_span, desc_read,threshold, file_write,file_read_prefix, file_read_suff,measure_to_plot)
			

	if plot_variation_fraction_flag:
		f_idx=9 # int(sys.argv[1])
		list_of_method= ['cherrypick','Robust_cherrypick']
		m_idx=1 # int(sys.argv[2])
		for file_prefix in [file_prefix_list[f_idx]]:
			# for method in ['cherrypick','Robust_cherrypick']:
			method=list_of_method[m_idx]
			time_span=0.2
			measure='MSE' # FR
			file_to_read_prefix='../result_variation_fraction/'+file_prefix+'w10v10'+'t'+str(time_span)+'.res.'+method+'.slant'
			# read desc file
			file_to_read_desc='../Plots/slant_tuned/FR/'+file_prefix+'_desc'
			index_dict=load_data(file_to_read_desc)
			l_5=index_dict['slant'][2]
			if method == 'cherrypick':
				l_3=index_dict['cpk'][2]
			else:
				l_3=index_dict['rcpk'][2]
			if l_3 == 1.0 or l_3==2.0:
				l_3 = int(l_3)
			if l_5 == 2.0 or l_5==1.0:
				l_5 = int(l_5)
			list_of_lambda=list( set([.5,.7,.9,1.1]).union( set([l_3,l_5]) ) )
			list_of_fraction=[.5,.6,.7,.8,.9,1.0]
			file_to_plot='../Plots/variation_of_fraction/'+file_prefix+'_'+method+'_'+'t'+str(time_span)+'_'+measure
			file_to_write_desc='../Plots/variation_of_fraction/'+file_prefix+'_'+method+'_'+'t'+str(time_span)+'_'+measure+'_desc'
			# plot_variation_fraction(file_to_read_prefix, list_of_lambda, l_3,l_5,list_of_fraction, file_to_plot, file_to_write_desc,measure, time_span,threshold=.3)
			print 'done'

			#-------------replots variation of fraction-----------------------

			# file_to_load=
			# file_to_plot
			# plot_variation_fraction_combined( file_to_load,file_to_plot)

	
if __name__=='__main__':
	main()
