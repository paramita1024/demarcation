import numpy as np
import datetime
import os 
import sys
import matplotlib.pyplot as plt
import numpy.random as rnd
import time
from slant import slant 
from myutil import *

class eval_slant_module:
	def __init__(self,file_prefix,res_prefix,subset_res_prefix,w,v,method,int_gen,list_of_trfrac,list_of_frac,list_of_lambda,list_of_time_span,num_simul):
		self.file_prefix=file_prefix
		self.res_prefix=res_prefix
		self.subset_res_prefix=subset_res_prefix
		self.w=w
		self.v=v
		self.method=method
		self.int_gen=int_gen
		self.list_of_trfrac=list_of_trfrac
		self.list_of_frac=list_of_frac
		self.list_of_lambda=list_of_lambda
		self.list_of_time_span=list_of_time_span
		self.num_simul=num_simul
	
	def get_data(self,prefix,suffix):
		file_to_read_obj =prefix+self.file_prefix+suffix
		self.obj = load_data( file_to_read_obj) 
		self.full_train=np.copy(self.obj.train)
	
	def get_subset(self,trfrac,frac,lamb):
		param_str='w'+str(self.w)+'trfr'+str(trfrac)+'f'+str(frac)+'l'+str(lamb)+'.'+self.get_method_str()
		file_to_read=self.subset_res_prefix+self.file_prefix+param_str
		return load_data(file_to_read)['data']
	
	def init_estimate_slant(self,lamb):
		slant_obj=slant( obj=self.obj,init_by='object',data_type='real',tuning_param=[self.w,self.v,lamb],int_generator=self.int_gen)
		slant_obj.estimate_param()
		return slant_obj
	
	def predict_over_time(self,slant_obj):
		res_dict={}
		for time_span_input in self.list_of_time_span:
			start = time.time()
			if time_span_input==0:
				result_obj = slant_obj.predict( num_simulation=1, time_span_input = time_span_input )
			else:
				result_obj = slant_obj.predict( num_simulation=self.num_simul,time_span_input=time_span_input )
			print 'Time Span:' + str( time_span_input ) + '		Prediction time :' + str( time.time() - start ) + ', Current TIME: ' + str( datetime.datetime.now() )
			result_obj['time_span_input'] = time_span_input 
			res_dict[str(time_span_input)]=result_obj
		return res_dict
	
	def get_method_str(self):
		return ['cpk','rcpk','slant'][self.method]
	
	def write_res_file(self,trfrac,fr,lamb,t,res_obj):
		res_obj['lambda'] = lamb
		res_obj['frac_end'] = fr
		msg=dict( res_obj['msg_set'])
		del res_obj['msg_set']
		param_str='w'+str(self.w)+'v'+str(self.v)+'trfr'+str(trfrac)+'f'+str(fr)+'l'+str(lamb)+'t'+str(t)+'.'+self.get_method_str()
		file_to_write=self.res_prefix+self.file_prefix+param_str
		save(res_obj,file_to_write)
		save(msg,file_to_write+'.msg')

	def modify_obj_train(self,trfrac):
		num_points = int(trfrac*self.full_train.shape[0])
		self.obj.train=self.full_train[:num_points]
	
	def print_train(self,obj=None):
		if obj:
			print 'Eval slant module ---------- train sample ---', obj.train.shape[0]
		else:
			print 'Eval slant module ---------- train sample ----', self.obj.train.shape[0]
	
	def eval_subset_selection_outer(self):
		for trfrac in self.list_of_trfrac:
			self.modify_obj_train(trfrac)
			self.print_train()
			full_train =  np.copy( self.obj.train )	
			for frac in self.list_of_frac:
				for lamb in self.list_of_lambda:
					end_msg=self.get_subset(trfrac,frac,lamb)
					self.obj.train=full_train[end_msg]
					self.print_train()
					slant_obj=self.init_estimate_slant(lamb)
					slant_obj.set_train(self.full_train)
					self.print_train(slant_obj)
					res_dict = self.predict_over_time(slant_obj)
					for time_span in self.list_of_time_span:
						self.write_res_file(trfrac,frac,lamb,time_span,res_dict[str(time_span)])

	def eval_subset_selection_synthetic(self): # required in eval slant synthetic 
	
		res={}
		self.print_train()
		full_train =  np.copy( self.obj.train )	
		for lamb in self.list_of_lambda:
			end_msg=load_data(self.subset_res_prefix+self.file_prefix+'l'+str(lamb)+'.'+self.get_method_str())['data']
			self.obj.train=full_train[end_msg]
			self.print_train()
			slant_obj=self.init_estimate_slant(lamb)
			slant_obj.set_train(self.full_train)
			self.print_train(slant_obj)
			res[str(lamb)]= self.predict_over_time(slant_obj)[str(self.list_of_time_span[0])]
		return res
				
	
	def eval_slant_outer(self):
		frac=1.0
		for trfrac in self.list_of_trfrac:
			self.modify_obj_train(trfrac)
			for lamb in self.list_of_lambda:
				slant_obj=self.init_estimate_slant(lamb)
				slant_obj.set_train(self.full_train)
				res_dict = self.predict_over_time(slant_obj)
				for time_span in self.list_of_time_span:
					self.write_res_file(trfrac,frac,lamb,time_span,res_dict[str(time_span)])

	def eval_slant_synthetic(self): # requited in eval slant synthetic
		res={}
		for lamb in self.list_of_lambda:
			slant_obj=self.init_estimate_slant(lamb)
			res[str(lamb)]=self.predict_over_time(slant_obj)[str(self.list_of_time_span[0])]
		return res 


	def eval_slant_partial(self,lamb): # required in slant predict rank 
		if self.method==2:
			return self.init_estimate_slant(lamb)
		else:
			trfrac=1.0
			frac=0.8
			full_train =  np.copy( self.full_train )	
			end_msg=self.get_subset(trfrac,frac,lamb)
			self.obj.train=full_train[end_msg]
			self.print_train()
			return self.init_estimate_slant(lamb)

	def eval_subset_selection_classic_od(self,end_msg): # required in classic outlier detection 
		res={}
		# self.print_train()
		full_train =  np.copy( self.obj.train )	
		for lamb in self.list_of_lambda:
			self.obj.train=full_train[end_msg]
			# self.print_train()
			slant_obj=self.init_estimate_slant(lamb)
			slant_obj.set_train(self.full_train)
			# self.print_train(slant_obj)
			res[str(lamb)]= self.predict_over_time(slant_obj)['0.0']
		return res


					


def main():

	file_prefix_list=['barca','british_election','GTwitter','jaya_verdict', 'JuvTwitter' , 'MlargeTwitter','MsmallTwitter', 'real_vs_ju_703', 'trump_data' ,'Twitter','VTwitter']
	#-------------INPUT PARAMETERS--------------------------------------------------------------------------------
	file_prefix=file_prefix_list[int(sys.argv[1])]
	method=int(sys.argv[2])
	list_of_trfrac=np.array([.5,.6,.7,.8,.9,1.0])
	list_of_frac=np.array([.8])
	list_of_lambda=np.array([.01,.05,.1,.3])#(#[[.5,.7,.9,1.5])#[int(sys.argv[3])]])#([float(sys.argv[3])])
	int_gen=['Hawkes','Poisson'][1] # [int(sys.argv[4])]
	list_of_time_span=np.array([ 0.2])
	num_simul=20
	w=load_data('w_v')[file_prefix]['w']
	v=load_data('w_v')[file_prefix]['v']
	# w=int(sys.argv[5])
	# v=int(sys.argv[6])
	list_of_methods=['cherrypick' , 'Robust_cherrypick','slant'] 	
	prefix='../Cherrypick_others/Data_opn_dyn_python/'
	suffix='_10ALLXContainedOpinionX.obj'
	res_prefix='../result_subset_selection_slant/'
	subset_res_prefix='../result_subset_selection/'
	#---------Printing-----------------------------------
	print 'File:' + file_prefix + ', Method:' + list_of_methods[method]
	print('lambda',list_of_lambda)
	#--------Exp-----------------------------------------
	start= time.time()
	obj=eval_slant_module(file_prefix,res_prefix,subset_res_prefix,w,v,method,int_gen,list_of_trfrac,list_of_frac,list_of_lambda,list_of_time_span,num_simul)
	obj.get_data(prefix,suffix)
	if method < 2 :
		obj.eval_subset_selection_outer()
	else:
		obj.eval_slant_outer()
	#------------------Done--------------------------------
	print 'Evaluation Done in ', str(time.time()-start),' seconds'

if __name__=='__main__':
	main()
