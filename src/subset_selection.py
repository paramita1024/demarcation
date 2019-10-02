import time
import matplotlib.pyplot as plt
import numpy as np
from slant import *
from myutil import *
from Robust_Cherrypick import Robust_Cherrypick 
from cherrypick import cherrypick 
class subset_selection:
	def __init__(self,file_prefix,subset_prefix,w,method,list_of_trfrac,list_of_frac,list_of_lambda):
		self.file_prefix=file_prefix
		self.subset_prefix=subset_prefix
		self.w=w
		self.method=method
		self.list_of_trfrac=list_of_trfrac
		self.list_of_frac=list_of_frac
		self.list_of_lambda=list_of_lambda
		
	def get_data(self,prefix,suffix):
		file_to_read_obj =prefix+self.file_prefix+suffix
		self.obj = load_data( file_to_read_obj) 
		self.full_train=np.copy(self.obj.train) 
		
	def get_method_str(self):
		return ['cpk','rcpk','slant'][self.method]

	def get_sigma_covar(self,lamb):
		slant_obj = slant(obj=self.obj,init_by='object',data_type='real',tuning_param=[self.w,10,lamb])
		return slant_obj.get_sigma_covar_only()
		
	def run_cpk(self,lamb,batch_size=1):
		res_dict={}
		param={'lambda':lamb,'sigma_covariance':self.get_sigma_covar(lamb)}
		cherrypick_obj = cherrypick( obj = self.obj , init_by = 'object', param = param,w=self.w,batch_size=batch_size) 
		cherrypick_obj.demarkate_process(frac_end=1)
		for frac in self.list_of_frac:
			res_dict[str(frac)]=cherrypick_obj.save_end_msg(frac_end=frac) 
		return res_dict
		
	def write_subset(self,trfr,frac,lamb,res_obj):
		file_to_write=self.subset_prefix+self.file_prefix+'w'+str(self.w)+'trfr'+str(trfr)+'f'+str(frac)+'l'+str(lamb)+'.'+self.get_method_str()
		# print file_to_write
		save(res_obj,file_to_write)

	def modify_obj_train(self,trfrac):
		num_points = int(trfrac*self.full_train.shape[0])
		self.obj.train=self.full_train[:num_points]

	def run_rcpk(self,lamb):
		res_dict={}
		Cherrypick_obj = Robust_Cherrypick( obj = self.obj , init_by = 'object', lamb = lamb , w_slant=self.w) 
		Cherrypick_obj.initialize_data_structures()
		for frac in self.list_of_frac:
			w, active_set, norm_of_residual = Cherrypick_obj.robust_regression_via_hard_threshold( method = 'FC', max_itr = 50 , frac_end = frac) 
			res_dict[str(frac)]=Cherrypick_obj.save_active_set( norm_of_residual = norm_of_residual)
		print res_dict.keys()
		return res_dict
			
	def run_subset_loop(self):
		for trfrac in self.list_of_trfrac:
			self.modify_obj_train(trfrac)
			for lamb in self.list_of_lambda:
				if self.method==0:
					res_dict=self.run_cpk(lamb)
				else:
					res_dict=self.run_rcpk(lamb)
				for f in self.list_of_frac:
					self.write_subset(trfrac,f,lamb,res_dict[str(f)])
	def run_subset_loop_synthetic(self):
		res={}
		for lamb in self.list_of_lambda:
			print '---------------------------------------------------------------------------------------'
			print 'subset selection ---- lamb:',str(lamb),self.get_method_str(),self.file_prefix,'........'
			start=time.time()
			if self.method==0:
				res[str(lamb)]=self.run_cpk(lamb,batch_size=20)['0.8']

			else:
				res[str(lamb)]=self.run_rcpk(lamb)['0.8']
			print time.time()-start,'seconds----------------------------------------------------------------'
		return res 


def main():
	file_prefix_list=['barca','british_election','GTwitter','jaya_verdict', 'JuvTwitter' , 'MlargeTwitter','MsmallTwitter', 'real_vs_ju_703', 'trump_data' ,'Twitter','VTwitter']
	#-------------INPUT PARAMETERS--------------------------------------------------------------------------------
	file_prefix=file_prefix_list[int(sys.argv[1])]
	method=int(sys.argv[2])
	list_of_trfrac=np.array([.5,.6,.7,.8,.9,1.0])
	list_of_frac=np.array([.8])
	# list_of_lambda=np.array([[0.5,.7,.9,1.5][int(sys.argv[3])]])
	list_of_lambda=np.array([.01,.05,.1,.3]) # prev ([.5,.7,.9,1.5])
	# w=int(sys.argv[5])
	w=load_data('w_v')[file_prefix]['w']
	list_of_methods=['cherrypick' , 'Robust_cherrypick'] 	
	prefix='../Cherrypick_others/Data_opn_dyn_python/'
	suffix='_10ALLXContainedOpinionX.obj'
	subset_prefix='../result_subset_selection/'
	#---------Printing-----------------------------------
	print 'File:' + file_prefix + ', Method:' + list_of_methods[method]
	print('lambda',list_of_lambda)
	#--------Exp-----------------------------------------
	start= time.time()
	obj=subset_selection(file_prefix,subset_prefix,w,method,list_of_trfrac,list_of_frac,list_of_lambda)
	obj.get_data(prefix,suffix)
	obj.run_subset_loop()
	#------------------Done--------------------------------
	print 'Evaluation Done in ', str(time.time()-start),' seconds'

if __name__=='__main__':
	main()