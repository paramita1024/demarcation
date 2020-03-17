import sys
from myutil import *
import numpy as np
from myutil import *
from synthetic_data import synthetic_data
from synthetic_data_msg import synthetic_data_msg
from subset_selection import subset_selection
from eval_slant_module import eval_slant_module

class eval_slant_synthetic:
	
	def __init__(self,file_prefix,data_prefix,data_suffix,subset_prefix,slant_prefix,lamb_list,method_list,noise_list,w,v):
		self.file_prefix=file_prefix
		self.prefix=data_prefix
		self.suffix=data_suffix
		self.subset_prefix=subset_prefix
		self.slant_prefix=slant_prefix
		self.lamb_list=lamb_list
		self.method_list=method_list
		self.noise_list=noise_list
		self.w=w
		self.v=v

	def subset_selection(self,method,noise): 
		file_prefix=self.file_prefix+'.noise.'+str(noise)
		obj=subset_selection(file_prefix,'',self.w,method,[],np.array([0.8]),self.lamb_list)
		obj.get_data(self.prefix,self.suffix)
		res=obj.run_subset_loop_synthetic()
		for lamb in self.lamb_list:
			file_save=self.subset_prefix+file_prefix+'l'+str(lamb)+'.'+self.method_list[method]
			save(res[str(lamb)],file_save)
		# return res

	def subset_selection_outer_loop(self):
		for method in [0,1]:
			for noise in self.noise_list:
				self.subset_selection(method,noise)

	def eval_slant_outer_loop(self):
		for noise in self.noise_list:
			for method in range(len(self.method_list)):
				print '-------------------',noise,'--------',method,'------------'
				self.eval_slant_module(noise,method)
				print '----------------------------------------------------------'

	def eval_slant_module(self,noise,method):
		
		file_prefix=self.file_prefix+'.noise.'+str(noise)
		num_simul=1
		obj=eval_slant_module(file_prefix,'',self.subset_prefix,self.w,self.v,method,'Poisson',[],np.array([0.8]),self.lamb_list,np.array([0.0]),num_simul)
		obj.get_data(self.prefix,self.suffix)
		if method < 2 :
			res=obj.eval_subset_selection_synthetic()
		else:
			res=obj.eval_slant_synthetic()
		for lamb in self.lamb_list:
			file_save=self.slant_prefix+file_prefix+'l'+str(lamb)+'.'+self.method_list[method]
			save(res[str(lamb)],file_save)

	def general_outer_loop(self):
		# self.subset_selection_outer_loop()
		self.eval_slant_outer_loop()

def main():

	file_list=['barabasi']+['kron_'+strng+'_512' for strng in ['std','CP','Hetero','Hier','Homo']]
	file_prefix=file_list[int(sys.argv[1])]
	data_prefix='../result_synthetic_dataset/dataset/'
	data_suffix=''
	subset_prefix='../result_synthetic_dataset/subset/'
	slant_prefix='../result_synthetic_dataset/slant_res/'
	lamb_list=np.array([.01,.05,.1,.5,.7,1,1.5])
	method_list=['cpk','rcpk','slant'] 
	noise_list=np.array([.5,.75,1,1.5,2,2.5])
	w=load_data('../result_synthetic_dataset/w_v_synthetic')[file_prefix]['w']
	v=load_data('../result_synthetic_dataset/w_v_synthetic')[file_prefix]['v']
	obj=eval_slant_synthetic(file_prefix,data_prefix,data_suffix,subset_prefix,slant_prefix,lamb_list,method_list,noise_list,w,v)
	obj.general_outer_loop()

if __name__=="__main__":
	main()

