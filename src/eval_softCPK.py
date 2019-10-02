import sys
from eval_slant_module import eval_slant_module
from data_preprocess import data_preprocess
from myutil import *
from slant_wtout_network import slant_wtout_network
from softRCPK import softRCPK
from softRCPK_wtout_network import softRCPK_wtout_network

class eval_softRCPK:

	def __init__(self,file_prefix,files):
		self.file_prefix=file_prefix
		self.files=files

	def load_data(self):
		self.obj=load_data(self.files['data']['prefix']+self.file_prefix+self.files['data']['suffix'])
		self.marks=self.obj.train[:,2]

	def load_slant_param(self):
		dict_lamb={'GTwitter':1.0,'Twitter':0.5,'VTwitter':2.2,'STACK':[100,0.0001],'MIMIC':[10,0.0001],'FOOD':[100,0.05]}
		dict_frac={'STACK':0.95,'MIMIC':0.89,'FOOD':0.8}
		if self.file_prefix in ['GTwitter','Twitter','VTwitter']:
			self.w=load_data('w_v')[self.file_prefix]['w']
			self.v=load_data('w_v')[self.file_prefix]['v']
			self.lamb=dict_lamb[self.file_prefix]
			self.frac=0.8
		if self.file_prefix in ['STACK','MIMIC','FOOD']:
			self.tuning_param=dict_lamb[self.file_prefix]
			self.lamb=self.tuning_param[1]
			self.win=self.tuning_param[0]
			self.frac=dict_frac[self.file_prefix]

	def set_lamb_wt_network(self,lamb):
		self.lamb=lamb
		# if has_attr(self,'tuning_param'):
		# 	self.tuning_param[1]=lamb

	def remove_outlier(self):
		scpk_obj = softRCPK(self.obj,{'lamb':self.lamb,'w':self.w,'frac':self.frac}) 
		scpk_obj.init_DS()
		self.msg_end=scpk_obj.RR_via_soft_threshold(max_itr=20) 
		
	def remove_outlier_wtout_network(self):
		scpk_obj = softRCPK_wtout_network(self.obj,{'lamb':self.lamb,'win':self.win,'frac':self.frac}) 
		scpk_obj.init_DS()
		self.msg_end=scpk_obj.RR_via_ST(max_itr=20) 

	def run_slant_wtout_network(self):
		full_train=np.copy(self.obj.train)
		self.obj.train=self.obj.train[self.msg_end]
		sl_obj=slant_wtout_network(self.obj,self.tuning_param)
		sl_obj.estimate_param()
		sl_obj.train=full_train
		self.sl_res={str(self.lamb):sl_obj.predict(num_simulation=1,time_span_input=0.0)}

	def run_slant(self):
		sl_obj=eval_slant_module(self.file_prefix,[],[],self.w,self.v,2,'Poisson',[],[],np.array([self.lamb]),np.array([0.0]),1)
		sl_obj.get_data(self.files['data']['prefix'],self.files['data']['suffix'])
		self.sl_res=sl_obj.eval_subset_selection_classic_od(self.msg_end)

	def show_result(self):
		print '********************************'
		print self.file_prefix
		print self.lamb 
		if self.file_prefix in ['STACK','MIMIC']:
			print self.sl_res[str(self.lamb)]['FR']
		else:		
			print self.sl_res[str(self.lamb)]['MSE']
		print '********************************'

def main():
	#-------------INPUT PARAMETERS--------------------------------------------------------------------------------
	file_prefix_list=['GTwitter','Twitter','VTwitter','FOOD','STACK','MIMIC']
	file_prefix=file_prefix_list[int(sys.argv[1])]
	prefix='../Cherrypick_others/Data_opn_dyn_python/'
	suffix='_10ALLXContainedOpinionX.obj'
	obj=eval_softRCPK(file_prefix,{'data':{'prefix':prefix,'suffix':suffix}})
	obj.load_data()
	obj.load_slant_param()
	if int(sys.argv[1]) < 3:
		obj.remove_outlier()
		obj.run_slant()
	else:
		obj.remove_outlier_wtout_network()
		obj.run_slant_wtout_network()
	obj.show_result()

if __name__=="__main__":
	main()


	# {'STACK':{'lamb':0.0001,'win':100,'frac':0.95},'MIMIC':{'lamb':0.0001,'win':10,'frac':0.89},'FOOD':{'lamb':{'Robust_Lin': 10,'Robust NonLinear': 10 , 'MTPP':0.05 },'win':100,'frac':0.8}}
	# file_prefix_list=['Series','Election','Movie']#['GTwitter','Twitter','VTwitter']
	# for file_name in file_prefix_list:
	# 	dic=load_data(file_name+'.selected')
	# 	print dic['lambda']['hawkes']['MTPP'][0]
	# return 

