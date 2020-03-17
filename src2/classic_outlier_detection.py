import sys
from eval_slant_module import eval_slant_module
from data_preprocess import data_preprocess
from myutil import *
from slant_wtout_network import slant_wtout_network

class classic_outlier_detection:
	def __init__(self,file_prefix,files):
		self.file_prefix=file_prefix
		self.files=files
	def load_data(self):
		self.obj=load_data(self.files['data']['prefix']+self.file_prefix+self.files['data']['suffix'])
		self.marks=self.obj.train[:,2]
	def load_slant_param(self):
		# self.lamb=load_data(self.file['lambda'])['lambda'][int_generator][method][index of time_span_input(0 for nowcasting)] #file_name.selected.pkl
		dict_lamb={'GTwitter':1.0,'Twitter':0.5,'VTwitter':2.2,'STACK':[100,0.0001],'MIMIC':[10,0.0001],'FOOD':[100,0.05]}
		# dict_frac={'GTwitter':0.8,'Twitter':0.8,'VTwitter':0.8,'STACK':0.95,'MIMIC':0.89,'FOOD':0.8}
		# self.frac=dict_frac[self.file_prefix]
		if self.file_prefix in ['GTwitter','Twitter','VTwitter']:
			self.lamb=dict_lamb[self.file_prefix]
			self.w=load_data('w_v')[self.file_prefix]['w']
			self.v=load_data('w_v')[self.file_prefix]['v']
		if self.file_prefix in ['STACK','MIMIC','FOOD']:
			self.tuning_param=dict_lamb[self.file_prefix]
			self.lamb=self.tuning_param[1]
		
	def remove_outlier(self):
		mark_mean=np.average(self.marks)
		mark_std=np.std(self.marks)
		self.msg_end = (self.marks <= mark_mean+2*mark_std) & (self.marks >= mark_mean-2*mark_std)

	def run_slant_wtout_network(self):
		self.obj.train=self.obj.train[self.msg_end]
		sl_obj=slant_wtout_network(self.obj,self.tuning_param)
		sl_obj.estimate_param()
		self.sl_res={str(self.lamb):sl_obj.predict(num_simulation=1,time_span_input=0.0)}


	def run_slant(self):
		# obj=eval_slant_module(file_prefix,res_prefix,subset_res_prefix,w,v,method,int_gen,list_of_trfrac,list_of_frac,list_of_lambda,list_of_time_span,num_simul)
		sl_obj=eval_slant_module(self.file_prefix,[],[],self.w,self.v,2,'Poisson',[],[],np.array([self.lamb]),np.array([0.0]),1)
		sl_obj.get_data(self.files['data']['prefix'],self.files['data']['suffix'])
		self.sl_res=sl_obj.eval_subset_selection_classic_od(self.msg_end)

	def show_result(self):
		print '********************************'
		print self.file_prefix
		print self.lamb 
		print self.sl_res[str(self.lamb)]['MSE']
		print '********************************'

def main():
	#-------------INPUT PARAMETERS--------------------------------------------------------------------------------
	file_prefix_list=['GTwitter','Twitter','VTwitter','STACK','MIMIC','FOOD']
	file_prefix=file_prefix_list[int(sys.argv[1])]
	prefix='../Cherrypick_others/Data_opn_dyn_python/'
	suffix='_10ALLXContainedOpinionX.obj'
	obj=classic_outlier_detection(file_prefix,{'data':{'prefix':prefix,'suffix':suffix}})
	obj.load_data()
	obj.remove_outlier()	
	obj.load_slant_param()
	if int(sys.argv[1]) < 3:
		obj.run_slant()
	else:
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

