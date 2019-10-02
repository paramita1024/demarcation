from myutil import *
from data_preprocess import data_preprocess
import pickle
import numpy as np
from eval_slant_module import *
import matplotlib.pyplot as plt

class plotting_module_synthetic:
	def __init__(self,file_prefix,location,lamb_list,w,v):

		self.location=location#'../result_synthetic_dataset/' # location
		self.file_prefix=file_prefix
		self.demark_res_file=location+'demark_res/'
		self.demark_plot_file=location+'demark_res/plots/'
		self.pred_res_file=location+'pred_res/'+file_prefix+'.all'
		self.pred_plot_file=location+'pred_res/plots/'+file_prefix
		self.sel_lamb_file=location+'pred_res/'+file_prefix+'.sel_lamb.'
		self.sel_lamb_file=location+'demark_res/'+file_prefix+'.sel_lamb.'
		self.pred_res_file_specific=location+'pred_res/'+file_prefix
		self.subset_prefix=location+'subset/'
		self.slant_res_prefix=location+'slant_res/'
		self.datafile_pre=location+'dataset/'

		

		self.demark_method_list=['cpk','rcpk']
		self.method_list=['cpk','rcpk','slant']
		self.lamb_list=lamb_list
		self.w=w
		self.v=v
		
	def demarcation_performance(self):
		self.get_demark_data()
		self.plot_demark()
		self.specify_param_demark()
		self.get_demark_data_specific()
		self.plot_demark_final()

	def find_jaccard_index(self,array1,array2):
		return np.count_nonzero(np.logical_and(array1,array2))/np.count_nonzero(np.logical_or(array1,array2))

	def get_demark_file(self,noise,lamb,method):
		file_prefix=self.file_prefix+'.noise.'+str(noise)
		param_str='l'+str(lamb)+'.'+self.method_list[method]
		file_to_read=self.subset_res_prefix+file_prefix+param_str
		file_original=self.datafile_pre+file_prefix
		msg_end_pred=load_data(file_to_read)['data']
		msg_end_true=load_data(file_original)['msg_end_tr']
		return find_jaccard_index(msg_end_true,msg_end_pred)

	def get_demark_data(self):
		res_demark={}
		for method in self.demark_method_list:
			res_demark[method]={}
			for lamb in self.lamb_list:
				res_demark[method][lamb]=[]
				for noise in self.noise_list:
					res_demark[method][lamb].append(self.get_demark_file(noise,lamb,method))
		save(self.demark_res_file,res_demark)
		# return res_demark

	def other_plot_params_demark(self,param_str):
		plt.grid(True)
		plt.legend()
		plt.xlabel('Noise',rotation=0,fontsize=20.7,weight='bold')
		plt.ylabel('demarcation_measure', fontsize=22.7,weight='bold')
		if param_str=='all':
			plt.title(self.file_prefix+'.demark res',rotation=0,fontsize=20.7,weight='bold')
			plt.tight_layout()
			plt.savefig(self.demark_plot_file+'.all.jpg')
		else:
			plt.title(self.file_prefix+'.demark res',rotation=0,fontsize=20.7,weight='bold')
			plt.tight_layout()
			plt.savefig(self.demark_plot_file+'.jpg')
		plt.show()
			
	def plot_demark(self,res):
		res=load_data(self.demark_res_file)
		for method in self.demark_method_list:
			for lamb in self.lamb_list:
				plt.plot(res_demark[method][lamb],label=method+lamb)
		self.other_plot_params_demark('all')


	def specify_param_demark(self):
		sel_lamb={'cpk':[],'rcpk':[]}
		sel_lamb['cpk']=[float(sys.argv[3])]*6
		sel_lamb['rcpk']=[float(sys.argv[4])]*6
		params = load_data(self.sel_lamb_demark,'ifexists')
		if params:
			if self.file_prefix in params:
				print params[self.file_prefix]
				params[self.file_prefix]=sel_lamb
			else:
				params[self.file_prefix]=sel_lamb
			save(params, self.sel_lamb_file)
		else:
			sel_lamb_all={}
			sel_lamb_all[self.file_prefix]=sel_lamb
			save(sel_lamb_all,self.sel_lamb_file)


	def get_demark_data_specific(self):
		res_all=load_data(self.demark_res_file)
		sel_lamb=load_data(self.sel_lamb_demark)[self.file_prefix]
		res_specific={}
		for method in self.demark_method_list:
			res_specific[method]=[]
			for lamb,ind in zip(sel_lamb[method],range(self.noise_list.shape[1])):
				res_specific[method].append(res_all[method][str(lamb)][ind])
		save(res_specific,self.demark_res_file_specific)

	def plot_demark_data_final(self):
		res=load_data(self.demark_res_file_specific)
		marker_seq=['o','*']
		ls_seq=['-.','--']
		lw=4
		mk_size=10
		f=plt.figure()
		for method,mk,ls in zip(self.demark_method_list,marker_seq,ls_seq):
			line=plt.plot(ma(res[method],3), label=method) 
			plt.setp(line, linewidth=lw,linestyle=ls,marker=mk, markersize=mk_size)
		self.other_plot_params_demark('final')

	def prediction_performance(self):
		self.get_pred_data()
		self.plot_predict_data_all()
		self.specify_param_pred()
		self.get_pred_data_specific()
		self.plot_predict_data_final()
		
	def get_pred_file(self,noise,lamb,method):
		file_prefix=self.file_prefix+'.noise.'+str(noise)
		param_str='l'+str(lamb)+'.'+self.method_list[method]
		file_to_read=self.slant_res_prefix+file_prefix+param_str
		return load_data(file_to_read)['MSE']

	def get_pred_data(self):
		res_pred={}
		for noise in self.noise_list:
			for method in self.method_list:
				for lamb in self.lamb_list:
					res_pred[method][lamb].append(self.get_pred_file(noise,method,lamb))
		save(res_pred,self.pred_res_file)

	def plot_pred_data_all(self):

		res=load_data(self.pred_res_file)
		for method in self.method_list:
			for lamb in self.lamb_list:
				plt.plot(res[method][lamb],label=method+lamb)
		self.other_plot_params_pred('all')
		
	def other_plot_params_pred(self,param_str):
		plt.grid(True)
		plt.legend()
		plt.xlabel('Noise',rotation=0,fontsize=20.7,weight='bold')
		plt.ylabel('Preiction Error(MSE)', fontsize=22.7,weight='bold')
		if param_str=='all':
			plt.title(self.file_prefix,rotation=0,fontsize=20.7,weight='bold')
			plt.tight_layout()
			plt.savefig(self.pred_plot_file+'.all.jpg')
		else:
			plt.title(self.file_prefix,rotation=0,fontsize=20.7,weight='bold')
			plt.tight_layout()
			plt.savefig(self.pred_plot_file+'.jpg')
		plt.show()
		plt.clf()

	def specify_param_pred(self):
		sel_lamb={'slant':[],'cpk':[],'rcpk':[]}
		sel_lamb['slant']=[float(sys.argv[2])]*6
		sel_lamb['cpk']=[float(sys.argv[3])]*6
		sel_lamb['rcpk']=[float(sys.argv[4])]*6
		params = load_data(self.sel_lamb_file,'ifexists')
		if params:
			if self.file_prefix in params:
				print params[self.file_prefix]
				params[self.file_prefix]=sel_lamb
			else:
				params[self.file_prefix]=sel_lamb
			save(params, self.sel_lamb_file)
		else:
			sel_lamb_all={}
			sel_lamb_all[self.file_prefix]=sel_lamb
			save(sel_lamb_all,self.sel_lamb_file)

	def get_pred_data_specific(self):
		res_all=load_data(self.pred_res_file)
		sel_lamb=load_data(self.sel_lamb_file)[self.file_prefix]
		res_specific={}
		for method in self.method_list:
			res_specific[method]=[]
			for lamb,ind in zip(sel_lamb[method],range(self.noise_list.shape[1])):
				res_specific[method].append(res_all[method][str(lamb)][ind])
		save(res_specific,self.pred_res_file_specific)

	def plot_predict_data_final(self):
		res=load_data(self.pred_res_file_specific)
		marker_seq=['P','o','*']
		ls_seq=[':','-.','--']
		lw=4
		mk_size=10
		f=plt.figure()
		for method,mk,ls in zip(self.method_list,marker_seq,ls_seq):
			line=plt.plot(ma(res[method],3), label=method) 
			plt.setp(line, linewidth=lw,linestyle=ls,marker=mk, markersize=mk_size)
		self.other_plot_params_pred('final')

	

def main():

	# file_prefix_list=[]
	#-------------INPUT PARAMETERS----------------
	file_prefix='barabasi'#file_prefix_list[int(sys.argv[1])]
	location='../result_synthetic_dataset/'
	lamb_list=np.array([.5,.7,.9,1.5])

	w=load_data('../result_synthetic_dataset/w_v')[file_prefix]['w']
	v=load_data('../result_synthetic_dataset/w_v')[file_prefix]['v']
	#---------------------------------------------
	obj=plotting_module_synthetic(file_prefix,location,lamb_list,w,v)
	obj.demarcation_performance()
	obj.prediction_performance()
	#---------------------------------------------
if __name__=="__main__":
	main()

