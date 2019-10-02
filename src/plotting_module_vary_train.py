from myutil import *
from data_preprocess import data_preprocess
import pickle
import numpy as np
from eval_slant_module import *
import matplotlib.pyplot as plt

class plot_vary_train_set:
	def __init__(self,file_prefix,res_file_prefix,plot_file_prefix,trfr_list,lamb_list,time_span,measure,w,v):
		self.file_prefix=file_prefix
		self.res_file=res_file_prefix+self.file_prefix+'.t.'+str(time_span)+'.vary_train.'+measure
		self.final_res_file=res_file_prefix+self.file_prefix+'.t.'+str(time_span)+'.vary_train.'+measure+'.final'
		self.final_res_file_matlab=res_file_prefix+self.file_prefix+'.t.'+str(time_span)+'.vary_train.'+measure+'.final.matlab.txt'
		self.plot_file=plot_file_prefix+self.file_prefix+'_vary_train_'+measure
		self.sel_lamb_file=res_file_prefix+'t.'+str(time_span)+'.vary_train'
		self.res_file_prefix=res_file_prefix
		self.method_list=['slant','cpk','rcpk']
		self.trfr_list=trfr_list
		self.frac=0.8
		self.lamb_list=lamb_list
		self.time_span=time_span
		self.measure=measure
		self.w=w
		self.v=v
	def read_from_file(self,trfr,lamb,method):
		if method=='slant':
			frac=1.0
		else:
			frac=self.frac
		param_str='w'+str(self.w)+'v'+str(self.v)+'trfr'+str(trfr)+'f'+str(frac)+'l'+str(lamb)+'t'+str(self.time_span)+'.'+method
		file_name=self.res_file_prefix+self.file_prefix+param_str
		return load_data(file_name)[self.measure]
	def get_data_method(self,method):
		final_res=[]
		for lamb in self.lamb_list:
			res=[]
			for trfr in self.trfr_list:
				res.append(self.read_from_file(trfr,lamb,method))
			final_res.append(res)
		return np.array(final_res)
	def other_plot_params(self):
		plt.grid(True)
		plt.legend()
		plt.xlabel('Train set size',rotation=0,fontsize=20.7,weight='bold')
		plt.ylabel(self.measure, fontsize=22.7,weight='bold')
		plt.title(self.file_prefix+'.vary train.'+self.measure,rotation=0,fontsize=20.7,weight='bold')
		plt.tight_layout()
		plt.savefig(self.plot_file+'.all.jpg')	
	def plot_data(self,res):
		marker_seq=['P','o','*']
		ls_seq=[':','-.','--']
		lw_seq=[4,6,8]
		mk_size_seq=[4,6,8]
		f=plt.figure()
		for method,mk,ls,lw,mk_size in zip(self.method_list,marker_seq,ls_seq,lw_seq,mk_size_seq):
			self.plot_data_method(res[method],method,lw,ls,mk,mk_size)
		self.other_plot_params()
		plt.show()
		plt.clf()

	def plot_data_method(self,dataset,method,lw,ls,mk,mk_size):
		for data,lamb in zip(dataset,self.lamb_list):
			line=plt.plot(ma(data,3), label=method+'.l.'+str(lamb)) 
			plt.setp(line, linewidth=lw,linestyle=ls,marker=mk, markersize=mk_size)
		
	def specify_param(self):
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
		
	def iterator_for_plot_all(self,load):
		if load:
			res=load_data(self.res_file)
		else:
			res={}
			for method in self.method_list:
				res[method]=self.get_data_method(method)
			save(res,self.res_file)
		self.plot_data(res)

	def get_data_method_specific(self,method,param):
		res=[]
		for l,trfr in zip(param,self.trfr_list):
			res.append(self.read_from_file(trfr,l,method))
		return np.array(res)

	def get_final_data(self):
		param=load_data(self.sel_lamb_file)[self.file_prefix] 
		res={}
		for method in self.method_list:
			res[method]=self.get_data_method_specific(method,param[method])
		save(res,self.final_res_file)

	def other_plot_params_final(self,res):

		plt.grid(True)
		plt.legend()
		plt.xlabel('Fraction of Train Data Used',rotation=0,fontsize=20.7,weight='bold')
		plt.ylabel(self.measure, fontsize=22.7,weight='bold')
		plt.xticks(np.arange(self.trfr_list.shape[0]),self.trfr_list,rotation=0,fontsize=20.7,weight='bold')
		
		# if int(sys.argv[2])>0:
		# 	ytick=[.001*i for i in range(65,81,5)]
		# 	res=load_data(self.final_res_file)
		# 	res['yticks']=ytick
		# 	save(res,self.final_res_file)
		ytick=res['yticks']
		plt.yticks(ytick,rotation=0,fontsize=20.7,weight='bold')
		
		plt.title(self.file_prefix,rotation=0,fontsize=20.7,weight='bold')
		plt.tight_layout()
		plt.savefig(self.plot_file+'.pdf',dpi=600, bbox_inches='tight')

	def plot_final_data(self):
		res=load_data(self.final_res_file)
		marker_seq=['P','o','*']
		ls_seq=[':','-.','--']
		lw=4
		mk_size=10
		f=plt.figure()
		for method,mk,ls in zip(self.method_list,marker_seq,ls_seq):
			line=plt.plot(ma(res[method],3), label=method) 
			plt.setp(line, linewidth=lw,linestyle=ls,marker=mk, markersize=mk_size)
		self.other_plot_params_final(res)
		plt.show()
		plt.clf()

	def save_final_data_matlab(self):
		res=load_data(self.final_res_file)
		arr=np.zeros((len(self.method_list),self.trfr_list.shape[0]))
		for method,i in zip(self.method_list,range(len(self.method_list))):
			arr[i]=np.array(ma(res[method],3))
		arr=np.concatenate((self.trfr_list.reshape(1,self.trfr_list.shape[0]),arr),axis=0)
		write_txt_file(arr.transpose(),self.final_res_file_matlab)
	

def main():
	file_prefix_list=['barca','british_election','GTwitter','jaya_verdict', 'JuvTwitter' , 'MlargeTwitter','MsmallTwitter', 'real_vs_ju_703', 'trump_data' ,'Twitter','VTwitter']
	#-------------INPUT PARAMETERS----------------
	file_prefix=file_prefix_list[int(sys.argv[1])]
	res_file_prefix='../result_vary_train/'
	plot_file_prefix='../Plots/vary_train/'
	trfr_list=np.array([.5,.6,.7,.8,.9,1.])
	lamb_list=np.array([.5,.7,.9,1.5])
	time_span=0.2
	measure='MSE'
	w=load_data('w_v')[file_prefix]['w']
	v=load_data('w_v')[file_prefix]['v']
	#---------------------------------------------
	obj=plot_vary_train_set(file_prefix,res_file_prefix,plot_file_prefix,trfr_list,lamb_list,time_span,measure,w,v)
	# option=str(sys.argv[2])
	# switcher={'1':obj.iterator_for_plot_all, '2':obj.specify_param, '3':obj.iterator_for_final_plot}
	# switcher[option]()
	#---------------------------------
	# obj.iterator_for_plot_all(load=False)
	# return 
	# obj.specify_param()
	# obj.get_final_data()
	# obj.plot_final_data()
	obj.save_final_data_matlab()
	#---------------------------------

if __name__=="__main__":
	main()

