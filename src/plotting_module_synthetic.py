from myutil import *
from data_preprocess import data_preprocess
import pickle
import numpy as np
from eval_slant_module import *
import matplotlib.pyplot as plt
from synthetic_data_msg import synthetic_data_msg

class plotting_module_synthetic:
	def __init__(self,file_prefix,loc,files,plot,lamb_list,noise_list,w,v):

		self.file_prefix=file_prefix
		self.files=files 
		self.loc=loc
		self.plot=plot 
		self.demark_method_list=['cpk','rcpk']
		self.method_list=['cpk','rcpk','slant']
		self.lamb_list=lamb_list
		self.noise_list=noise_list
		self.w=w
		self.v=v

	def demarcation_performance(self):
		# self.get_demark_data()
		# self.plot_demark()
		# self.specify_param_demark()
		# self.get_demark_data_specific()
		# self.plot_demark_data_final()
		self.save_demark_data_matlab()

	def find_jaccard_index(self,array1,array2):
		return float(np.count_nonzero(np.logical_and(array1,array2)))/np.count_nonzero(np.logical_or(array1,array2))

	def get_demark_file(self,noise,lamb,method):
		file_prefix=self.file_prefix+'.noise.'+str(noise)
		param_str='l'+str(lamb)+'.'+method
		file_to_read=self.loc['subset']+file_prefix+param_str
		file_original=self.loc['data']+file_prefix
		msg_end_pred=load_data(file_to_read)['data']
		# print msg_end_pred.shape
		obj=load_data(file_original)
		# print obj.__dict__.keys()
		# print obj.msg_end
		# print obj.msg_end_tr[:5]
		msg_end_true=obj.msg_end_tr
		# print msg_end_true
		# measure = self.find_jaccard_index(msg_end_true,msg_end_pred)
		# print measure
		return self.find_jaccard_index(msg_end_true,msg_end_pred)

	def get_demark_data(self):
		res_demark={}
		for method in self.demark_method_list:
			res_demark[method]={}
			for lamb in self.lamb_list:
				res_demark[method][str(lamb)]=[]
				for noise in self.noise_list:
					res_demark[method][str(lamb)].append(self.get_demark_file(noise,lamb,method))
		save(res_demark,self.files['demark']['all'])

	def other_plot_params_demark(self,param_str):
		plt.grid(True)
		plt.legend()
		plt.xlabel('Noise(Mean)',rotation=0,fontsize=20.7,weight='bold')
		plt.ylabel('Jaccard Distance', fontsize=22.7,weight='bold')
		plt.xticks(range(self.noise_list.shape[0]),self.noise_list,rotation=0,fontsize=20.7,weight='bold')
		
		# ytick=load_data(self.files['pred']['final'])['yitcks']
		
		ytick=[.1*i for i in range(5,11,1)]
		# res=load_data(self.files['pred']['final'])
		# res['yticks']=ytick
		# save(res,self.files['pred']['final'])
		
		plt.yticks(ytick,rotation=0,fontsize=20.7,weight='bold')
		
		if param_str=='all':
			plt.title(self.file_prefix+'.demark res',rotation=0,fontsize=20.7,weight='bold')
			plt.tight_layout()
			plt.savefig(self.plot['demark']['all'])
		else:
			plt.title(self.file_prefix,rotation=0,fontsize=20.7,weight='bold')
			plt.tight_layout()
			plt.savefig(self.plot['demark']['final'], dpi=600, bbox_inches='tight')
		plt.show()
			
	def plot_demark(self):
		print self.files['demark']['all']
		res=load_data(self.files['demark']['all'])
		lw=4
		ls_seq=['-','--']
		f=plt.figure()
		for method,ls in zip(self.demark_method_list,ls_seq):
			for lamb in self.lamb_list:
				line=plt.semilogy(res[method][str(lamb)], label=method+str(lamb)) 
				plt.setp(line, linewidth=lw,linestyle=ls)
		self.other_plot_params_demark('all')

	def specify_param_demark(self):
		sel_lamb={'cpk':[],'rcpk':[]}
		sel_lamb['cpk']=[float(sys.argv[3])]*6
		sel_lamb['rcpk']=[float(sys.argv[4])]*6
		params = load_data(self.files['demark']['sel_lamb'],'ifexists')
		if params:
			print params
		save(sel_lamb, self.files['demark']['sel_lamb'])
		
	def get_demark_data_specific(self):
		res_all=load_data(self.files['demark']['all'])
		sel_lamb=load_data(self.files['pred']['sel_lamb'])
		res_specific={}
		for method in self.demark_method_list:
			res_specific[method]=[]
			for lamb,ind in zip(sel_lamb[method],range(self.noise_list.shape[0])):
				res_specific[method].append(res_all[method][str(lamb)][ind])
		save(res_specific,self.files['demark']['final'])

	def plot_demark_data_final(self):
		res=load_data(self.files['demark']['final'])
		marker_seq=['o','*']
		ls_seq=['-.','--']
		lw=4
		mk_size=10
		f=plt.figure()
		for method,mk,ls in zip(self.demark_method_list,marker_seq,ls_seq):
			line=plt.plot(ma(res[method],3), label=method) 
			plt.setp(line, linewidth=lw,linestyle=ls,marker=mk, markersize=mk_size)
		self.other_plot_params_demark('final')

	def save_demark_data_matlab(self):
		res=load_data(self.files['demark']['final'])
		arr=np.zeros((2,self.noise_list.shape[0]))
		for method,i in zip(self.demark_method_list,range(2)):
			arr[i]=np.array(ma(res[method],3))
		arr=np.concatenate((self.noise_list.reshape(1,self.noise_list.shape[0]),arr),axis=0)
		write_txt_file(arr.transpose(),self.files['demark']['final_matlab'])
	

	def prediction_performance(self):
		# self.get_pred_data()
		# self.plot_pred_data_all()
		# self.specify_param_pred()
		# self.get_pred_data_specific()
		# self.plot_pred_data_final()
		self.save_pred_data_matlab()
		
	def get_pred_file(self,method,lamb,noise):
		file_prefix=self.file_prefix+'.noise.'+str(noise)
		param_str='l'+str(lamb)+'.'+method
		# print param_str
		file_to_read=self.loc['slant']+file_prefix+param_str
		return load_data(file_to_read)['MSE']

	def get_pred_data(self):
		res_pred={}
		for method in self.method_list:
			res_pred[method]={}
			for lamb in self.lamb_list:
				res_pred[method][str(lamb)]=[]
				for noise in self.noise_list:
					res_pred[method][str(lamb)].append(self.get_pred_file(method,lamb,noise))
		save(res_pred,self.files['pred']['all'])

	def plot_pred_data_all(self):

		res=load_data(self.files['pred']['all'])
		ls_seq=[':','-.','--']
		lw_seq=[4,4,4]
		f=plt.figure()
		for method,ls,lw in zip(self.method_list,ls_seq,lw_seq):
			for lamb in self.lamb_list:
				line=plt.plot(res[method][str(lamb)], label=method+'.l.'+str(lamb)) 
				plt.setp(line, linewidth=lw,linestyle=ls)
		self.other_plot_params_pred('all')
		
	def other_plot_params_pred(self,param_str):
		plt.grid(True)
		plt.legend()
		plt.xlabel('Mean of Noise',rotation=0,fontsize=20.7,weight='bold')
		plt.ylabel('Prediction Error(MSE)', fontsize=22.7,weight='bold')
		plt.xticks(range(self.noise_list.shape[0]),self.noise_list,rotation=0,fontsize=20.7,weight='bold')
		
		# ytick=load_data(self.files['pred']['final'])['yitcks']
		
		ytick=[.1,1,10]
		res=load_data(self.files['pred']['final'])
		res['yticks']=ytick
		save(res,self.files['pred']['final'])
		
		plt.yticks(ytick,rotation=0,fontsize=20.7,weight='bold')
		if param_str=='all':
			plt.title(self.file_prefix,rotation=0,fontsize=20.7,weight='bold')
			plt.tight_layout()
			plt.savefig(self.plot['pred']['all'])
		else:
			plt.title(self.file_prefix,rotation=0,fontsize=20.7,weight='bold')
			plt.tight_layout()
			# plt.savefig(self.plot['pred']['final'])
			plt.savefig(self.plot['pred']['final'], dpi=600, bbox_inches='tight')
		plt.show()
		plt.clf()

	def specify_param_pred(self):
		sel_lamb={'slant':[],'cpk':[],'rcpk':[]}
		sel_lamb['slant']=[float(sys.argv[2])]*6
		sel_lamb['cpk']=[float(sys.argv[3])]*6
		sel_lamb['rcpk']=[float(sys.argv[4])]*6
		params = load_data(self.files['pred']['sel_lamb'],'ifexists')
		if params:
			print params
		save(sel_lamb, self.files['pred']['sel_lamb'])

	def get_pred_data_specific(self):
		res_all=load_data(self.files['pred']['all'])
		sel_lamb=load_data(self.files['pred']['sel_lamb'])
		res_specific={}
		for method in self.method_list:
			res_specific[method]=[]
			for lamb,ind in zip(sel_lamb[method],range(self.noise_list.shape[0])):
				res_specific[method].append(res_all[method][str(lamb)][ind])
		save(res_specific,self.files['pred']['final'])

	def plot_pred_data_final(self):
		res=load_data(self.files['pred']['final'])
		marker_seq=['P','o','*']
		ls_seq=[':','-.','--']
		lw=4
		mk_size=10
		f=plt.figure()
		for method,mk,ls in zip(['slant','cpk','rcpk'],marker_seq,ls_seq):
			line=plt.semilogy(ma(res[method],3), label=method) 
			plt.setp(line, linewidth=lw,linestyle=ls,marker=mk, markersize=mk_size)
		self.other_plot_params_pred('final')

	def save_pred_data_matlab(self):
		res=load_data(self.files['pred']['final'])
		arr=np.zeros((3,self.noise_list.shape[0]))
		for method,i in zip(['slant','cpk','rcpk'],range(3)):
			arr[i]=np.array(ma(res[method],3))
		arr=np.concatenate((self.noise_list.reshape(1,self.noise_list.shape[0]),arr),axis=0)
		write_txt_file(arr.transpose(),self.files['pred']['final_matlab'])
	

def main():

	# a=[]
	# a.append(True)
	# a.append(False)
	# print np.array(a,dtype=bool)
	# return
	#-------------INPUT PARAMETERS----------------
	file_list=['barabasi']+['kron_'+strng+'_512' for strng in ['std','CP','Hetero','Hier','Homo']]
	file_prefix=file_list[int(sys.argv[1])]
	folder='../result_synthetic_dataset/'
	loc={}
	loc['data']=folder+'dataset/'
	loc['subset']=folder+'subset/'
	loc['slant']=folder+'slant_res/'
	loc['demark']=folder+'demark_res/'
	loc['pred']=folder+'pred_res/'
	# files={'all':{},'final':{}}
	files={'pred':{},'demark':{}}
	plot={'pred':{},'demark':{}}
	files['demark']['all']=loc['demark']+file_prefix+'.all'
	files['pred']['all']=loc['pred']+file_prefix+'.all'
	files['pred']['sel_lamb']=loc['pred']+'sel_lamb'
	files['demark']['final']=loc['demark']+file_prefix+'.final'
	files['demark']['final_matlab']=loc['demark']+file_prefix+'.final.matlab.txt'
	files['pred']['final']=loc['pred']+file_prefix+'.final'
	files['pred']['final_matlab']=loc['pred']+file_prefix+'.final.matlab.txt'
	plot['demark']['all']=loc['demark']+file_prefix+'.all.jpg'
	plot['pred']['all']=loc['pred']+file_prefix+'.all.jpg'
	plot['demark']['final']=loc['demark']+file_prefix+'_demark_final.pdf'
	plot['pred']['final']=loc['pred']+file_prefix+'_final.pdf'
	lamb_list=np.array([0.01,.05,.1,.5,.7,1,1.5])
	noise_list=np.array([.5,.75,1,1.5,2,2.5])
	w=load_data('../result_synthetic_dataset/w_v_synthetic')[file_prefix]['w']
	v=load_data('../result_synthetic_dataset/w_v_synthetic')[file_prefix]['v']
	#---------------------------------------------
	obj=plotting_module_synthetic(file_prefix,loc,files,plot,lamb_list,noise_list,w,v)
	# obj.demarcation_performance()
	obj.prediction_performance()
	#---------------------------------------------
if __name__=="__main__":
	main()

