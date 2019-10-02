from myutil import *
from data_preprocess import data_preprocess
import pickle
import numpy as np
from eval_slant_module import *
import matplotlib.pyplot as plt

class plot_ranking:
	def __init__(self,file_prefix,res_ranks_prefix,plot_ranks_prefix,method_list,lamb_list):
		self.file_prefix=file_prefix
		self.res_ranks_prefix=res_ranks_prefix
		self.plot_ranks_prefix=plot_ranks_prefix
		self.plot_file=plot_ranks_prefix+self.file_prefix+'.ranking.'
		self.res_file=res_ranks_prefix+self.file_prefix+'.ranking'
		self.method_list=method_list
		self.lamb_list=lamb_list

	def get_nodes_in_data(self):
		file_name = '../Cherrypick_others/Data_opn_dyn_python/'+self.file_prefix+'_10ALLXContainedOpinionX.obj'
		obj=load_data(file_name)
		return obj.edges.shape[0]

	def get_aggregates(self,res_dict):
		res={}
		rank_list=[value['rank'] for value in res_dict.values()]
		res['avgRank']=np.average(rank_list)
		plt.plot(rank_list)
		plt.show()
		# res['avgRelativeRank']=float(res['avgRank'])/self.get_nodes_in_data()
		# res['fracTopInt']=np.average([value['myInt']/value['predictTopInt'] for value in res_dict.values()])
		return res 

	def get_data_ranking(self,method,lamb):
		file_name = self.res_ranks_prefix+self.file_prefix+'.l'+str(lamb)+'.'+method
		res = load_data(file_name)
		return self.get_aggregates(res)

	def save_data(self):
		res={}
		for method in self.method_list:
			res[method]={}
			for lamb in self.lamb_list:
				res[method][lamb]=self.get_data_ranking(method,lamb)
		save(res, self.res_file)

	def other_plot_params(self,measure,flag_all):
		plt.grid(True)
		plt.legend()
		plt.xlabel('Method ',rotation=0,fontsize=20.7,weight='bold')
		plt.ylabel(measure, fontsize=22.7,weight='bold')
		plt.title(self.file_prefix+'.ranking.'+measure,rotation=0,fontsize=20.7,weight='bold')
		plt.tight_layout()
		if flag_all:
			plt.savefig(self.plot_file+'.'+measure+'.all.jpg')
		else:
			plt.savefig(self.plot_file+'.'+measure+'.jpg')

	def bar_plot(self,res,measure):
		fig, ax = plt.subplots()
		bar_width=.1
		index=np.arange(self.lamb_list.shape[0])
		arr=np.arange(12).reshape(3,4)
		for method,ind in zip(self.method_list,np.arange(len(self.method_list))):
			res_list=[]
			for lamb in self.lamb_list:
				res_list.append(res[method][lamb][measure])
			print res_list
			plt.bar(index+ind*bar_width,res_list,bar_width,label=method)
		# plt.bar(index+.1,[1,2,3,4],bar_width,label='method1')
		# plt.bar(index+.2,[5,6,7,8],bar_width,label='method2')
		# plt.bar(index+.3,[9,10,11,12],bar_width,label='method3')
		self.other_plot_params(measure,True)
		# plt.show()
		plt.clf()

	# def 



	def plot_data(self):
		res=load_data(self.res_file)
		self.bar_plot(res,'avgRank')
		# self.bar_plot(res,'avgRelativeRank')
		# self.bar_plot(res,'fracTopInt')
		

def main():
	file_prefix_list=['barca','british_election','GTwitter','jaya_verdict', 'JuvTwitter' , 'MlargeTwitter','MsmallTwitter', 'real_vs_ju_703', 'trump_data' ,'Twitter','VTwitter']
	#-------------INPUT PARAMETERS----------------
	file_prefix=file_prefix_list[int(sys.argv[1])]
	res_ranks_prefix='../result_ranking/'
	plot_ranks_prefix='../Plots/ranking/'
	lamb_list=np.array([.5])#,.7,.9,1.5])
	method_list=['slant','cpk','rcpk']
	#---------------------------------------------
	obj=plot_ranking(file_prefix,res_ranks_prefix,plot_ranks_prefix,method_list,lamb_list)
	obj.save_data()
	obj.plot_data()
	#---------------------------------
	# obj.iterator_for_plot_all
	# obj.specify_param()
	# obj.iterator_for_final_plot()
	#---------------------------------

if __name__=="__main__":
	main()
