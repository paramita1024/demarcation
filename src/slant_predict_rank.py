from myutil import *
from data_preprocess import data_preprocess
import pickle
import numpy as np
from eval_slant_module import *

class slant_predict_rank:
	def __init__(self,obj,file_prefix,res_ranks,lamb,method):
		self.file_prefix=file_prefix
		self.res_ranks_prefix=res_ranks
		self.edges = obj.edges
		self.nodes = obj.nodes
		self.train = obj.train
		self.test = obj.test

		self.mu=obj.mu
		self.B=obj.B
		self.num_node = self.edges.shape[0]
		self.num_train= self.train.shape[0]
		self.num_test= self.test.shape[0]
		self.v = obj.v		
		self.lamb=lamb
		self.method=method
		self.curr_int=self.mu 
		self.time_last=0
	
	def update_history(self):
		for user,time,sentiment in self.train:
			user=int(user)
			self.relax(user,time)

	def print_info(self):
		print 'Slant_predict_rank-----'+self.file_prefix+'-----'
		print 'Number of Sample: ',(self.num_train+self.num_test)
		print 'Number of nodes: ',self.num_node
		print 'Lambda',self.lamb
		print '------------------------------'

	def get_method_str(self):
		return ['cpk','rcpk','slant'][self.method]
	
	def write_ranks(self,res):
		res_name = self.res_ranks_prefix+self.file_prefix+'.l'+str(self.lamb)+'.'+self.get_method_str()
		save(res,res_name)

	def relax_and_generate_rank(self,user,time):
		time_diff=time-self.time_last
		self.curr_int = self.mu + ( self.curr_int - self.mu) * np.exp( - self.v * time_diff) 
		rank_dict = self.generate_rank(user)
		self.curr_int += self.B[user]
		return rank_dict

	def predict_rank(self):
		res={}
		for user,time,sentiment in self.test:
			user=int(user)
			res[str(time)]=self.relax_and_generate_rank(user,time)
		self.write_ranks(res)

	def generate_rank(self,user):
		res_dict={'rank':0,'myInt':0,'predictTopInt':0,'totalInt':0}
		res_dict['myInt']=self.curr_int[user]
		res_dict['predictTopInt']=np.max(self.curr_int)
		res_dict['totalInt']=np.sum(self.curr_int)
		tmp=np.sort(self.curr_int)[::-1]
		res_dict['rank']=np.argmin(tmp==res_dict['myInt'])
		return res_dict
	def relax(self,user,time):
		time_diff=time-self.time_last
		self.curr_int = self.mu + ( self.curr_int - self.mu) * np.exp( - self.v * time_diff) + self.B[user]
		self.time_last = time

class slant_predict_rank_outer:
	def __init__(self,file_prefix,method,res_ranks,list_of_lambda):
		self.file_prefix=file_prefix
		self.method=method
		self.res_ranks=res_ranks
		self.list_of_lambda=list_of_lambda
	def get_eval_slant_module(self):
		#------------------------dummies---------------------------#
		res_prefix='../result_subset_selection_slant/'
		subset_res_prefix='../result_subset_selection/'
		w=load_data('w_v')[self.file_prefix]['w']
		v=load_data('w_v')[self.file_prefix]['v']
		int_gen=0
		list_of_trfrac=np.array([1.0])
		list_of_frac=np.array([.8])
		list_of_lambda=np.array([.5,.7,.9,1.5])
		list_of_time_span=np.array([0.0])
		num_simul=1
		prefix='../Cherrypick_others/Data_opn_dyn_python/'
		suffix='_10ALLXContainedOpinionX.obj'
		#------------------------------------------------------------#
		self.eval_slant=eval_slant_module(self.file_prefix,res_prefix,subset_res_prefix,w,v,self.method,int_gen,list_of_trfrac,list_of_frac,list_of_lambda,list_of_time_span,num_simul)
		self.eval_slant.get_data(prefix,suffix)



	def run_iterator_over_lambda(self):
		for lamb in self.list_of_lambda:
			slant_obj = self.eval_slant.eval_slant_partial(lamb)
			obj = slant_predict_rank(slant_obj,self.file_prefix,self.res_ranks,lamb,self.method)
			obj.print_info()
			obj.update_history()
			obj.predict_rank()


def check():
	file_prefix_list=['barca','british_election','GTwitter','jaya_verdict', 'JuvTwitter' , 'MlargeTwitter','MsmallTwitter', 'real_vs_ju_703', 'trump_data' ,'Twitter','VTwitter']
	#-------------INPUT PARAMETERS--------------------------------------------------------------------------------
	file_prefix=file_prefix_list[int(sys.argv[1])]
	
	for method in [0,1,2]:
		res_ranks='../result_ranks/'	
		list_of_lambda=np.array([.5])#,.7,.9,1.5])
		lamb=list_of_lambda[0]
		obj=slant_predict_rank_outer(file_prefix,method,res_ranks,list_of_lambda)
		obj.get_eval_slant_module()
		sl_obj=obj.eval_slant.eval_slant_partial(lamb)
		plt.plot(sl_obj.mu)
	plt.show()



def main():
	check()
	return 

	file_prefix_list=['barca','british_election','GTwitter','jaya_verdict', 'JuvTwitter' , 'MlargeTwitter','MsmallTwitter', 'real_vs_ju_703', 'trump_data' ,'Twitter','VTwitter']
	#-------------INPUT PARAMETERS--------------------------------------------------------------------------------
	file_prefix=file_prefix_list[int(sys.argv[1])]
	method=int(sys.argv[2])
	res_ranks='../result_ranks/'	
	list_of_lambda=np.array([.5])#,.7,.9,1.5])
	obj=slant_predict_rank_outer(file_prefix,method,res_ranks,list_of_lambda)
	obj.get_eval_slant_module()
	obj.run_iterator_over_lambda()




if __name__=="__main__":
	main()
