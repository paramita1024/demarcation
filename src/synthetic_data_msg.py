import time
from math import sqrt
from myutil import *
import numpy as np
import numpy.random as rnd
from math import ceil
import pickle
from synthetic_data import synthetic_data
import sys
from slant import slant
import math

class synthetic_data_msg_outer:
	def __init__(self,data_file,plot_file,noise_list,edges,A,B,alpha,mu,w,v):
		self.data_file=data_file
		self.plot_file=plot_file
		self.noise_list=noise_list
		self.edges=edges
		self.A=A
		self.B=B
		self.alpha=alpha
		self.mu=mu
		self.w=w
		self.v=v 

	def generate_msg_outer_loop(self,time_span,max_msg):
		for noise in self.noise_list:
			n_str='.noise.'+str(noise)
			syn_obj=synthetic_data_msg(self.plot_file+n_str+'.jpg',self.edges,self.A,self.B,self.mu,self.alpha,self.w,self.v)
			syn_obj.generate_msg(time_span,max_msg,var=.05,option=1,p_exo=0.2,noise=noise)
			syn_obj.plot_msg()
			syn_obj.split_data(.9)
			save(syn_obj,self.data_file+n_str)
	
class synthetic_data_msg:
	def __init__(self,plot_file,edges,A,B,mu,alpha,w,v):
		self.plot_file=plot_file
		self.edges=edges
		self.num_node=self.edges.shape[0]
		self.nodes=np.arange(self.num_node)
		self.A=A
		self.B=B
		self.alpha=alpha 
		self.mu=mu 
		self.w=w 
		self.v=v 

	def generate_msg(self,time_span,max_msg,var,option,p_exo=None,noise=None):
		
		dict_obj={'edges':self.edges}
		if option==1:
			self.exo=True
		self.msg, self.msg_end=self.simulate_events(time_span,self.mu,self.alpha,self.A,self.B,max_msg,var,option,p_exo,noise)
		self.num_msg = self.msg.shape[0]
		print 'maximum sentiment--->', np.max(self.msg[:,2])

	def plot_msg(self):
		
		plt.subplot(2,1,1)
		plt.plot(self.msg[:,1], self.msg[:,2])
		plt.xlabel('time')
		plt.ylabel('opinion')
		plt.legend()
		plt.subplot(2,1,2)
		plt.plot(range(self.num_msg), self.msg[:,1])
		plt.xlabel('index')
		plt.ylabel('time')
		plt.legend()
		plt.savefig(self.plot_file)
		plt.show()
		plt.clf()

	def split_data(self, train_fraction):
		num_tr = int( self.num_msg * train_fraction )
		self.train = self.msg[:num_tr]
		self.test = self.msg[num_tr:]
		if hasattr(self,'msg_end'):
			self.msg_end_tr=self.msg_end[:num_tr]
			self.msg_end_test=self.msg_end[num_tr:]
	
	def eval_slant(self,lamb):
		data_dict={'train':self.train,'test':self.test,'edges':self.edges}
		slant_obj=slant( obj=data_dict,init_by='dict',data_type='real',tuning_param=[self.w,self.v,lamb],int_generator='Poisson')
		pred_param=slant_obj.estimate_param()
		pred_res=slant_obj.predict( num_simulation=1,time_span_input=0.0)
		return pred_param,pred_res

	def plot_prediction(self,pred_res):
		plt.plot(self.test[:,2],label='true')
		plt.plot(pred_res['predicted'],label='predicted')
		plt.title('prediction')
		plt.show()
		

	def vec_nonzero(self,A):
		return A.flatten()[np.nonzero(self.A.flatten())]
		
	def plot_param(self,pred_param):

		plt.plot(self.alpha,label='true')
		plt.plot(pred_param['alpha'],label='predicted')
		plt.legend()
		plt.title('alpha')
		plt.show()

		plt.plot(self.vec_nonzero(self.A),label='true')
		plt.plot(self.vec_nonzero(pred_param['A']),label='predicted')
		plt.title('A')
		plt.legend()
		plt.show()
		
	def get_msg(self,option,x_new,var,p_exo,noise,msg_end_list):
		if option==0:
			m = rnd.normal( x_new , math.sqrt(var) )
		if option==1:
			seed=rnd.uniform(0,1,1)
			if seed<p_exo:
				msg_end_list.append(False)
				seed1=rnd.uniform(0,1,1)
				if seed1 < 0.5:
					m=rnd.normal( x_new , math.sqrt(var) )+rnd.normal(noise,math.sqrt(var))
				else:
					m=rnd.normal( x_new , math.sqrt(var) )-rnd.normal(noise,math.sqrt(var))
			else:
				msg_end_list.append(True)
				m = rnd.normal( x_new , math.sqrt(var) )
		return msg_end_list,m 

	def simulate_events(self,time_span,mu,alpha,A,B,max_msg,var,option,p_exo=None,noise=None):#option= {0:no noise, 1:noise  }
		msg_end_list=[]
		time_init = np.zeros((self.num_node,1))
		opn_update = np.concatenate((time_init, alpha.reshape(self.num_node , 1 )), axis=1)
		int_update =  np.concatenate((time_init, mu.reshape( self.num_node , 1 )), axis=1)
		msg_set = []
		tQ=np.zeros(self.num_node)
		for user in self.nodes:
			tQ[user] = self.sample_event( mu[user] , 0 , user, time_span, mu[user] ) 
		t_new = 0
		num_msg = 0
		while t_new < time_span:
			u = np.argmin(tQ)
			t_new = tQ[u]
			tQ[u] = float('inf')
			t_old,x_old = opn_update[u,:]
			x_new=alpha[u]+(x_old-alpha[u])*np.exp(-self.w*(t_new-t_old))
			opn_update[u,:]=np.array([t_new,x_new])
			msg_end_list,m=self.get_msg(option,x_new,var,p_exo,noise,msg_end_list)
			# self.influence_nbr(u,int_update,mu,t_new,B,opn_update,alpha,A,m)
			for nbr in np.nonzero(self.edges[u,:])[0]:
				t_old,lda_old = int_update[nbr]
				lda_new = mu[nbr]+(lda_old-mu[nbr])*np.exp(-self.v*(t_new-t_old))+B[u,nbr]# use sparse matrix
				int_update[nbr,:]=np.array([t_new,lda_new])
				t_old,x_old=opn_update[nbr]
				x_new = alpha[nbr] + ( x_old - alpha[nbr] )*np.exp(-self.w*(t_new-t_old)) + A[u,nbr]*m
				opn_update[nbr]=np.array([t_new,x_new])
				t_nbr=self.sample_event(lda_new,t_new,nbr, time_span, mu[nbr] )
				tQ[nbr]=t_nbr
			msg_set.append(np.array([u,t_new,m]))
			num_msg=num_msg+1 
			if num_msg % 1000 ==0:
				print 'num_msg====>',num_msg
			if num_msg > max_msg :
				break
		return np.array(msg_set),np.array(msg_end_list, dtype=bool)
		
	def sample_event(self,lda_init,t_init,user, T,mu ): 
		lda_old=lda_init
		t_new= t_init
		while t_new< T : 
			u=rnd.uniform(0,1)
			delta_t =math.log(u)/lda_old 
			t_new -= delta_t
			lda_new =mu + (lda_init-mu)*np.exp(-self.v*(t_new-t_init))
			d = rnd.uniform(0,1)
			if d*lda_old < lda_new  :
				break
			else:
				lda_old = lda_new
		return t_new 

def main():
	file_list={}
	file_list['old']=['barabasi_albert_500_5_2']+['kronecker'+strng+'512' for strng in ['_','CP','Heterophily','Hier','Homophily']]
	file_prefix_old=file_list['old'][int(sys.argv[1])]
	filename='../result_synthetic_dataset_old/'+file_prefix_old+'_msg_12500_data'
	obj=load_data(filename)
	edges = obj.edges
	A=obj.A
	alpha=obj.alpha
	mu=obj.mu
	B=obj.B
	# file_prefix_old=['','kronecker_512','kroneckerCP512','kroneckerHeterophily512','kroneckerHier512','kroneckerHomophily512'][int(sys.argv[1])]
	
	file_list['new']=['barabasi']+['kron_'+strng+'_512' for strng in ['std','CP','Hetero','Hier','Homo']]
	file_prefix=file_list['new'][int(sys.argv[1])]
	
	time_span=1000
	max_msg=30000

	flag_tune=False # True
	flag_generate_msg=True # False

	#---------------------Write w v ---------------------
	# save({'barabasi':{'w':100,'v':25}},'../result_synthetic_dataset/w_v_synthetic')
	# dict_w_v={}
	# for names in file_list['new']:
	# 	if names == 'barabasi':
	# 		dict_w_v[names]={'w':100,'v':25}
	# 	if names in  ['kron_'+strng+'_512' for strng in ['std','CP','Hetero','Homo']]:
	# 		dict_w_v[names]={'w':25,'v':25}
	# 	if names == 'kron_Hier_512':
	# 		dict_w_v[names]={'w':5,'v':1}
	# save(dict_w_v,'../result_synthetic_dataset/w_v_synthetic')
	# save({'barabasi':{'w':100,'v':25}},'../result_synthetic_dataset/w_v_synthetic')
	# print load_data('../result_synthetic_dataset/w_v_synthetic')
	# return
	#------------------------------------------------------
	if flag_tune:
		lamb=.01
		w=int(sys.argv[2])
		v=int(sys.argv[3])		
		plot_file='../result_synthetic_dataset/plots/'+file_prefix+'.msg.png'
		syn_obj=synthetic_data_msg(plot_file,edges,A,B,mu,alpha,w,v)
		# syn_obj.generate_msg(time_span,max_msg,var=.05)
		syn_obj.generate_msg(time_span,max_msg,var=.05,option=0,p_exo=0.2,noise=0)
		syn_obj.plot_msg()
		syn_obj.split_data(.9)
		param,pred=syn_obj.eval_slant(lamb) 
		syn_obj.plot_prediction(pred)
		syn_obj.plot_param(param)

	if flag_generate_msg:
		data_file='../result_synthetic_dataset/dataset/'+file_prefix
		plot_file='../result_synthetic_dataset/dataset/plots/'+file_prefix
		noise_list=np.array([.5,.75,1,1.5,2,2.5])
		w=load_data('../result_synthetic_dataset/w_v_synthetic')[file_prefix]['w']
		v=load_data('../result_synthetic_dataset/w_v_synthetic')[file_prefix]['v']
		obj=synthetic_data_msg_outer(data_file,plot_file,noise_list,edges,A,B,alpha,mu,w,v)
		obj.generate_msg_outer_loop(time_span,max_msg)


if __name__=="__main__":
	main()



			# if p_exo!=None:
			# 	seed=rnd.uniform(0,1,1)
			# 	if seed<p_exo:
			# 		msg_end_list.append(False)
			# 		if p_dist!=None:
			# 			m=p_dist() # rnd.beta(5,1,1)
			# 		if p_dist==None and noise==None:
			# 			m= rnd.normal( rnd.normal(0,1) , math.sqrt(.1))
			# 		if noise!=None: # indicates you are purturbing 
			# 			if seed < 0.5:
			# 				m=rnd.normal( x_new , math.sqrt(var) )+rnd.normal(noise,math.sqrt(var))
			# 			else:
			# 				m=rnd.normal( x_new , math.sqrt(var) )-rnd.normal(noise,math.sqrt(var))
			# 	else:
			# 		msg_end_list.append(True)
			# 		m = rnd.normal( x_new , math.sqrt(var) )
			# else:
			# 	m = rnd.normal( x_new , math.sqrt(var) )
			# 