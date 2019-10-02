# def simulate_events(self,time_span,mu,alpha,A,B,max_msg,var,option,p_exo=None,noise=None):#option= {0:no noise, 1:noise  }
		

class generate_events:
	def __init__(self,edges,param,var,option,noise,p_exo):
		self.edges=edges
		self.num_node=edges.shape[0]
		self.alpha=param['alpha']
		self.mu=param['mu']
		self.A=param['A']
		self.B=param['B']
		self.var=var
		self.option=option
		self.noise=noise
		self.p_exo=p_exo

	def gen_events(self,time_span,max_msg):
		self.init_var()
		t_new = 0
		num_msg = 0
		while t_new < time_span:
			new_msg=self.generate_event()
			self.influence_nbr(new_msg,time_span)
			self.msg_set.append(new_msg)
			num_msg=num_msg+1 
			if num_msg % 1000 ==0:
				print 'num_msg====>',num_msg
			if num_msg > max_msg :
				break
		return np.array(self.msg_set),np.array(self.msg_end_list, dtype=bool)
		
	def init_var(self,time_span):
		self.msg_end_list=[]
		time_init = np.zeros((self.num_node,1))
		self.opn_update = np.concatenate((time_init, self.alpha.reshape(self.num_node , 1 )), axis=1)
		self.int_update =  np.concatenate((time_init, self.mu.reshape( self.num_node , 1 )), axis=1)
		self.msg_set = []
		self.tQ=np.zeros(self.num_node)
		for user in self.nodes:
			self.tQ[user] = self.sample_event( self.mu[user] , 0 , user, time_span, self.mu[user] ) 
	
	def generate_event(self):
		u = np.argmin(self.tQ)
		t_new = self.tQ[u]
		self.tQ[u] = float('inf')
		t_old,x_old = self.opn_update[u,:]
		x_new=alpha[u]+(x_old-alpha[u])*np.exp(-self.w*(t_new-t_old))
		self.opn_update[u,:]=np.array([t_new,x_new])
		m=self.get_msg(x_new)
		return np.array([u,t_new,m])
		
	def influence_nbr(self,new_msg,time_span):
		u,t_new,m=new_msg
		for nbr in np.nonzero(self.edges[u,:])[0]:
			t_old,lda_old = self.int_update[nbr]
			lda_new = self.mu[nbr]+(lda_old-self.mu[nbr])*np.exp(-self.v*(t_new-t_old))+self.B[u,nbr]# use sparse matrix
			self.int_update[nbr,:]=np.array([t_new,lda_new])
			t_old,x_old=self.opn_update[nbr]
			x_new = self.alpha[nbr] + ( x_old - self.alpha[nbr] )*np.exp(-self.w*(t_new-t_old)) + self.A[u,nbr]*m
			self.opn_update[nbr]=np.array([t_new,x_new])
			t_nbr=self.sample_event(lda_new,t_new,nbr, time_span, self.mu[nbr] )
			tQ[nbr]=t_nbr
	
	def get_msg(self,x_new):
		if self.option==0:
			m = rnd.normal( x_new , math.sqrt(self.var) )
		if self.option==1:
			seed=rnd.uniform(0,1,1)
			if seed<self.p_exo:
				self.msg_end_list.append(False)
				seed1=rnd.uniform(0,1,1)
				if seed1 < 0.5:
					m=rnd.normal( x_new , math.sqrt(self.var) )+rnd.normal(self.noise,math.sqrt(self.var))
				else:
					m=rnd.normal( x_new , math.sqrt(self.var) )-rnd.normal(self.noise,math.sqrt(self.var))
			else:
				self.msg_end_list.append(True)
				m = rnd.normal( x_new , math.sqrt(self.var) )
		return m 

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