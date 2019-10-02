class simulation_exogenious:

	def __init__(self, edges):
		self.edges = edges
		self.nodes=np.arange(edges.shape[0])
		self.num_node=self.nodes.shape[0]

	# def sample_msg_times():
	# 	pass

	# def sample_opns():
	# 	pass

	def simulate_events(self, time_span , p_exo,mu, alpha, A, B, flag_check_num_msg = False, return_only_opinion_updates= False):
		
		time_init = np.zeros((self.num_node,1))
		opn_update = np.concatenate((time_init, alpha.reshape(self.num_node , 1 )), axis=1)
		int_update =  np.concatenate((time_init, mu.reshape( self.num_node , 1 )), axis=1)
		
		msg_set = []

		if time_span == 0:
			# opn_update[:,0] = self.time_last
			# int_update[:,0] = self.time_last # check whether such assignment works

			return np.array(msg_set), opn_update, int_update

		tQ=np.zeros(self.num_node)
		for user in self.nodes:
			# if mu[user] == 0 :
			# 	print "initial intensity  = zero "
			tQ[user] = self.sample_event( mu[user] , 0 , user, time_span ) 
			# tQ[user] = rnd.uniform(0,T)
		# Q=PriorityQueue(tQ) # set a chcek on it
		# print "----------------------------------------"
		# print "sample event starts"
		t_new = 0
		#--------------------------------------------------
		if flag_check_num_msg==True:
			num_msg = 0
		#--------------------------------------------------
		while t_new < time_span:
			u = np.argmin(tQ)
			t_new = tQ[u]
			tQ[u] = float('inf')
			# t_new,u=Q.extract_prior()# do not we need to put back t_new,u * what is this t_new > T 
			# u = int(u)

			# print " extracted user " + str(u) + "---------------time : " + str(t_new)
			# t_old=opn_update_time[u]
			# x_old=opn_update_val[u]
			t_old,x_old = opn_update[u,:]
			x_new=alpha[u]+(x_old-alpha[u])*np.exp(-self.w*(t_new-t_old))
			# opn_update_time[u]=t_new
			# opn_update_val[u]=x_new
			opn_update[u,:]=np.array([t_new,x_new])
			seed=rnd.uniform(0,1,1)
			if seed<p_exo:
				m=rnd.beta(5,1,1)
			else:
				m = rnd.normal( x_new , math.sqrt(self.var) )
			msg_set.append(np.array([u,t_new,m]))
			if flag_check_num_msg == True:
				num_msg = num_msg + 1 
				if num_msg > max_msg :
					break
			# update neighbours
			for nbr in np.nonzero(self.edges[u,:])[0]:
				# print " ------------for nbr " + str(nbr) + "-------------------------"
				# change above 
				t_old,lda_old = int_update[nbr]
				lda_new = mu[nbr]+(lda_old-mu[nbr])*np.exp(-self.v*(t_new-t_old))+B[u,nbr]# use sparse matrix
				int_update[nbr,:]=np.array([t_new,lda_new])
				t_old,x_old=opn_update[nbr]
				x_new = alpha[nbr] + ( x_old - alpha[nbr] )*np.exp(-self.w*(t_new-t_old)) + A[u,nbr]*m
				opn_update[nbr]=np.array([t_new,x_new])

				# print " updated int " + str(lda_new) + " ------------ updated opinion -----" + str(x_new)
				t_nbr=self.sample_event(lda_new,t_new,nbr, time_span )
				# print " update next event time of " + str( nbr ) + "  as " + str(t_nbr)
				tQ[nbr]=t_nbr
				# Q.update_key(t_nbr,nbr) 
			# Q.print_elements()
		# for m in msg_set:
		# 	print m
		# print len(msg_set)	
		# duration = time.time( ) - start_time 
		# print str(len(msg_set)) + ' msg is generated in ' + str(duration) + ' seconds.'
		if return_only_opinion_updates == True:
			return opn_update
		else:
			return np.array(msg_set) , opn_update, int_update
	

	def sample_event(self,lda_init,t_init,user, T ): # to be checked
		lda_old=lda_init
		t_new= t_init
		 
		
		# print "------------------------"
		# print "start tm "+str(t_init) + " --- int --- " + str(lda_init)
		# print "------------start--------"
		# itr = 0 
		while t_new< T : 
			u=rnd.uniform(0,1)
			# if lda_old == 0:
			# 	print "itr no " + str(itr)
			t_new -= math.log(u)/lda_old
			# print "new time ------ " + str(t_new)
			lda_new = self.mu[user] + (lda_init-self.mu[user])*np.exp(-self.v*(t_new-t_init))
			# print "new int  ------- " + str(lda_new)
			d = rnd.uniform(0,1)
			# print "d*upper_lda : " + str(d*lda_upper)
			if d*lda_old < lda_new  :
				break
			else:
				lda_old = lda_new
			# itr += 1
				

		return t_new # T also could have been returned
