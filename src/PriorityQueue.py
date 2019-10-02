from math import floor
import numpy as np

class tree_op:
	def __init__(self):
		return
	def parent(self,i):
		if i==0:
			print('given index is root')
			return -1
		return int(floor((i-1)/2))
	def left(self,i):
		return 2*i+1
	def right(self,i):
		return 2*i+2


class Qarray:
	def __init__(self,Q):
		n=len(Q)
		self.Q=np.zeros((n,2))		
		for u in range(n):
			self.Q[u]=[Q[u],u]
	def cmp(self,i,j):
		#print type(i)
		#print type(j)
		return (self.Q[i][0] < self.Q[j][0])
	def cmp_with_val(self,key,i):
		return ( key < self.Q[i][0] )
	def exchange(self,i,j):
		temp = np.array(self.Q[i])
		self.Q[i]=self.Q[j]
		self.Q[j]=temp
	def get(self,i):
		return self.Q[i]
	def update_key(self,i,key):
		self.Q[i][0]=key
	def check_eq(self,i,j):
		return self.Q[i][0]==self.Q[j][0]
	def size(self):
	 	return self.Q.shape
	def get_index_and_val(self,u,stop_index):
		for i in range(stop_index):
			if self.Q[i][1]==u:
				return i,self.Q[i][0]
		print "user not found in heap"
		return -1,0
	def set(self,i,v): 
		self.Q[i]=v
		
class PriorityQueue(tree_op):
	def __init__(self,Q):
		tree_op.__init__(self)
		self.Q=Qarray(Q)
		self.flag_user=np.ones(len(Q))
		self.heapsize=len(Q)
		self.max_size = self.heapsize
		self.build_heap()
	def build_heap(self):
		for i in range(int(floor(self.heapsize/2))-1,-1,-1):
			self.heapify(i)
	def heapify(self,i):
		l=self.left(i) 
		r=self.right(i)  
		selected=i # selected is the index to be heapified next time
		if l<self.heapsize:
			if self.Q.cmp(l,selected): 
				selected=l
		if r<self.heapsize:
			if self.Q.cmp(r,selected): 
				selected=r
		if selected != i:
			self.Q.exchange(i,selected) 
			self.heapify(selected)
	def insert(self,t): 
		self.heapsize=self.heapsize+1
		if self.heapsize <= self.max_size:
			self.Q.set(self.heapsize-1,[float('Inf'),t[1]])
			self.minheap_dec_key(self.heapsize-1,t[0])
		else:
			print "maximum size of heap is reached" 
	def extract_prior(self):
		if self.heapsize < 1 :
			print("heap underflow")
		val=np.array(self.Q.get(0))
		#print "val"

		self.Q.exchange(0,self.heapsize-1)
		self.heapsize=self.heapsize-1
		self.heapify(0)
		self.flag_user[int(val[1])]=0
		return val
	def minheap_inc_key(self,i,key):
		if self.Q.cmp_with_val(key,i): 
			print("new key is smaller than current key")
		self.Q.update_key(i,key)
		self.heapify(i) 
	def minheap_dec_key(self,i,key):		
		if ~self.Q.cmp_with_val(key,i):
			print("new key is not smaller than current key")
		self.Q.update_key(i,key)
		while i>0:
			p=self.parent(i)	
			if self.Q.cmp(i,p): 
				self.Q.exchange(i,p)
				i=p
			else:
				break
		return
		
	def update_key(self,t,u):
		if self.flag_user[u]==0:
			self.flag_user[u]=1
			self.insert(np.array([t,u]))
		else:
			ind,old_t=self.Q.get_index_and_val(u,self.heapsize)
			if t > old_t:
				self.minheap_inc_key(ind,t)
			else:
				self.minheap_dec_key(ind,t)
	def print_heap(self):
		i=0
		c=1
		while i<self.heapsize:
			# for j in range(i,i+c):
			# 	print self.Q.Q[j][0]
			# print "----------"
			if i+c<=self.heapsize:
				print self.Q.Q[i:i+c,0]
			else:
				print self.Q.Q[i:self.heapsize,0]
			i=i+c
			c=c*2
	def print_elements(self):
		print "--------------------------Heap---------------------------------"
		index = 0 
		for time,user in self.Q.Q:
			if index < self.heapsize:
				print "user : "+ str(user) + ", -----time : " + str(time)
			index = index + 1