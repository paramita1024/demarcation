			# output={'y-axis':{'MSE':MSE,'FR':FR},'lambda':desc_array,'threshold':threshold,'x-axis':list_of_fraction}
			#************************************************************************

			# file_to_load='../result_variation_fraction_pkl/'+file+'_t0.2_cherrypick'
			# file_to_write='../result_performance_txt/'+file+'.rcpk.VarFrac.FR.txt'
			# file_to_write_legend='../result_performance_txt/'+file+'.rcpk.VarFrac.FR.legend.txt'


			# res=load_data(file_to_load)
			# y=ma(res['y-axis']['FR'],3)
			# print '*----------------------------------'
			# print y			
			# x=res['x-axis']
			# print type(x)
			# print '*----------------------------------'
			# print x 

			# fw=open(file_to_write,'w')
			# for x_elm,y_elm in zip( x,y):
			# 	fw.write(str(x_elm)+' '+str(y_elm)+'\n')
				
			# fw.close()

			# fw=open(file_to_write_legend,'w')
			# for label in ['Robust Cherrypick']:
			# 	fw.write(label+' ')
			# fw.close()



		# directory_src='../result_slant_later_set_of_lambdas/'
		# directory_dest='../result_variation_fraction_f1/'
		# for file in os.listdir(directory_src):
		# 	file_prefix=file.split('w0v0')[0]
		# 	rest=file.split('w0v0')[1].split('.res.')[0]
		# 	lamb=rest.split('t')[0].split('l')[1]
		# 	time=rest.split('t')[1]
		# 	if time=='0.2' and lamb not in ['0.5','0.7','0.9','1.1']:
		# 		file_cpk=file_prefix+'w10v10f1.0lc'+lamb+'ls'+lamb+'t0.2.res.cherrypick.slant.pkl'
		# 		file_rcpk=file_prefix+'w10v10f1.0ls'+lamb+'t0.2.res.Robust_cherrypick.slant.pkl'
		# 		copy(directory_src+file,directory_dest+file_cpk)
		# 		copy(directory_src+file,directory_dest+file_rcpk)
		
		# directory='../result_subset_selection/wrong_names/'
		# for file in os.listdir(directory):
		# 	if file.split('Robust_cherrypick')[0].find('.res.') == -1:
		# 		new_file_name=file.split('Robust_cherrypick')[0]+'res.Robust_cherrypick.pkl'
		# 		os.rename(directory+file,directory+new_file_name)



# if print_sub_sel_param:
	# 	# method=0
	# 	file_prefix='../Plots/slant_tuned/FR/'
	# 	print_subset_sel_parameter( file_prefix_list, file_prefix )

	# if change_name:
	# 	# a='a_t0.2_'
	# 	# print a.replace('_t0.2_','_')
	# 	# return
	# 	# old_dir='temp/'#../paper_working_copy/FIG_new/'
	# 	# new_dir='temp1/'
	# 	old_dir='../result_synthetic_dataset/synthetic_plots/'
	# 	new_dir='../result_synthetic_dataset/synthetic_plots_new/'
	# 	for file in os.listdir(old_dir):
	# 		file_new=str(file)

	# 		# if '_t0.2_MSE' in file_new:	
	# 		print '**************'
	# 		print file_new
	# 		# file_new=file_new.replace('_t0.2_','_')
	# 		file_new=file_new.replace('.final_plot.','.')
	# 		print file_new
	# 		copyfile(old_dir+file,new_dir+file_new)

	# if create_full_data:
	# 	for file_prefix in file_prefix_list:
	# 		file_to_read='../Cherrypick_others/Data_opn_dyn_python/'+file_prefix+'_10ALLXContainedOpinionX.obj'
	# 		file_to_write='../Cherrypick_others/Data_opn_dyn_python/'+file_prefix+'_full'
	# 		obj = load_data(file_to_read)
	# 		obj.train=np.concatenate((obj.train, obj.test),axis=0)
	# 		obj.test=np.array([])
	# 		save(obj,file_to_write)

	# if copy_slant_result:
	# 	list_of_lambda=[.5,.7,.9,1.1]
	# 	for file_prefix in file_prefix_list:
	# 		for lamb in list_of_lambda:
	# 			src='../result_slant_later_set_of_lambdas/'+file_prefix+'w0v0l'+str(lamb)+'t0.2.res.slant.pkl'
	# 			if os.path.isfile(src):
	# 				dst='../result_variation_fraction/'+file_prefix+'w10v10f1.0ls'+str(lamb)+'t0.2.res.Robust_cherrypick.slant.pkl'
	# 				copyfile(src, dst)
	# 				dst='../result_variation_fraction/'+file_prefix+'w10v10f1.0lc'+str(lamb)+'ls'+str(lamb)+'t0.2.res.cherrypick.slant.pkl'
	# 				copyfile(src, dst)
	# 			else:
	# 				print 'error:'+src


	
	# if read_desc_file: # param for forecasting performance
	# 	file_prefix=file_prefix_list[4]
	# 	file_to_read='../Plots/slant_tuned/'+file_prefix+'_desc'
	# 	lamb_tuned_dict=load_data(file_to_read)
	# 	list_of_time_span =np.array([0.0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0])
	# 	for time_span_input in list_of_time_span:
	# 		if time_span_input in lamb_tuned_dict:
	# 			# print time_span_input
	# 			# print type(lamb_tuned_dict[time_span_input])
	# 			print 'time = '+str(time_span_input)+'sl,cpk,rcpk=' 
	# 			print lamb_tuned_dict[time_span_input]
	# 			# print lamb_tuned_dict[time_span_input][0]
	# 			# print lamb_tuned_dict[time_span_input][1]
	# 			# print lamb_tuned_dict[time_span_input][2]


	
	# if separate_msg_flag==True:
	# 	input_directory='../test_in/'
	# 	output_directory='../test_out/'
	# 	separate_msg(input_directory,output_directory)

	# if plot_final_tuned_images:
	# 	list_of_time_span=[]#*
	# 	for file_prefix in file_prefix_list:
	# 		file_to_save='../Plots/slant_tuned/'+file_prefix.replace('_','-')+'-slant.jpg'
	# 		plot_final_tuned_image(file_prefix,list_of_time_span,file_to_save)

	# if split_result_files_into_smaller_files:
	# 	for file_prefix in file_prefix_list:
	# 		file_to_read='../result_cpk_rcpk_new_w_after_checking/'+file_prefix+'.res.Robust_cherrypick'
	# 		split(file_to_read)
	# if check_slant_results:
	# 	for file_prefix in file_prefix_list:
	# 		file_to_read='../result_after_checking/slant/'+file_prefix+'.res.slant'
	# 		file_to_write='../Plots/After_checking/slant/'+file_prefix+'.jpg'
	# 		Plot_title=file_prefix+':: MSE vs Time span'
	# 		load_data(file_to_read)
	# 		plot_local(data,['slant'],plot_title,'time_span','MSE',list_of_time_span,file_to_write)
	# if plot_cpk_over_lambda:
	# 	list_of_lambda = [ 1e-03, 1e-02, 1e-01, 1, 10, 100  ]
	# 	for file_prefix in file_prefix_list : 
	# 		plot_slant_results_v2( file_prefix, list_of_lambda)
	# if print_slant_result :
	# 	for file_prefix in file_prefix_list : 
	# 		print_slant_results( file_prefix  )
	# if plot_slant_v2 :
	# 	for file_prefix in file_prefix_list :
	# 		plot_slant_results_v2( file_prefix, list_of_methods = [ 'cherrypick','Robust_cherrypick'] , list_of_frac = [.6,.7,.8,.9,1] )
	# if plot_slant:
	# 	for file_prefix in file_prefix_list :
	# 		plot_slant_results( file_prefix , method = 'Robust_cherrypick')
	# if plot_nowcasting : 
	# 	plot_slant_results_nowcasting( file_prefix_list , list_of_methods = [ 'slant', 'cherrypick','Robust_cherrypick'] )
	# if plot_subset_selection_slant_with_lambda : # forecasting , not sure whether last one 
	# 	file_index_set = [0,1,4,5,6,7,10]
	# 	list_of_fraction = [ .8, .9 ]
	# 	list_of_lambda_cpk = [ .5,1 ]
	# 	list_of_lambda = [.5, 1]
	# 	list_of_time_span = [ 0 , .1, .2, .3 , .4, .5 ]
	# 	list_of_method = ['cherrypick','Robust_cherrypick','slant']
	# 	for index in file_index_set:
	# 		plot_slant_result_with_lambda( file_prefix_list[index], list_of_fraction, list_of_lambda_cpk, list_of_lambda ,list_of_time_span, list_of_method )
	# # for file_prefix in file_prefix_list:
	

	# if plot_results_tuning_par_sample :
	# 	file_prefix_list.remove('trump_data')
	# 	file_prefix_list.remove('MsmallTwitter')
	# 	list_of_fraction = [ .6 ,.7, .8, .9 , 1 ]
	# 	list_of_time_span = [ 0 , .1, .2, .3 , .4, .5 ]
	# 	list_of_lambda = [  1e-02, 1e-01, .5, 1, 10  ]
	# 	list_of_method = ['cherrypick','Robust_cherrypick','slant']
	# 	num_fraction = len( list_of_fraction) 
	# 	num_lambda = len( list_of_lambda ) 
	# 	num_time_span = len( list_of_time_span )
	# 	for file_prefix in file_prefix_list:
	# 		plot_results_tuning_par_samples(file_prefix, num_fraction, num_lambda, num_time_span, list_of_fraction, list_of_lambda ,list_of_time_span, list_of_method , 'MSE')
	# if print_image_in_latex_flag:
	# 	# print_image_in_latex_sequential_way()
	# 	print_image_in_latex_using_subfig()
	# 	# file_index_set = [ 0,1,4,7]
	# 	# for index in file_index_set:
	# 	# 	print_image_in_latex( file_prefix_list[index] )
	# if plot_intensities_flag:
	# 	# plot_intensities('../result/MsmallTwitter.res.slant')
	# 	# index = int(sys.argv[1])
	# 	# file_prefix_list = [ file_prefix_list[index]]

	# 	# list_of_lambdas = [.001, .01,.05,.1,.5,.8,1,1.5,5]
	# 	list_of_w=[4]#,2,4,10]
	# 	list_of_v=[1,2,4,10]
	# 	list_of_lambdas = [1]#,.0005,.001,.005,.01,.5,1]# , .01,.05,.1,.5,.8,1,1.5,5]
	# 	num_w=len(list_of_w)
	# 	num_v=len(list_of_v)
	# 	num_l=len(list_of_lambdas)

	# 	for file_prefix in file_prefix_list:

	# 		# get train time last 
	# 		# obj=load_data('../Cherrypick_others/Data_opn_dyn_python/'+file_prefix+'_10ALLXContainedOpinionX.obj')
	# 		# train_time_last=obj.train[-1,1]
	# 		# plotting
	# 		# list_of_MSE=[]
	# 		for w_index in range( num_w):
	# 			for v_index in range( num_v):
	# 				for l_index in range(num_l):
	# 					file_to_read='../result_Analysing_slant_by_opn_int/vary_w_v_lambda_partial_users/' + file_prefix + 'w'+str(2)+'v'+str(v_index)+'l'+str(6)+'.res.slant'
	# 					# file_to_read='../result_Analysing_slant_by_opn_int/vary_w_v_lambda_partial_users/' + file_prefix + 'w'+str(w_index)+'v'+str(v_index)+'l'+str(l_index)+'.res.slant'
	# 					param={}
	# 					param['w']=4#list_of_w[w_index]
	# 					param['v']=list_of_v[v_index]
	# 					param['l']=1#list_of_lambdas[l_index]
	# 					file_to_write = '../Plots/Analysing_slant/vary_v_intensity_with_timestamps_w2l6/' + file_prefix + '_v'+str(v_index) + '.jpg'
	# 					plot_empirical_vs_estimated_quantities( file_to_read , list_of_windows, file_to_write, param)# , train_time_last=train_time_last)
	# 					# result = load_data(file_to_read)
	# 					# list_of_MSE.append(( result[0]['MSE'], param))
	# 		# file_to_write = '../Plots/Analysing_slant/vary_w_l_MSE/' + file_prefix + '.jpg'
	# 		# plot_MSE( list_of_MSE, file_prefix, file_to_write)
	
		
	# if post_process_result:
	# 	directory='.'
	# 	for file in os.listdir(directory) :
	# 		file_prefix = file.split('.res.')[0]
	# 		l_no=file_prefix[-1]
	# 		file_sub_prefix=file_prefix[:-1]
	# 		if l_no=='0':
	# 			os.rename( file, file_sub_prefix+'5.res.slant.pkl')
	# 		else:
	# 			os.rename( file, file_sub_prefix+'6.res.slant.pkl')
	

		# directory='../result_slant_manual_tuning/correct_wrong_names/'
		# for file in os.listdir(directory):
		# 	# print file 
		# 	# return 
		# 	file_prefix=file.split('w0v0l')[0]
		# 	file_suffix = file.split('w0v0')[1]
		# 	# print file_prefix
		# 	# print file_suffix 
		# 	# return
		# 	rest=file_suffix.split('.res.slant')[0].split('t')
		# 	# print rest
		# 	l_prefix=rest[0]
		# 	time_span=.1*int(round(float(rest[1])*10))
		# 	# print time_span
		# 	new_file_name = file_prefix+'w0v0'+l_prefix+'t'+str(time_span)+'.res.slant.pkl'
		# 	# print new_file_name
		# 	# return
		# 	os.rename(directory+file,directory+new_file_name)


# def plot_slant_results_nowcasting( list_of_files , list_of_methods ): 
# 	MSE = np.zeros(( len( list_of_files) , len( list_of_methods ) ) ) 
# 	FR = np.zeros(( len( list_of_files) , len( list_of_methods ) ) ) 
	
# 	data = {}# []
# 	for row, file_prefix in zip( range(len(list_of_files)) , list_of_files):
# 		for col , method in zip( range(len(list_of_methods)), list_of_methods ): 
# 			print method
# 			if method == 'slant':
# 				file_to_read = '../result/' + method + '/' + file_prefix + '.res.' + method + '.nowcasting'
# 				result = load_data( file_to_read )[0]
# 			else:
# 				file_to_read = '../result/' + method + '/' + file_prefix + '.res.' + method + '.slant.nowcasting'
# 				result = load_data( file_to_read )[0][0]
			
# 			data[method] = np.mean(result['predicted'], axis = 1)
# 			if method == 'slant':
# 				data['actual']=result['true_target']
# 			MSE[ row, col ] = result['MSE']
# 			FR[ row, col ] = result['FR']
# 		f = plt.figure()
# 		plt.plot( data['actual'], label='true opinion')
# 		plt.plot( data['slant'], label='slant')
# 		plt.plot( data['cherrypick'], label = 'cherrypick')
# 		plt.plot( data['Robust_cherrypick'], label='Robust_cherrypick')
# 		plt.legend( )
# 		plt.title('Dataset : ' + file_prefix + ' opinion for first 400-500 samples in test set ')
# 		plt.xlim((400,500))
# 		# print label
# 		# plt.show()
# 		plt.savefig( file_prefix + '_opinion for first 400-500 samples in test set .png')
# 		plt.clf()


		# for row, method in zip(data_to_plot,list_of_methods):
		# 	plt.plot( row, label = method )

	# print data['slant'].shape
	# print data['cherrypick'].shape
	# print data['Robust_cherrypick'].shape
	
	# print_for_kile( MSE, list_of_files, list_of_methods)
	# print_for_kile( FR, list_of_files, list_of_methods)

	#----------------------------------------------------------
	# for MSE_row , file_prefix in zip( MSE , list_of_files ):
	# 	f=  plt.figure()
	# 	plt.plot( MSE_row )
	# 	plt.title('MSE for dataset : '+file_prefix )
	# 	plt.savefig('MSE.'+file_prefix+'.png')
	# 	plt.clf()

	

	# for FR_row , file_prefix in zip( FR , list_of_files ):
	# 	f=  plt.figure()
	# 	plt.plot( FR_row )
	# 	plt.title('FR for dataset : '+file_prefix )
	# 	plt.savefig('FR.'+file_prefix+'.png')
	# 	plt.clf()
	#----------------------------------------------------------





			# print type(result)
			# plt.plot( result['predicted'], label=method)
			# if method=='slant':
				
			# 	plt.plot( result['true_target'], label='actual')
			#--------------------------------------------------------
			# if method=='slant':
			# 	s=plt.plot( result['predicted'], 'c')
			# 	t=plt.plot( result['true_target'], 'b')
			# if method == 'cherrypick':
			# 	cpk=plt.plot(result['predicted'], 'r' )
			# if method == 'Robust_cherrypick':
			# 	rcpk=plt.plot(result['predicted'], 'g' )
			#--------------------------------------------------------
			# print 'MSE' + str(result['MSE'])
			# print 'FR' + str(result['FR'])
			
			# data_to_plot.append( result['predicted'])
			# if method == 'Robust_cherrypick':
			# 	data_to_plot.append( result['true_target'])











	# label_list = [ 'time_span 0' , 'time span .1', 'time span .2' , 'time_span .3', 'time span .4', 'time span .5']
	# num_plot = num_time_span
	# title_txt = file_prefix 
	# xtitle = 'Fraction of endogenious msg ' # set such that fractions are shown here 
	# ytitle = 'MSE error'
	# image_title = file_prefix + '.MSE.cherrypick.png'
	# plot_result( MSE , label_list , num_plot, title_txt , xtitle, ytitle , image_title )


	# label_list = [ 'time_span 0' , 'time span .1', 'time span .2' , 'time_span .3', 'time span .4', 'time span .5']
	# num_plot = num_time_span
	# title_txt = file_prefix 
	# xtitle = 'Fraction of endogenious msg ' # set such that fractions are shown here 
	# ytitle = 'FR error'
	# image_title = file_prefix + '.FR.cherrypick.png'
	# plot_result(FR , label_list , num_plot, title_txt , xtitle, ytitle , image_title )



# def eval_using_slant_Robust_cherrypick( file_prefix, time_span_input_list,list_of_lambda = None,  num_simulation=1 ):
# 	# time_span_input_slant = # ***************************
# 	# print ' file name ' + file_prefix 
# 	method = 'Robust_cherrypick'
# 	path = '../Cherrypick_others/Data_opn_dyn_python/'
# 	file_suffix = '_10ALLXContainedOpinionX.obj'

# 	directory = '../result'
# 	if not os.path.exists(directory):
# 		os.makedirs(directory)
# 	file_to_read_obj = path + file_prefix + file_suffix 
# 	file_to_read_result = '../result/' + file_prefix + '.res.' + method
# 	file_to_write_result = '../result/' + file_prefix + '.res.' + method +'.slant'

# 	obj = load_data( file_to_read_obj) 
# 	result = load_data( file_to_read_result)
# 	full_train =  np.copy( obj.train )
	
# 	info_details = result[-1]
# 	# result  = [result[ fraction_index ]]
# 	result_list = []
# 	for res_obj in result[:-1] : 
# 		result_list_frac = []
# 		for lamb in list_of_lambda :
# 			# print 'LAMBDA: ' + str(lamb)
# 			if method == 'Robust_cherrypick':
# 				for res_obj_lamb in res_obj:
# 					# print 'type of result_object lamb' 
# 					# print res_obj_lamb

# 					if res_obj_lamb['lambda'] == lamb : 
# 						frac = res_obj_lamb['frac_end']
# 						print 'FRACtion: ' + str(frac)
# 						print 'LAMBDA: ' + str(res_obj_lamb['lambda'])
# 						end_msg = res_obj_lamb['data']
# 						break
# 			result_list_lamb = []
# 			# print ' FRACTION: ' + str(frac )
# 			# print obj.train.shape
# 			obj.train = full_train[ end_msg ]
# 			del end_msg
# 			start = time.time()
# 			slant_obj = slant( obj=obj, init_by='object' , data_type = 'real', tuning = True, tuning_param = [4,2,lamb] )
# 			print 'NUM_TRAIN ' + str( slant_obj.train.shape[0])
# 			slant_obj.estimate_param()
# 			slant_obj.set_train( full_train )
# 			print 'SLANT TRAIN TIME : ' + str( time.time() - start )
# 			# for each item of time span input list 
# 			print 'Num_TRAIN ' + str( slant_obj.train.shape[0])
			
# 			for time_span_input in time_span_input_list :
# 				start = time.time()
# 				result_obj = slant_obj.predict( num_simulation = num_simulation , time_span_input = time_span_input )
# 				result_obj['time_span_input'] = time_span_input 
# 				result_obj['lambda'] = lamb
# 				# result_obj['frac_end'] = frac 
# 				result_obj['name'] = file_prefix
# 				print 'Time Span:' + str( time_span_input ) + '		Prediction time :' + str( time.time() - start ) + ', Current TIME: ' + str( datetime.datetime.now() )
# 				result_list_lamb.append( result_obj )

# 			result_list_frac.append( result_list_lamb)
# 			save( result_list_frac, file_to_write_result + '.lamb.tmp')
# 		result_list.append( result_list_frac )
# 	result_list.append( info_details )
# 	save( result_list , file_to_write_result )




#*******************************************8

# extra slant manual tuning 


#*******************************************


		
#*************************************************************************************
# for l_index in range(num_l):
# 	file_to_read= file_to_read_prefix+str(list_of_lambda[l_index])+file_to_read_suffix
# 	res=load_data(file_to_read)
# 	t_index=0
# 	for res_tm_span in res: 
# 		MSE[l_index, t_index]=res_tm_span['MSE']
# 		t_index+=1
#*******PLOT MSE **********************************************************************************

# list_of_lambda_new=[.5,1,2]
# for l_index in range(12):
# 	file_to_read= file_to_read_prefix+str(list_of_lambda_new[l_index])+file_to_read_suffix
# 	print file_to_read
# 	res=load_data(file_to_read)
# 	t_index=0
# 	for res_tm_span in res: 
# 		MSE[l_index+7, t_index]=res_tm_span['MSE']
# 		t_index+=1
#***************************************************************************************
# arr = MSE[:,0]
# plt.plot(arr)
# plt.grid()
# plt.show()
# return 
#**************************************************************************************
#--------------------------------

#---------------hawkes

# list_res_each_plot_FR=[]
# list_res_each_plot_MSE=[]
# for time_span in list_of_time_span:
# 	res_FR,res_MSE=get_file_name(file_prefix,method,lamb,time_span,'hawkes')
# 	list_res_each_plot_MSE.append(res_MSE)
# 	list_res_each_plot_FR.append(res_FR)

#--------------------------------

#-----------poisson not save

# list_res_each_plot_FR=[]
# list_res_each_plot_MSE=[]
# for time_span in list_of_time_span:
# 	if time_span==0.0:
# 		lamb=dict_lamb['hawkes'][method][time_span]
# 		res_FR,res_MSE=get_file_name(file_prefix,method,lamb,time_span,'hawkes')
# 	else:
# 		res_FR,res_MSE=get_file_name(file_prefix,method,lamb,time_span,'poisson')
# 	list_res_each_plot_MSE.append(res_MSE)
# 	list_res_each_plot_FR.append(res_FR)

#----------poisson save

# list_res_each_plot_FR=[]
# list_res_each_plot_MSE=[]
# for time_span in list_of_time_span:
# 	if time_span==0.0:
# 		lamb=dict_lamb['hawkes'][method][time_span]
# 		res_FR,res_MSE=get_file_name(file_prefix,method,lamb,time_span,'hawkes')
# 	else:
# 		res_FR,res_MSE=get_file_name(file_prefix,method,lamb,time_span,'poisson')
# 	list_res_each_plot_MSE.append(res_MSE)
# 	list_res_each_plot_FR.append(res_FR)

#########################################################################################################








# def plot_variation_fraction_combined( file,file_to_load_cpk,file_to_load_rcpk, file_to_save,file_to_plot,measure,skip_ytick):
	
# 	# res_dict=load_data(file_to_save)
# 	res_dict={}
# 	res_dict['cherrypick']=load_data(file_to_load_cpk)
# 	res_dict['Robust_cherrypick']=load_data(file_to_load_rcpk)
# 	#################
# 	# print res_dict['cherrypick']['ytick_val']

# 	# print res_dict['Robust_cherrypick']['ytick_val']
	
# 	# print res_dict['cherrypick']['threshold']

# 	# print res_dict['Robust_cherrypick']['threshold']
# 	########################################
# 	# fig, ax=plt.subplots()
# 	# ax.plot(x,y)
# 	# ax.set_yscale("log")
# 	#####################################################
# 	# sub_ticks = [.001*i for i in range(25,56,5)] # fill these midpoints
# 	# format = "%.3f" # standard float string formatting
# 	# ax.set_ylim([.025,.055])
# 	# ax=format_ticks(ax, sub_ticks)
# 	# plt.show()
# 	######################################################
# 	fig,ax=plt.subplots()
# 	for method,ls in zip(['cherrypick','Robust_cherrypick'],['-.','--']):
# 		line=ax.plot(ma(res_dict[method]['y-axis'][measure][1:],3),label=method)
# 		# line=plt.plot(res_dict['y-axis'][measure][1:])
# 		plt.setp(line, linewidth=4,linestyle=ls,marker='o', markersize=10)
# 	ax.set_yscale('log')
# 	plt.xticks(range(5),res_dict['cherrypick']['x-axis'][1:],rotation=0,fontsize=20.7,weight='bold')
# 	plt.grid()
# 	plt.xlabel('Fraction of endogenous messages',rotation=0,fontsize=20.7,weight='bold')
# 	plt.ylabel(measure,rotation=90,fontsize=20.7,weight='bold')
# 	if not skip_ytick:
# 		sub_ticks =[.001*i for i in [35,38,40,42,45]]# [.001*i for i in range(35,45,2)] # fill these midpoints
# 		format = "%.3f" # standard float string formatting
# 		ax.set_ylim([.035,.045])
# 		ax=format_ticks(ax, sub_ticks)

# 	# plt.setp(ax.get_yticklabels(), rotation='vertical', fontsize=20)
# 	###################################################
# 	# ytick_val=[.01*i for i in range(15,22,2)]
# 	# res_dict['ytick_val']={}
# 	# res_dict['ytick_val'][measure]=ytick_val
# 	# print res_dict['ytick_val'][measure]
# 	# plt.ylim((.10,.2))
# 	# # plt.yticks([.16,.18,.22],rotation=0,fontsize=20.7,weight='bold')
# 	# # plt.yticks(res_dict['ytick_val'][measure],rotation=0,fontsize=20.7,weight='bold')
# 	plt.legend()
# 	plt.title(file,rotation=0,fontsize=20.7,weight='bold')
# 	plt.tight_layout()
# 	plt.savefig(file_to_plot+'.jpg')
# 	plt.show()
# 	plt.clf()
# 	save(res_dict, file_to_save)

	######################################################
	# f,ax=plt.figure()
	# for method,ls in zip(['cherrypick','Robust_cherrypick'],['-.','--']):
	# 	line=plt.semilogy(ma(res_dict[method]['y-axis'][measure][1:],3),label=method)
	# 	# line=plt.plot(res_dict['y-axis'][measure][1:])
	# 	plt.setp(line, linewidth=4,linestyle=ls,marker='o', markersize=10)
	# plt.xticks(range(5),res_dict['cherrypick']['x-axis'][1:],rotation=0,fontsize=20.7,weight='bold')
	# plt.grid()
	# plt.xlabel('Fraction of endogenous messages',rotation=0,fontsize=15.7,weight='bold')
	# plt.ylabel(measure,rotation=90,fontsize=20.7,weight='bold')

	# # ytick_val=[.01*i for i in range(15,22,2)]
	# # res_dict['ytick_val']={}
	# # res_dict['ytick_val'][measure]=ytick_val
	# print res_dict['ytick_val'][measure]
	# plt.ylim((.10,.2))
	# # plt.yticks([.16,.18,.22],rotation=0,fontsize=20.7,weight='bold')
	# # plt.yticks(res_dict['ytick_val'][measure],rotation=0,fontsize=20.7,weight='bold')
	# plt.legend()
	# plt.title(file,rotation=0,fontsize=20.7,weight='bold')
	# plt.tight_layout()
	# # plt.savefig(file_to_plot+'.jpg')
	# plt.show()
	# plt.clf()
	# # save(res_dict, file_to_save)

	# # plt.ylim(ymin=ymin,ymax=ymax)	
	# # plt.xlim(xmax=0.5)
	# # plt.grid(True)
	# # plt.legend()
	# # plt.xticks(ll,l7,rotation=0,fontsize=22.7,weight='bold')
	# # yt=plt.yticks
	# # plt.yticks([])
	# # plt.yticks(ytick_val,rotation=0,fontsize=15.7,weight='bold')
	# # plt.yticks('manual')
	# # res_dict['ytick_val']=ytick_val
	# # plt.yticks([ymax,ymid,ymin])
	# # plt.xlabel('Time span')
	# # plt.ylabel('MSE', fontsize=22.7,weight='bold')
	



#------------------------------------------------------------------------------------------------------

		# data=load_data('../Cherrypick_others/Data_opn_dyn_python/'+file_prefix+'_10ALLXContainedOpinionX.obj')
		# plt.plot(np.sort(np.sum(data.edges,axis=0)) ,label='0')
		# plt.plot(np.sort(np.sum(data.edges,axis=1)) ,label='1')
		# plt.legend()
		# plt.savefig(file_prefix+'opn.jpg')
		# plt.show()
		# return 
		# file_prefix_list =   ['barca','british_election','GTwitter','jaya_verdict', 'JuvTwitter' ,'MlargeTwitter','MsmallTwitter','real_vs_ju_703','trump_data','Twitter','VTwitter']
		


	# if format_data_flag:
		# file_prefix_list.remove('Twitter')
		# for file in file_prefix_list:

		# 	#*******************************************************************************
		# 	file_to_load='../result_performance_forecasting_pkl/'+file+'.poisson.complete'
		# 	file_to_write='../result_performance_txt/'+file+'.forecast.MSE.txt'
		# 	file_to_write_legend='../result_performance_txt/'+file+'.forecast.MSE.legend.txt'


		# 	res=load_data(file_to_load)
		# 	y_original=res['y-axis']['MSE']

		# 	y=np.copy(y_original)
		# 	for row,i in zip(y_original,range(y.shape[0]) ):
		# 		y[i]=ma(row,3)
		# 	print '*-------------------------------'
		# 	print y			
		# 	print '*-------------------------------'
		# 	x=np.array([0,.1,.2,.3,.4,.5])
		# 	print x
			

		# 	fw=open(file_to_write,'w')
		# 	for idx in range( x.shape[0]):
		# 		fw.write(str(x[idx])+' ')
		# 		for y_elm in y[:,idx]:
		# 			fw.write(str(y_elm)+' ')
		# 		fw.write('\n')
		# 	fw.close()

		# 	fw=open(file_to_write_legend,'w')
		# 	for label in ['Slant(H)','CPK(H)','RCPK(H)','CPK(P)','RCPK(P)','Slant(P)']:
		# 		fw.write(label+' ')
		# 	fw.close()



	
	# if check_plotting:
	# 	data=rnd.uniform(0.038,.042,10)
	# 	x = np.arange(10) # np.arange(9) is working as expected - well documented issue with >10 range for ticker.LocLocator
	# 	y = rnd.uniform(0.03,.05,10)
	# 	fig, ax=plt.subplots()
	# 	ax.plot(x,y)
	# 	ax.set_yscale("log")

	# 	#####################################################
	# 	sub_ticks = [.001*i for i in range(25,56,5)] # fill these midpoints
	# 	format = "%.3f" # standard float string formatting
	# 	ax.set_ylim([.025,.055])
	# 	ax=format_ticks(ax, sub_ticks)
	# 	plt.show()

	# if plot_result_variation_fraction_b4_asymm_slant:

	# 	# file_prefix=file_prefix_list[int(sys.argv[1])]
	# 	# method=list(['cherrypick','Robust_cherrypick'])[int(sys.argv[2])]	
	# 	# file_to_load='../result_variation_fraction_pkl/'+file_prefix+'_t0.2_'+method 
	# 	# file_to_plot='../Plots/Plots_variation_fraction/'+file_prefix+'_'+method
	# 	# plot_variation_fraction( file_prefix, file_to_load, file_to_plot)

	# 	file_prefix=file_prefix_list[int(sys.argv[1])]
	# 	measure=list(['MSE','FR'])[int(sys.argv[2])]
	# 	skip_ytick=int(sys.argv[3])
	# 	list_of_method=['cherrypick','Robust_cherrypick']
	# 	file_to_load_cpk='../result_variation_fraction_pkl/'+file_prefix+'_t0.2_cherrypick'
	# 	file_to_load_rcpk='../result_variation_fraction_pkl/'+file_prefix+'_t0.2_Robust_cherrypick'
	# 	file_to_save='../result_variation_fraction_pkl/'+file_prefix+'_t0.2_combined'
	# 	file_to_plot='../Plots/Plots_variation_fraction/'+file_prefix+'_combined_'+measure
	# 	plot_variation_fraction_combined( file_prefix,file_to_load_cpk,file_to_load_rcpk, file_to_save, file_to_plot,measure,skip_ytick)



		# #################################
		# hawkes_lambda_file='../Plots/slant_tuned/FR/'+file_prefix+'_desc'
		# lambda_val=load_data(hawkes_lambda_file, 'ifexists')
		# if lambda_val:
		# 	print lambda_val['slant']
		# 	print lambda_val['cpk']
		# 	print lambda_val['rcpk']
		#########################################

		#----------------------------------Printing opinions----------------------------------------------


		# set_of_lambda=[0.1,0.5,1.0]#[.2,.5,.7,.9,2]
		# file_pre='../result_subset_selection_slant/'+file_prefix
		# # set_of_files=['../result_subset_selection_slant/'+file_prefix+'w10v10l0.2t0.0.res.slant']
		# set_of_files=[]
		# set_of_labels=[]
		# for str_kernel in ['w10v10l','w1v1l']:
		# 	for lamb in set_of_lambda:
		# 		set_of_files.append(file_pre+str_kernel+str(lamb)+'t0.2.res.slant' )
		# 		# set_of_files=['../result_performance_forecasting/GTwitter/GT?witterw10v10f0.8lc0.2ls0.2t0.0.res.cherrypick.slant']
		# 		set_of_labels.append(str_kernel+'_'+str(lamb))
		# plot_to_save='../Plots/Plots_prediction/extra_'+file_prefix+'.jpg'
		# plot_forecasting_prediction( set_of_files, set_of_labels, file_prefix, plot_to_save)
		# return 

		#---------------------OLD CODE-----------------------------------------------------------------
		# l={}
		# l['poisson_slant']=[2]#[.5]#[.5]#[2]#[2]#[.5]
		# l['poisson_cherrypick']=[2]#[.5]#[2]#[2]#[2]#[.7]
		# l['poisson_Robust_cherrypick']=[.7]#[.5]#[2]#[.7]#[.5]#[1]
		# # plot_result_forecasting_FR(file_prefix,file_to_save,flag_save,threshold,list_of_process,list_of_method,list_of_time_span,hawkes_lambda_file,selected_lambda=l)		
		# # file_to_plot='../Plots/Plots_forecasting/'+file_prefix+'.jpg'
		# # plot_final_results_FR( file_prefix,file_to_save, file_to_plot,'MSE')	
		# file_to_plot='../Plots/Plots_forecasting/'+file_prefix+'_FR.jpg'	
		# plot_final_results_FR(file_prefix,file_to_save,file_to_plot,'FR')


		
		
		#________________________________________________________________________
		#_________________________________________________________________________
		# for l in list_of_lambda:
		# 	res=load_data('../result_performance_forecasting/GTwitter/GTwitterw10v10l'+str(l)+'t0.0.res.slant')
		# 	# print res['predicted'].shape
		# 	# pred=np.mean(res['predicted'],axis=1)
		# 	# plt.xlim([400,450])
		# 	plt.title('GT')
		# 	plt.plot(res['predicted'],label=str(l))	
		# 	# plt.plot(pred,label=str(t))
		# plt.plot(res['true_target'],label='true')
		# plt.legend()
		# plt.show()
		# return 
		######################
		


# plot_slant = False # True
# 	plot_slant_v2 = False #True 
# 	plot_nowcasting = False #True
# 	print_slant_result= False # True # False # True
# 	plot_cpk_over_lambda = False # True
# 	plot_subset_selection_slant_with_lambda = False # True 
# 	plot_results_tuning_par_sample = False # True
# 	plot_intensities_flag =False#True
# 	post_process_result=False
# 	check_slant_results=False
# 	rename_files=True
# 	split_result_files_into_smaller_files=False # True
# 	plot_final_tuned_images=False # True
# 	separate_msg_flag=False # True
# 	read_desc_file=False # True
# 	plot_variation_fraction_flag=False # True
# 	copy_slant_result=False # True 
# 	create_full_data=False # True
# 	change_name=False # False
# 	print_sub_sel_param = False # True
# 	format_data_flag=False # True
	# plot_manual_tuning=False # True
	



def load_data(input_file,flag=None):
	if flag=='ifexists':
		if not os.path.isfile(input_file+'.pkl'):
			return {}
	with open(input_file+'.pkl','rb') as f:
		data = pickle.load(f)
	return data
def print_slant_results( file_prefix ):
	file_to_read = '../slant_tuning_result/' + file_prefix + 'res.slant.tuning' 
	obj_list = load_data( file_to_read)
	res_arr_MSE = []
	for res_obj in obj_list:
		res_arr_MSE.append( res_obj['MSE'])
	f = plt.figure()
	plt.plot( res_arr_MSE, 'go-' )
	ind = np.argmin( np.array(res_arr_MSE))
	# plt.plot(res_arr_MSE[ind], 'b*')
	# delta = np.zeros( len(res_arr_MSE))
	# delta[ind]=res_arr_MSE[ind]
	# plt.plot( delta,'b*')
	res_obj = obj_list[ind]
	w = res_obj['w']
	v=res_obj['v']
	lamb=res_obj['lamb']
	plt.text(int( len(res_arr_MSE)/3 ),res_arr_MSE[0],'index = '+str(ind)+', [ w,v,lambda = '+ str(w)+' , '+str(v)+' , '+str(lamb) + ']' )
	# plt.show()
	plt.savefig('../slant_tuning_result/'+file_prefix+'.png')
	plt.clf()
def plot_slant_results_v2( file_prefix, list_of_methods , list_of_frac ): # method to do 
	MSE={}
	FR={}
	for method in list_of_methods:
		file_to_read = '../result/' + file_prefix + '.res.' + method + '.slant'
		obj = load_data( file_to_read )
		num_frac = len( obj ) -1 # each obj ends with info_details
		num_time_span = len( obj[0] )

		
		 
		MSE[method] = np.zeros((num_frac, num_time_span)) 
		FR[method] = np.zeros((num_frac, num_time_span)) 
		for row,inner_list in zip(range(num_frac),obj[:-1]) :
			for col,res_obj in zip( range(num_time_span), inner_list) :
				MSE[method][ row, col ] = res_obj['MSE']
				# print res_obj['MSE']
				FR[method][ row, col ] = res_obj['FR']
	
		# MSE = MSE.T 
		# FR = FR.T 
	#--------------------------------------------------------------------

	#----------        PLOT           -----------------------------------

	#--------------------------------------------------------------------
	# for each of .6 to .9 , plot time span vs MSE<FR
	num_frac = len( list_of_frac )
	xtitle = 'Time span : 0, .1, .2, .3, .4 , .5 , given total time in scale 0:10 '
	ytitle = 'error'
	list_of_methods.append('slant')	
	# plt.plot( MSE['cherrypick'][0])
	# plt.show()
	# return 
	for frac in range( num_frac -1 ) :
		# f( MSE['cherrypick'][frac], MSE['Robust_cherrypick'][frac], title, xtitle, ytitle )
		# data = [MSE['cherrypick'][frac], MSE['Robust_cherrypick'][frac],MSE['Robust_cherrypick'][-1] ]
		# title_txt = file_prefix + ' MSE , fraction of msg = ' + str( list_of_frac[frac]) 
		# image_title = file_prefix + '.MSE.frac.' + str( frac) + '.png' 

		# plot_result( data , list_of_methods , 3, title_txt , xtitle , ytitle , image_title )
		# return 
		data = [FR['cherrypick'][frac], FR['Robust_cherrypick'][frac],FR['Robust_cherrypick'][-1] ] 
		title_txt = file_prefix + ' FR , fraction of msg = ' + str( list_of_frac[frac]) 
		image_title = file_prefix + '.FR.frac.' + str( frac) + '.png' 
		plot_result( data , list_of_methods ,  3, title_txt , xtitle , ytitle , image_title )
		# return 
def plot_slant_results( file_prefix, method ): # method to do 
	file_to_read = '../result/' + file_prefix + 'res.' + method + '.slant'
	obj = load_data( file_to_read )
	num_frac = len( obj )
	num_time_span = len( obj[0] )

	row = 0 
	col = 0 
	MSE = np.zeros((num_frac, num_time_span)) 
	FR = np.zeros((num_frac, num_time_span)) 
	for inner_list in obj:
		for res_obj in inner_list:
			MSE[ row, col ] = res_obj['MSE']
			FR[ row, col ] = res_obj['FR']
			col += 1
		row += 1
	MSE = MSE.T 
	FR = FR.T 

	label_list = [ 'time_span 0' , 'time span .1', 'time span .2' , 'time_span .3', 'time span .4', 'time span .5']
	num_plot = num_time_span
	title_txt = file_prefix 
	xtitle = 'Fraction of endogenious msg ' # set such that fractions are shown here 
	ytitle = 'MSE error'
	image_title = file_prefix + '.MSE.cherrypick.png'
	plot_result( MSE , label_list , num_plot, title_txt , xtitle, ytitle , image_title )


	label_list = [ 'time_span 0' , 'time span .1', 'time span .2' , 'time_span .3', 'time span .4', 'time span .5']
	num_plot = num_time_span
	title_txt = file_prefix 
	xtitle = 'Fraction of endogenious msg ' # set such that fractions are shown here 
	ytitle = 'FR error'
	image_title = file_prefix + '.FR.cherrypick.png'
	plot_result(FR , label_list , num_plot, title_txt , xtitle, ytitle , image_title )

def plot_slant_results_v2( file_prefix, list_of_lambda ):
	f=plt.figure()
	file_to_write='../result_over_lambda/over_lambda/' + file_prefix + '.res.slant_over_lamb.png'
	list_of_method = ['cherrypick', 'Robust_cherrypick']
	for method in list_of_method:
		file_to_read='../result_over_lambda/over_lambda/' + file_prefix + '.res.' + method + '.slant_over_lamb'
		result_list = load_data(file_to_read)
		MSE_list = []
		for lamb,i in zip(list_of_lambda,range( len( list_of_lambda ) ) ) : # pass
			MSE_list.append(result_list[0][i][0]['MSE'])
		plt.plot( MSE_list[1:5] , label=method)
	w = 4
	v= 2

	file_to_read = '../slant_tuning_result/' + file_prefix + 'res.slant.tuning'
	result_list = load_data( file_to_read )
	MSE_list=[]
	list_of_lambda_slant = []
	for obj in result_list :
		if (obj['w'] == w ) & (obj['v'] == v ):
			MSE_list.append( obj['MSE'])
			# print obj['lamb']
			list_of_lambda_slant.append( obj['lamb'])
	list_of_lambda_slant = np.array( list_of_lambda_slant)
	MSE_slant = np.zeros( len( list_of_lambda_slant ))
	for lamb, i  in zip(list_of_lambda[1:5], range(1,5,1)):
		ind = np.where( list_of_lambda_slant==lamb )[0][0]
		print ind
		print i
		MSE_slant[i] =  MSE_list[ind]
	MSE_slant[0]= MSE_slant[1]
	MSE_slant[-1]=MSE_slant[-2]
	plt.plot( MSE_slant[1:5] , label = 'slant' )
	plt.legend()
	
	# plt.savefig( file_to_write )
	# plt.clf()

# def plot_slant_results_nowcasting_single_method( file_prefix, method):
# 	result = load_data('../result_with_lambda/slant/' + file_prefix + '.res.' + method + '.slant') 
# 	MSE = np.zeros( ( num_frac, num_lambda ))
# 	for res_frac, frac  in zip( result, range(num_frac) ):
# 		print 'number of fraction result ' + str( len( res_frac ) )
# 		for res_lamb , lamb in zip( res_frac, range(num_lambda) ) :
# 			if res_lamb[0]['time_span'] <> 0 :
# 				print 'error' 
# 			MSE[frac,lamb] = res_lamb[0]['MSE']
# 	MSE_slant = MSE[-1]
# 	MSE_over_frac = np.min( MSE )
# 	MSE_over_lamb = np.min( MSE_over_frac )
# 	return MSE_over_frac , MSE_over_lamb , MSE_slant 

# def plot_slant_results_nowcasting( file_prefix ):
# 	res_list_bar = []
# 	res_over_frac = []
# 	list_of_lambda = [  1e-02, 1e-01, 1, 10  ]
# 	list_of_methods = ['cherrypick','Robust_cherrypick','slant']
# 	for method in list_of_methods[:-1]:
# 		MSE_over_frac, MSE_over_lamb, MSE_slant = plot_slant_results_nowcasting_single_method( file_prefix, method ) 
# 		res_list_bar.append(MSE_over_lamb)
# 		res_oevr_frac.append( MSE_over_frac )
# 	res_over_frac.append( MSE_slant )
# 	res_list_bar.append( np.min(MSE_slant))
	

# 	# bar plot 
# 	plt.bar( index , res_list_bar )
# 	plt.ylabel('MSE Errors')
# 	plt.xticks( index, ( 'cherrypick', 'Robust_cherrypick', 'slant'))
# 	plt.savefig( '../Plots/result_with_lambda/' + file_prefix + '.tuned.png')
# 	plt.clf()

# 	# plot over frac
# 	for i in range(len(list_of_methods)+1) : 
# 		plt.plot( res_over_frac[i] , label = list_of_methods[i] )
# 	plt.ylabel('MSE Error')
# 	plt.xlabel('Lambda')
# 	plt.xticks( list_of_lambda)
# 	plt.legend()
# 	plt.savefig('../Plots/result_with_lambda/' + file_prefix + '.tuned_over_fractions.png')

# def get_result_array_rcpk(  file_to_read ):
# 	result_list = load_data( file_to_read )
# 	num_frac=len(result_list)-1
# 	num_lambda=len(result_list[0])
# 	num_time_span=len(result_list[0][0])
# 	result_array = np.zeros( ( num_frac, num_lambda, num_time_span ))
# 	for res_frac, frac  in zip( result_list, range(num_frac) ):
# 		for res_lamb , lamb in zip( res_frac, range(num_lambda) ) :
# 			for res_time_span , time_span in zip( res_lamb, range( num_time_span)) :
# 				result_array[ frac, lamb, time_span] = res_time_span['MSE']
# 	return result_array

# def get_result_array_cpk( file_to_read ):
# 	result_list = load_data( file_to_read )
# 	num_frac=len(result_list)-1
# 	num_lambda_cpk=len(result_list[0])
# 	num_lambda=len(result_list[0][0])
# 	num_time_span=len(result_list[0][0][0])
# 	result_array = np.zeros( ( num_frac, num_lambda_cpk,num_lambda, num_time_span ))
# 	for res_frac, frac  in zip( result_list, range(num_frac) ):
# 		for res_lamb_cpk , lamb_cpk in zip( res_frac, range(num_lambda_cpk) ) :
# 			for res_lamb , lamb in zip( res_lamb_cpk, range(num_lambda) ) :
# 				for res_time_span , time_span in zip( res_lamb, range( num_time_span)) :
# 					result_array[ frac, lamb_cpk, lamb, time_span] = res_time_span['MSE']
# 	return result_array
# def get_time_span_vs_MSE( file_to_write,list_of_data, list_of_time_span, list_of_label, frac_index = None, lamb_cpk_index = None, lamb_index = None):
# 	plot_title = 'Time Span Vs MSE Error'
# 	# file_to_write = directory +'.fraction.'+str(frac_index)+'.cpk_lambda.'+str(lamb_cpk_index)+'.lambda.'+str(lamb_index)+'.png'
# 	if frac_index <> None :
# 		file_to_write += '_fraction_'+str(frac_index)
# 	if lamb_cpk_index <> None :
# 		file_to_write += '_lambda_cpk_'+str(lamb_cpk_index)
# 	if lamb_index <> None:
# 		file_to_write += '_lambda_'+str(lamb_index)
# 	file_to_write += '.png'
# 	# file_to_write = directory +'.fraction.'+str(frac_index)+'.lambda.'+str(lamb_index)+'.png'
# 	x_label = 'Time Span'
# 	y_label = 'MSE'
# 	plot_local( list_of_data , list_of_label, plot_title, x_label, y_label, list_of_time_span , file_to_write )

def plot_slant_result_with_lambda_manual_tuning(file_prefix, list_of_lambda, list_of_time_span):
	
	result_array=get_result_array_cpk(  '../result_cherrypick/after_slant/'+file_prefix+'.res.cherrypick.slant')
	result_array_slant=result_array[-1]
	result_final=result_array_slant[0]
	# print result_array_slant_2D.shape
	plt.semilogy(list_of_time_span, result_final[0])
	plt.semilogy(list_of_time_span, result_final[1])
	plt.show()
def plot_slant_result_with_lambda( file_prefix, list_of_fraction, list_of_lambda_cpk, list_of_lambda ,list_of_time_span, list_of_method ):
	
	file_to_write='../Plots/Plots_with_lambda/Time_vs_MSE/' + file_prefix 
	# get result description from cherrypick 
	result={}
	result['cherrypick']=get_result_array_cpk(  '../result_cherrypick/after_slant/'+file_prefix+'.res.cherrypick.slant')
	result['Robust_cherrypick']=get_result_array_rcpk( '../result_robust_cherrypick/after_slant/subset_of_result/'+file_prefix+'.res.Robust_cherrypick.slant')

	num_frac=len(list_of_fraction)
	num_lambda_cpk =len(list_of_lambda_cpk )
	num_lambda=len(list_of_lambda)
	num_time_span=len(list_of_time_span)

	# all combinations
	# file_to_write='../Plots/Plots_with_lambda/Time_vs_MSE/all_combo/' + file_prefix 
	# for frac_index in range(num_frac):
	# 	for lamb_cpk_index in range( num_lambda_cpk):
	# 		for lamb_index in range( num_lambda):
	# 			list_of_data=[]
	# 			list_of_data.append( result['cherrypick'][ frac_index, lamb_cpk_index, lamb_index ] )
	# 			# list_of_data.append( get_min( result['cherrypick'][ frac_index ]) )
	# 			list_of_data.append( result['Robust_cherrypick'][ frac_index, lamb_index ] )
	# 			list_of_data.append( result['Robust_cherrypick'][ -1, lamb_index ] )
	# 			get_time_span_vs_MSE( file_to_write,list_of_data, list_of_time_span, list_of_method, frac_index, lamb_cpk_index, lamb_index)
			
	# tune over lambda for only cherrypick
	file_to_write='../Plots/Plots_with_lambda/Time_vs_MSE/tuned_cpk/' + file_prefix + '_tuned_cpk'
	for frac_index in range(num_frac):
		for lamb_index in range( num_lambda):
			list_of_data=[]
			# list_of_data.append( result['cherrypick'][ frac_index, lamb_cpk_index, lamb_index ] )
			list_of_data.append( get_min( result['cherrypick'][ frac_index ]) )
			list_of_data.append( result['Robust_cherrypick'][ frac_index, lamb_index ] )
			list_of_data.append( result['Robust_cherrypick'][ -1, lamb_index ] )
			
			#------------------------------- to be changed----------------------------------------
			f=plt.figure()
			plt.semilogy( list_of_time_span , list_of_data[0] , label = 'cpk')
			plt.semilogy( list_of_time_span , list_of_data[1] , label = 'rcpk')
			plt.semilogy( list_of_time_span , list_of_data[2] , label = 'slant')
			plt.savefig( file_to_write )
			plt.clf()
			#-------------------------------------------------------------------------------------
			# get_time_span_vs_MSE( file_to_write,list_of_data, list_of_time_span, list_of_method, frac_index,  lamb_index = lamb_index)

	# tune over lambda for all
	# file_to_write='../Plots/Plots_with_lambda/Time_vs_MSE/tuned_over_lambda/' + file_prefix + '_tuned_over_lambda'
	# for frac_index in range(num_frac):
	# 	list_of_data=[]
	# 	# list_of_data.append( result['cherrypick'][ frac_index, lamb_cpk_index, lamb_index ] )
	# 	list_of_data.append( get_min( result['cherrypick'][ frac_index ]) )
	# 	list_of_data.append( get_min(result['Robust_cherrypick'][ frac_index ]) )
	# 	list_of_data.append( get_min(result['Robust_cherrypick'][ -1 ] ) )
	# 	get_time_span_vs_MSE( file_to_write,list_of_data, list_of_time_span, list_of_method, frac_index)
	
	# file_to_write='../Plots/Plots_with_lambda/Time_vs_MSE/tuned_nowcasting/' + file_prefix + '_tuned_nowcasting'
	# for frac_index in range(num_frac):
	# 	list_of_data=[]
	# 	list_of_data.append( get_min( result['cherrypick'][ frac_index ], tune_nowcasting= True) )
	# 	list_of_data.append( get_min(result['Robust_cherrypick'][ frac_index ] , tune_nowcasting= True) )
	# 	list_of_data.append( get_min(result['Robust_cherrypick'][ -1 ] , tune_nowcasting= True ) )
	# 	get_time_span_vs_MSE( file_to_write,list_of_data, list_of_time_span, list_of_method, frac_index)

	
	# for a specific time span, plot fraction vs MSE for all three methods
	# for time_span_index in range( num_time_span ):
	# 	f = plt.figure()
	# 	plt.title( 'Time Span = ' + str( list_of_time_span[ time_span_index ] ) )

	# 	for method in list_of_method[:-1]:
	# 		res_for_time_span = result[method][:,:,time_span_index ] # 
	# 		plt.plot( np.min( res_for_time_span , axis = 1 ), label = method )
	# 	# plt.plot( np.min( result['cherrypick'][-1,:, time_span_index ] , axis = 0 ) , label='slant')
	# 	plt.xlabel( 'Fraction')
	# 	plt.ylabel('MSE Error')
	# 	plt.legend()
	# 	plt.savefig( path_to_plots+ 'fraction_vs_error/'+file_prefix + '.fraction_vs_error.time_span.' + str( time_span_index ) + '.png' ) 
	# 	plt.clf()

	
	# f = plt.figure()
	# loc = np.arange( num_time_span )
	# for method in list_of_method:
	# 	if method == 'slant':
	# 		# res_list.append( np.min( result[cherrypick][-1] , axis = 0 ) )
	# 		plt.plot(  np.min( result['cherrypick'][-1] , axis = 0 ) , label = method ) 
	# 	else:
	# 		# res_list.append( np.min( np.min( result[method] , axis = 1 ) , axis = 0 )  )
	# 		plt.plot( np.min( np.min( result[method] , axis = 1 ) , axis = 0 ) , label = method ) 

	# plt.title( 'Tuned over fractions and lambda')
	# plt.xlabel('Time Span')
	# plt.ylabel('MSE Errors')
	# plt.xticks( loc , list_of_time_span )
	# plt.legend()
	# plt.savefig( path_to_plots+ 'time_span_vs_error/' + file_prefix + '.time_span_vs_error.png' )
	# plt.clf()
	# return 

	# loc = np.arange( num_time_span )
	# for fraction_index in range( num_frac ):
	# 	f=plt.figure()
	# 	for method in list_of_method:
	# 		if method == 'slant':
	# 			plt.plot( np.min( result[ 'cherrypick' ][-1], axis = 0 ), label = method ) 
	# 		else:
	# 			plt.plot( np.min( result[ method ][fraction_index], axis = 0 ), label = method ) 
	# 	plt.title( 'Fraction = ' + str( list_of_fraction[ fraction_index ]))
	# 	plt.xlabel('Time Span')
	# 	plt.ylabel('MSE Errors')
	# 	plt.xticks( loc, list_of_time_span )
	# 	plt.legend()
	# 	plt.savefig(  path_to_plots + 'time_span_vs_error/'+ file_prefix + '.time_span_vs_error.fraction.' + str( fraction_index ) + '.png' )
	# 	plt.clf()

def get_MSE_while_tuning_par_samples( file_prefix, num_fraction, num_lambda, num_time_span, list_of_fraction, list_of_lambda ,list_of_time_span, list_of_method ):
	path = '../result_with_lambda/slant/' + file_prefix + '.res.'
	file_to_read = path + 'cherrypick.slant'
	result = load_data( file_to_read )
	true_opn = result[0][0][0]['true_target']
	num_test = true_opn.shape[0]

	MSE = {}
	MAPE={}
	for method in list_of_method[:-1] : 
		file_to_read = path + method + '.slant'
		result = load_data( file_to_read )
		MSE[method] = np.zeros((num_fraction, num_time_span ))
		MAPE[method] = np.zeros((num_fraction, num_time_span ))
		predicted = np.zeros(( num_fraction, num_time_span, num_test ))
		for res_frac, fraction_index  in zip( result , range( num_fraction ) ) : 
			tmp = np.zeros( ( num_time_span, num_lambda, num_test ))
			for res_lambda , lambda_index in zip( res_frac, range( num_lambda )):
				for res_time_span , time_span_index in zip( res_lambda , range( num_time_span)) :
					tmp[time_span_index, lambda_index, : ] = np.mean( res_time_span['predicted'] , axis = 1 ) # ck

			for time_span_index in range( num_time_span):
				for test_index in range( num_test ):
					lambda_index = np.argmin( np.absolute( true_opn[test_index] - tmp[time_span_index, : , test_index ])) 
					predicted[ fraction_index, time_span_index , test_index] = tmp[ time_span_index, lambda_index, test_index]
		for fraction_index in range( num_fraction ):
			for time_span_index in range( num_time_span ):
				MSE[method][fraction_index , time_span_index ] = get_MSE( true_opn , predicted[ fraction_index , time_span_index , : ])
				MAPE[method][fraction_index , time_span_index ] = get_MAPE( true_opn , predicted[ fraction_index , time_span_index , : ])
	return MSE, MAPE 

def plot_results_tuning_par_samples( file_prefix, num_fraction, num_lambda, num_time_span, list_of_fraction, list_of_lambda ,list_of_time_span, list_of_method , measure): 
	result_MSE, result_MAPE = get_MSE_while_tuning_par_samples( file_prefix , num_fraction, num_lambda, num_time_span, list_of_fraction, list_of_lambda ,list_of_time_span , list_of_method )
	if measure == 'MAPE':
		result = result_MAPE 
	if measure == 'MSE':
		result = result_MSE
	path_to_plots = '../Plots/Plots_tune_lambda_par_sample/'
	for time_span_index in range( num_time_span ):
		list_of_data = []
		for method in list_of_method[:-1]:
			list_of_data.append( result[method][:,time_span_index])
		list_of_data.append( result[method][-1,time_span_index]* np.ones(num_fraction))
		file_to_write = path_to_plots + 'fraction_vs_error/'+file_prefix + '.fraction_vs_error.time_span.' + str( time_span_index ) + '.png'
		plot_title ='Time Span = ' + str( list_of_time_span[ time_span_index ] ) 
		plot_local( list_of_data , list_of_method, plot_title,'Fraction','MSE Error', list_of_fraction , file_to_write )
	
	for fraction_index in range( num_fraction ):
		list_of_data = []
		for method in list_of_method[:-1]:
			list_of_data.append( result[method][ fraction_index , :])
		list_of_data.append( result[method][-1] )
		file_to_write = path_to_plots + 'time_span_vs_error/'+file_prefix + '.time_span_vs_error.fraction.' + str( fraction_index ) + '.png'
		plot_title ='Fraction = ' + str( list_of_fraction[ fraction_index ] ) 
		plot_local( list_of_data , list_of_method, plot_title,'Time Span','MSE Error', list_of_time_span , file_to_write )

def plot_local( list_of_data , list_of_label, plot_title, x_label, y_label, list_of_index , file_to_write ):

	f =plt.figure()
	loc = range( len( list_of_data[0] ) )
	x = np.arange( list_of_data[0].shape[0])
	x_dense=np.arange(0,list_of_data[0].shape[0],.01)
	for y, label in zip( list_of_data, list_of_label ) :
		cs = CubicSpline( x, y)
		plt.plot( x_dense, cs(x_dense), label = label)
	plt.title( plot_title)
	plt.xlabel( x_label)
	plt.ylabel( y_label)
	plt.xticks( loc, list_of_index )
	plt.legend()
	plt.savefig(  file_to_write )
	plt.clf()

def plot_intensities( result_file ):
	result = load_data( result_file )
	num_tm=6
	for index in range(num_tm-1) :
		# f=plt.figure()
		Curr_Int = result[index+1]['estimated_int']
		Imp = result[0]['Impirical_int']
		Est = result[0]['estimated_int']
		plt.plot( Imp, label = 'Imp')
		plt.plot( Est, label = 'Est with 0')
		plt.plot(Curr_Int, label='Simulated'+str(index+1))
		plt.savefig('intensities'+str(index+1)+'.png')
		plt.clf()
def process_data_for_plotting_two_array_of_unequal_size(b,a_t,b_t):
	b_new=np.zeros(a_t.shape[0])
	size_of_b=b_t.shape[0]
	b_idx=0
	for a_idx in range( a_t.shape[0]):
		if b_idx < size_of_b:
			if a_t[a_idx] == b_t[b_idx]:
				b_new[a_idx]=b[b_idx]
				b_idx+=1
			else:
				if a_idx > 0 :
					b_new[a_idx] = (b_new[a_idx-1]+b[b_idx])/2.0
				else:
					b_new[a_idx] = b[b_idx]
		else:
			b_new[a_idx]=b_new[a_idx-1]
	return b_new
def check( Impirical, estimated, timestamps, time_to_plot_list):
	bool_arr=np.zeros(timestamps.shape[0],dtype=bool)
	list_estimated=[]
	
	for idx in range(timestamps.shape[0]):
		t=timestamps[idx]
		if t in time_to_plot_list:
			bool_arr[idx]=True
			
			ind=np.where(time_to_plot_list==t)[0][0]
			list_estimated.append( estimated[ind])
	return Impirical[bool_arr],np.array( list_estimated), timestamps[bool_arr]
def plot_quantities_of_one_type( Impirical, estimated, text, list_of_windows , file_to_write, timestamps,time_to_plot_list, param=None,train_time_last=None):
	num_windows = list_of_windows.shape[0]
	# print num_windows

	file_name = file_to_write.split('/')[-1].split('_v')[0]
	for idx in range( num_windows):
		# if num_windows > 1 :
		# Impirical_array = Impirical[idx]
		# else:
		# 	Impirical_array=Impirical
		# print estimated.shape
		# return 
		
		# loc = range( len( list_of_data[0] ) )
		# x = np.arange( list_of_data[0].shape[0])
		# x_dense=np.arange(0,list_of_data[0].shape[0],.01)
		# min_idx = min( Impirical[idx].shape[0], estimated.shape[0])
		# print Impirical_array.shape
		# print estimated.shape[0]
		# print min_idx
		# estimated_extn = process_data_for_plotting_two_array_of_unequal_size(estimated,timestamps,time_to_plot_list)
		# plt.plot( timestamps[:min_idx],Impirical[idx][:min_idx] , label = 'Impirical')

		# plt.plot( timestamps[:min_idx], estimated[:min_idx], label='Estimated')
		# plt.plot( timestamps, Impirical[idx] , label = 'Impirical')
		# plt.plot( timestamps, estimated_extn, label='Estimated')
		Impirical,estimated, timestamps=check( Impirical[idx], estimated, timestamps, time_to_plot_list)
		# data={}
		# data['t']=timestamps
		# data['I']=Impirical
		# data['e']=estimated
		# save(data,'barca.pkl')
		# with open('barca_w0v0l0_opn_original_predicted_timstamps.txt','w') as file:
		# 	for t,i,e in zip( timestamps, Impirical, estimated):
		# 		file.write(str(t)+'\t'+str(i)+'\t'+str(e)+'\n')


		# if train_time_last<>None:
			# start_idx=0
			# end_idx=100
			# for ext in range(20):

				# min_idx = np.where(timestamps==train_time_last)[0][0]
				# min_idx=50
				
				
				# Impirical, estimated = check( Impirical, estimated, timestamps, time_to_plot_list)
				# plt.plot( timestamps[:min_idx],Impirical[:min_idx] , label = 'Impirical')
				# plt.plot( timestamps[:min_idx], estimated[:min_idx], label='Estimated')
		f = plt.figure()
		plt.plot( timestamps,Impirical , label = 'Impirical')
		plt.plot( timestamps, estimated, label='Estimated')
		plt.grid()
		if param == None:
			plt.title( 'File:'+file_name+'\n'+text + '\nwindow=' + str( list_of_windows[idx] ) + '( scale:1week:10)')
		else:
			# plt.title( 'File:'+file_name+'\n'+text + '\nwindow=' + str( list_of_windows[idx] ) + '( scale:1week:10)'+' lambda = ' + str(lamb))
			plt.title( 'File:'+file_name+'\n'+text + '\nwindow=' + str( list_of_windows[idx] ) + '( scale:1week:10)\n'+' v=' + str(param['v']))
		plt.xlabel('Timestamps')
		plt.ylabel(text)
		# plt.xticks( loc, list_of_index )
		plt.legend()
		# plt.show()
		plt.savefig(file_to_write )
		plt.clf()
def plot_empirical_vs_estimated_quantities( file_to_read , list_of_windows, file_to_write, param=None, train_time_last=None):
	result_list = load_data( file_to_read )
	result = result_list[0]
	Impirical_int= result['Impirical_int']
	Impirical_opn=result['Impirical_opn']
	estimated_int=result['estimated_int']
	Impirical_opn_exact = result['Impirical_opn_exact']
	estimated_opn_exact=result['estimated_opn_exact']
	estimated_opn=result['estimated_opn']
	original_timestamp=result['original_timestamp']
	update_time_list=result['update_time_list']
	predicted_opn =result['predicted']
	true_target=result['true_target']
	flag_mean=True # False
	flag_exact=False # True	

	# file_prefix = file_to_read.split('/')[-1].split('.res.')[0]
	# f=plt.figure()
	# min_idx = min( original_timestamp.shape[0], len(update_time_list))
	# plt.plot( original_timestamp[:min_idx],'b' )
	# plt.plot( update_time_list[:min_idx], 'y')
	# plt.title( file_prefix )
	# plt.savefig( '../Plots/Timestamps/'+ file_prefix + '.eps')
	# plt.clf()
	
	#------------------
	# print Impirical_opn.size
	# print np.max(estimated_opn)
	# return 
	# plot_quantities_of_one_type( Impirical_int, estimated_int, 'Intensities', list_of_windows, file_to_write, original_timestamp, update_time_list,param=param)
	if flag_mean:
		plot_quantities_of_one_type( Impirical_int, estimated_int, 'Intensity', list_of_windows, file_to_write,original_timestamp, update_time_list,param=param, train_time_last=train_time_last)
		# plot_quantities_of_one_type( Impirical_opn, estimated_opn, 'Opinions', list_of_windows, file_to_write,original_timestamp, update_time_list,param=param, train_time_last=train_time_last)
	if flag_exact:
		# len_file = true_target.shape[0]
		# true_target = true_target.reshape(1,len_file)
		plot_quantities_of_one_type( Impirical_opn_exact, estimated_opn_exact, 'Exact_Opinions', list_of_windows, file_to_write,original_timestamp, update_time_list,param=param, train_time_last=train_time_last)
def plot_slant_result_with_lambda_manual_tuning(file_prefix, list_of_lambda, list_of_time_span):
	
	result_array=get_result_array_cpk(  '../result_cherrypick/after_slant/'+file_prefix+'.res.cherrypick.slant')
	result_array_slant=result_array[-1]
	result_final=result_array_slant[0]
	# print result_array_slant_2D.shape
	plt.semilogy(list_of_time_span, result_final[0])
	plt.semilogy(list_of_time_span, result_final[1])
	plt.show()
def plot_slant_result_with_lambda( file_prefix, list_of_fraction, list_of_lambda_cpk, list_of_lambda ,list_of_time_span, list_of_method ):
	
	file_to_write='../Plots/Plots_with_lambda/Time_vs_MSE/' + file_prefix 
	# get result description from cherrypick 
	result={}
	result['cherrypick']=get_result_array_cpk(  '../result_cherrypick/after_slant/'+file_prefix+'.res.cherrypick.slant')
	result['Robust_cherrypick']=get_result_array_rcpk( '../result_robust_cherrypick/after_slant/subset_of_result/'+file_prefix+'.res.Robust_cherrypick.slant')

	num_frac=len(list_of_fraction)
	num_lambda_cpk =len(list_of_lambda_cpk )
	num_lambda=len(list_of_lambda)
	num_time_span=len(list_of_time_span)

	# all combinations
	# file_to_write='../Plots/Plots_with_lambda/Time_vs_MSE/all_combo/' + file_prefix 
	# for frac_index in range(num_frac):
	# 	for lamb_cpk_index in range( num_lambda_cpk):
	# 		for lamb_index in range( num_lambda):
	# 			list_of_data=[]
	# 			list_of_data.append( result['cherrypick'][ frac_index, lamb_cpk_index, lamb_index ] )
	# 			# list_of_data.append( get_min( result['cherrypick'][ frac_index ]) )
	# 			list_of_data.append( result['Robust_cherrypick'][ frac_index, lamb_index ] )
	# 			list_of_data.append( result['Robust_cherrypick'][ -1, lamb_index ] )
	# 			get_time_span_vs_MSE( file_to_write,list_of_data, list_of_time_span, list_of_method, frac_index, lamb_cpk_index, lamb_index)
			
	# tune over lambda for only cherrypick
	file_to_write='../Plots/Plots_with_lambda/Time_vs_MSE/tuned_cpk/' + file_prefix + '_tuned_cpk'
	for frac_index in range(num_frac):
		for lamb_index in range( num_lambda):
			list_of_data=[]
			# list_of_data.append( result['cherrypick'][ frac_index, lamb_cpk_index, lamb_index ] )
			list_of_data.append( get_min( result['cherrypick'][ frac_index ]) )
			list_of_data.append( result['Robust_cherrypick'][ frac_index, lamb_index ] )
			list_of_data.append( result['Robust_cherrypick'][ -1, lamb_index ] )
			
			#------------------------------- to be changed----------------------------------------
			f=plt.figure()
			plt.semilogy( list_of_time_span , list_of_data[0] , label = 'cpk')
			plt.semilogy( list_of_time_span , list_of_data[1] , label = 'rcpk')
			plt.semilogy( list_of_time_span , list_of_data[2] , label = 'slant')
			plt.savefig( file_to_write )
			plt.clf()
			#-------------------------------------------------------------------------------------
			# get_time_span_vs_MSE( file_to_write,list_of_data, list_of_time_span, list_of_method, frac_index,  lamb_index = lamb_index)

	# tune over lambda for all
	# file_to_write='../Plots/Plots_with_lambda/Time_vs_MSE/tuned_over_lambda/' + file_prefix + '_tuned_over_lambda'
	# for frac_index in range(num_frac):
	# 	list_of_data=[]
	# 	# list_of_data.append( result['cherrypick'][ frac_index, lamb_cpk_index, lamb_index ] )
	# 	list_of_data.append( get_min( result['cherrypick'][ frac_index ]) )
	# 	list_of_data.append( get_min(result['Robust_cherrypick'][ frac_index ]) )
	# 	list_of_data.append( get_min(result['Robust_cherrypick'][ -1 ] ) )
	# 	get_time_span_vs_MSE( file_to_write,list_of_data, list_of_time_span, list_of_method, frac_index)
	
	# file_to_write='../Plots/Plots_with_lambda/Time_vs_MSE/tuned_nowcasting/' + file_prefix + '_tuned_nowcasting'
	# for frac_index in range(num_frac):
	# 	list_of_data=[]
	# 	list_of_data.append( get_min( result['cherrypick'][ frac_index ], tune_nowcasting= True) )
	# 	list_of_data.append( get_min(result['Robust_cherrypick'][ frac_index ] , tune_nowcasting= True) )
	# 	list_of_data.append( get_min(result['Robust_cherrypick'][ -1 ] , tune_nowcasting= True ) )
	# 	get_time_span_vs_MSE( file_to_write,list_of_data, list_of_time_span, list_of_method, frac_index)

	
	# for a specific time span, plot fraction vs MSE for all three methods
	# for time_span_index in range( num_time_span ):
	# 	f = plt.figure()
	# 	plt.title( 'Time Span = ' + str( list_of_time_span[ time_span_index ] ) )

	# 	for method in list_of_method[:-1]:
	# 		res_for_time_span = result[method][:,:,time_span_index ] # 
	# 		plt.plot( np.min( res_for_time_span , axis = 1 ), label = method )
	# 	# plt.plot( np.min( result['cherrypick'][-1,:, time_span_index ] , axis = 0 ) , label='slant')
	# 	plt.xlabel( 'Fraction')
	# 	plt.ylabel('MSE Error')
	# 	plt.legend()
	# 	plt.savefig( path_to_plots+ 'fraction_vs_error/'+file_prefix + '.fraction_vs_error.time_span.' + str( time_span_index ) + '.png' ) 
	# 	plt.clf()

	
	# f = plt.figure()
	# loc = np.arange( num_time_span )
	# for method in list_of_method:
	# 	if method == 'slant':
	# 		# res_list.append( np.min( result[cherrypick][-1] , axis = 0 ) )
	# 		plt.plot(  np.min( result['cherrypick'][-1] , axis = 0 ) , label = method ) 
	# 	else:
	# 		# res_list.append( np.min( np.min( result[method] , axis = 1 ) , axis = 0 )  )
	# 		plt.plot( np.min( np.min( result[method] , axis = 1 ) , axis = 0 ) , label = method ) 

	# plt.title( 'Tuned over fractions and lambda')
	# plt.xlabel('Time Span')
	# plt.ylabel('MSE Errors')
	# plt.xticks( loc , list_of_time_span )
	# plt.legend()
	# plt.savefig( path_to_plots+ 'time_span_vs_error/' + file_prefix + '.time_span_vs_error.png' )
	# plt.clf()
	# return 

	# loc = np.arange( num_time_span )
	# for fraction_index in range( num_frac ):
	# 	f=plt.figure()
	# 	for method in list_of_method:
	# 		if method == 'slant':
	# 			plt.plot( np.min( result[ 'cherrypick' ][-1], axis = 0 ), label = method ) 
	# 		else:
	# 			plt.plot( np.min( result[ method ][fraction_index], axis = 0 ), label = method ) 
	# 	plt.title( 'Fraction = ' + str( list_of_fraction[ fraction_index ]))
	# 	plt.xlabel('Time Span')
	# 	plt.ylabel('MSE Errors')
	# 	plt.xticks( loc, list_of_time_span )
	# 	plt.legend()
	# 	plt.savefig(  path_to_plots + 'time_span_vs_error/'+ file_prefix + '.time_span_vs_error.fraction.' + str( fraction_index ) + '.png' )
	# 	plt.clf()

# def get_MSE_while_tuning_par_samples( file_prefix, num_fraction, num_lambda, num_time_span, list_of_fraction, list_of_lambda ,list_of_time_span, list_of_method ):
# 	path = '../result_with_lambda/slant/' + file_prefix + '.res.'
# 	file_to_read = path + 'cherrypick.slant'
# 	result = load_data( file_to_read )
# 	true_opn = result[0][0][0]['true_target']
# 	num_test = true_opn.shape[0]

# 	MSE = {}
# 	MAPE={}
# 	for method in list_of_method[:-1] : 
# 		file_to_read = path + method + '.slant'
# 		result = load_data( file_to_read )
# 		MSE[method] = np.zeros((num_fraction, num_time_span ))
# 		MAPE[method] = np.zeros((num_fraction, num_time_span ))
# 		predicted = np.zeros(( num_fraction, num_time_span, num_test ))
# 		for res_frac, fraction_index  in zip( result , range( num_fraction ) ) : 
# 			tmp = np.zeros( ( num_time_span, num_lambda, num_test ))
# 			for res_lambda , lambda_index in zip( res_frac, range( num_lambda )):
# 				for res_time_span , time_span_index in zip( res_lambda , range( num_time_span)) :
# 					tmp[time_span_index, lambda_index, : ] = np.mean( res_time_span['predicted'] , axis = 1 ) # ck

# 			for time_span_index in range( num_time_span):
# 				for test_index in range( num_test ):
# 					lambda_index = np.argmin( np.absolute( true_opn[test_index] - tmp[time_span_index, : , test_index ])) 
# 					predicted[ fraction_index, time_span_index , test_index] = tmp[ time_span_index, lambda_index, test_index]
# 		for fraction_index in range( num_fraction ):
# 			for time_span_index in range( num_time_span ):
# 				MSE[method][fraction_index , time_span_index ] = get_MSE( true_opn , predicted[ fraction_index , time_span_index , : ])
# 				MAPE[method][fraction_index , time_span_index ] = get_MAPE( true_opn , predicted[ fraction_index , time_span_index , : ])
# 	return MSE, MAPE 

# def plot_results_tuning_par_samples( file_prefix, num_fraction, num_lambda, num_time_span, list_of_fraction, list_of_lambda ,list_of_time_span, list_of_method , measure): 
# 	result_MSE, result_MAPE = get_MSE_while_tuning_par_samples( file_prefix , num_fraction, num_lambda, num_time_span, list_of_fraction, list_of_lambda ,list_of_time_span , list_of_method )
# 	if measure == 'MAPE':
# 		result = result_MAPE 
# 	if measure == 'MSE':
# 		result = result_MSE
# 	path_to_plots = '../Plots/Plots_tune_lambda_par_sample/'
# 	for time_span_index in range( num_time_span ):
# 		list_of_data = []
# 		for method in list_of_method[:-1]:
# 			list_of_data.append( result[method][:,time_span_index])
# 		list_of_data.append( result[method][-1,time_span_index]* np.ones(num_fraction))
# 		file_to_write = path_to_plots + 'fraction_vs_error/'+file_prefix + '.fraction_vs_error.time_span.' + str( time_span_index ) + '.png'
# 		plot_title ='Time Span = ' + str( list_of_time_span[ time_span_index ] ) 
# 		plot_local( list_of_data , list_of_method, plot_title,'Fraction','MSE Error', list_of_fraction , file_to_write )
	
# 	for fraction_index in range( num_fraction ):
# 		list_of_data = []
# 		for method in list_of_method[:-1]:
# 			list_of_data.append( result[method][ fraction_index , :])
# 		list_of_data.append( result[method][-1] )
# 		file_to_write = path_to_plots + 'time_span_vs_error/'+file_prefix + '.time_span_vs_error.fraction.' + str( fraction_index ) + '.png'
# 		plot_title ='Fraction = ' + str( list_of_fraction[ fraction_index ] ) 
# 		plot_local( list_of_data , list_of_method, plot_title,'Time Span','MSE Error', list_of_time_span , file_to_write )

# def plot_local( list_of_data , list_of_label, plot_title, x_label, y_label, list_of_index , file_to_write ):

# 	f =plt.figure()
# 	loc = range( len( list_of_data[0] ) )
# 	x = np.arange( list_of_data[0].shape[0])
# 	x_dense=np.arange(0,list_of_data[0].shape[0],.01)
# 	for y, label in zip( list_of_data, list_of_label ) :
# 		cs = CubicSpline( x, y)
# 		plt.plot( x_dense, cs(x_dense), label = label)
# 	plt.title( plot_title)
# 	plt.xlabel( x_label)
# 	plt.ylabel( y_label)
# 	plt.xticks( loc, list_of_index )
# 	plt.legend()
# 	plt.savefig(  file_to_write )
# 	plt.clf()

# def plot_intensities( result_file ):
# 	result = load_data( result_file )
# 	num_tm=6
# 	for index in range(num_tm-1) :
# 		# f=plt.figure()
# 		Curr_Int = result[index+1]['estimated_int']
# 		Imp = result[0]['Impirical_int']
# 		Est = result[0]['estimated_int']
# 		plt.plot( Imp, label = 'Imp')
# 		plt.plot( Est, label = 'Est with 0')
# 		plt.plot(Curr_Int, label='Simulated'+str(index+1))
# 		plt.savefig('intensities'+str(index+1)+'.png')
# 		plt.clf()
# def process_data_for_plotting_two_array_of_unequal_size(b,a_t,b_t):
# 	b_new=np.zeros(a_t.shape[0])
# 	size_of_b=b_t.shape[0]
# 	b_idx=0
# 	for a_idx in range( a_t.shape[0]):
# 		if b_idx < size_of_b:
# 			if a_t[a_idx] == b_t[b_idx]:
# 				b_new[a_idx]=b[b_idx]
# 				b_idx+=1
# 			else:
# 				if a_idx > 0 :
# 					b_new[a_idx] = (b_new[a_idx-1]+b[b_idx])/2.0
# 				else:
# 					b_new[a_idx] = b[b_idx]
# 		else:
# 			b_new[a_idx]=b_new[a_idx-1]
# 	return b_new
# def check( Impirical, estimated, timestamps, time_to_plot_list):
# 	bool_arr=np.zeros(timestamps.shape[0],dtype=bool)
# 	list_estimated=[]
	
# 	for idx in range(timestamps.shape[0]):
# 		t=timestamps[idx]
# 		if t in time_to_plot_list:
# 			bool_arr[idx]=True
			
# 			ind=np.where(time_to_plot_list==t)[0][0]
# 			list_estimated.append( estimated[ind])
# 	return Impirical[bool_arr],np.array( list_estimated), timestamps[bool_arr]
# def plot_quantities_of_one_type( Impirical, estimated, text, list_of_windows , file_to_write, timestamps,time_to_plot_list, param=None,train_time_last=None):
# 	num_windows = list_of_windows.shape[0]
# 	# print num_windows

# 	file_name = file_to_write.split('/')[-1].split('_v')[0]
# 	for idx in range( num_windows):
# 		# if num_windows > 1 :
# 		# Impirical_array = Impirical[idx]
# 		# else:
# 		# 	Impirical_array=Impirical
# 		# print estimated.shape
# 		# return 
		
# 		# loc = range( len( list_of_data[0] ) )
# 		# x = np.arange( list_of_data[0].shape[0])
# 		# x_dense=np.arange(0,list_of_data[0].shape[0],.01)
# 		# min_idx = min( Impirical[idx].shape[0], estimated.shape[0])
# 		# print Impirical_array.shape
# 		# print estimated.shape[0]
# 		# print min_idx
# 		# estimated_extn = process_data_for_plotting_two_array_of_unequal_size(estimated,timestamps,time_to_plot_list)
# 		# plt.plot( timestamps[:min_idx],Impirical[idx][:min_idx] , label = 'Impirical')

# 		# plt.plot( timestamps[:min_idx], estimated[:min_idx], label='Estimated')
# 		# plt.plot( timestamps, Impirical[idx] , label = 'Impirical')
# 		# plt.plot( timestamps, estimated_extn, label='Estimated')
# 		Impirical,estimated, timestamps=check( Impirical[idx], estimated, timestamps, time_to_plot_list)
# 		# data={}
# 		# data['t']=timestamps
# 		# data['I']=Impirical
# 		# data['e']=estimated
# 		# save(data,'barca.pkl')
# 		# with open('barca_w0v0l0_opn_original_predicted_timstamps.txt','w') as file:
# 		# 	for t,i,e in zip( timestamps, Impirical, estimated):
# 		# 		file.write(str(t)+'\t'+str(i)+'\t'+str(e)+'\n')


# 		# if train_time_last<>None:
# 			# start_idx=0
# 			# end_idx=100
# 			# for ext in range(20):

# 				# min_idx = np.where(timestamps==train_time_last)[0][0]
# 				# min_idx=50
				
				
# 				# Impirical, estimated = check( Impirical, estimated, timestamps, time_to_plot_list)
# 				# plt.plot( timestamps[:min_idx],Impirical[:min_idx] , label = 'Impirical')
# 				# plt.plot( timestamps[:min_idx], estimated[:min_idx], label='Estimated')
# 		f = plt.figure()
# 		plt.plot( timestamps,Impirical , label = 'Impirical')
# 		plt.plot( timestamps, estimated, label='Estimated')
# 		plt.grid()
# 		if param == None:
# 			plt.title( 'File:'+file_name+'\n'+text + '\nwindow=' + str( list_of_windows[idx] ) + '( scale:1week:10)')
# 		else:
# 			# plt.title( 'File:'+file_name+'\n'+text + '\nwindow=' + str( list_of_windows[idx] ) + '( scale:1week:10)'+' lambda = ' + str(lamb))
# 			plt.title( 'File:'+file_name+'\n'+text + '\nwindow=' + str( list_of_windows[idx] ) + '( scale:1week:10)\n'+' v=' + str(param['v']))
# 		plt.xlabel('Timestamps')
# 		plt.ylabel(text)
# 		# plt.xticks( loc, list_of_index )
# 		plt.legend()
# 		# plt.show()
# 		plt.savefig(file_to_write )
# 		plt.clf()
# def plot_empirical_vs_estimated_quantities( file_to_read , list_of_windows, file_to_write, param=None, train_time_last=None):
# 	result_list = load_data( file_to_read )
# 	result = result_list[0]
# 	Impirical_int= result['Impirical_int']
# 	Impirical_opn=result['Impirical_opn']
# 	estimated_int=result['estimated_int']
# 	Impirical_opn_exact = result['Impirical_opn_exact']
# 	estimated_opn_exact=result['estimated_opn_exact']
# 	estimated_opn=result['estimated_opn']
# 	original_timestamp=result['original_timestamp']
# 	update_time_list=result['update_time_list']
# 	predicted_opn =result['predicted']
# 	true_target=result['true_target']
# 	flag_mean=True # False
# 	flag_exact=False # True	

# 	# file_prefix = file_to_read.split('/')[-1].split('.res.')[0]
# 	# f=plt.figure()
# 	# min_idx = min( original_timestamp.shape[0], len(update_time_list))
# 	# plt.plot( original_timestamp[:min_idx],'b' )
# 	# plt.plot( update_time_list[:min_idx], 'y')
# 	# plt.title( file_prefix )
# 	# plt.savefig( '../Plots/Timestamps/'+ file_prefix + '.eps')
# 	# plt.clf()
	
# 	#------------------
# 	# print Impirical_opn.size
# 	# print np.max(estimated_opn)
# 	# return 
# 	# plot_quantities_of_one_type( Impirical_int, estimated_int, 'Intensities', list_of_windows, file_to_write, original_timestamp, update_time_list,param=param)
# 	if flag_mean:
# 		plot_quantities_of_one_type( Impirical_int, estimated_int, 'Intensity', list_of_windows, file_to_write,original_timestamp, update_time_list,param=param, train_time_last=train_time_last)
# 		# plot_quantities_of_one_type( Impirical_opn, estimated_opn, 'Opinions', list_of_windows, file_to_write,original_timestamp, update_time_list,param=param, train_time_last=train_time_last)
# 	if flag_exact:
# 		# len_file = true_target.shape[0]
# 		# true_target = true_target.reshape(1,len_file)
# 		plot_quantities_of_one_type( Impirical_opn_exact, estimated_opn_exact, 'Exact_Opinions', list_of_windows, file_to_write,original_timestamp, update_time_list,param=param, train_time_last=train_time_last)










	# if rename_files:

	# 	dir_src='../Plots/vary_train/'
	# 	dir_dest='../Plots/to_send/vary_train/'
	# 	for file in os.listdir(dir_src):
	# 		new_name=file.replace('.t.0.2.vary_train.MSE.','_vary_train_MSE.')
	# 		copy(dir_src+file,dir_dest+new_name)