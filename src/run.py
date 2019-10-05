
def parse_command_line_input( list_of_methods):

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, 'p:m:r:', ['path', 'method','reg1',])

    # if len(opts) == 0 and len(opts) > 3:
    #     print ('usage: add.py -a <first_operand> -b <second_operand> -c tmp  ')
    # else:
    file_path =''
    sel_method=''
    frac=1.0
    reg=0.1
    reg2=1.5
    int_gen='Hawkes'
    time_span = 0.2
    for opt, arg in opts:
        if opt == '-p':
            file_path = arg[:-4]

        if opt == '-m':
            for method in list_of_methods:
                if method.startswith(arg):
                    sel_method = method
        if opt == '-f':
            frac = float(arg)

        if opt == '-r':
            reg = float( arg )

        if opt == '-r2':
            reg2 = float( arg )

        if opt == '-i':
            int_gen = arg

        if opt == '-t':
            time_span=arg 


    return file_path, sel_method, frac, reg, reg2, int_gen, time_span


def get_subset( data, w, v, method, frac, reg, reg2 ):

	num_train = int(frac*data['all_user']['train'].shape[0])
	subset = np.zeros(self.num_train, dtype='bool')
    data_dict = { 'edges': data['edges'], 'nodes': data['nodes'],\
            'train': np.copy( data['all_user']['train'][  subset ,: ] ) , 'test': data['all_user']['test'] } 


	if method == 'robust_cherrypick':
		obj = Robust_Cherrypick( obj = data_dict , init_by = 'dict', lamb =reg , w_slant=w) 
		obj.initialize_data_structures()
		tmp1, subset, tmp2 = obj.robust_regression_via_hard_threshold( method = 'FC', max_itr = 50 , frac_end = frac) 
	
	if method == 'cherrypick':
		slant_obj = slant( init_by = 'dict', obj = data_dict, tuning_param = [ w, v, reg])
		param={'lambda':reg,'sigma_covariance':self.get_sigma_covar(reg)}
		obj = cherrypick( data = data_dict , init_by = 'dict',param=param,w=w) 
		obj.demarkate_process(frac_end=frac)
		subset=obj.save_end_msg(frac_end=frac)['data']
		
	if method == 'slant':
		subset = np.ones(self.num_train, dtype='bool')
	if baseline == 'huber_regression':
		subset[Huber_loss_minimization( data, [reg], [reg2] )[str(reg)][str(reg2)]['indices'][:num_train]]=True
	if baseline == 'robust_lasso':
		subset[Extended_robust_lasso( data, [reg], [reg2] )[str(reg)][str(reg2)]['indices'][:num_train] ]=True
	if  baseline == 'soft_thresholding':
		subset[Soft_thresholding( data, [reg], [reg2] )[str(reg)][str(reg2)]['indices'][:num_train] ]=True
	
	return subset
    

    
def get_slant_res(data, subset, w, v, reg, int_gen, time_span): 

    data_dict = { 'edges': data['edges'], 'nodes': data['nodes'],\
            'train': np.copy( data['all_user']['train'][  subset ,: ] ) , 'test': data['all_user']['test'] } 
    slant_obj = slant( init_by = 'dict', obj = data_dict, tuning_param = [ w, v, reg] , \
        int_generator= int_gen )
    slant_obj.estimate_param()
    slant_obj.set_train( data['all_user']['train'] )
    if time_span==0:
        return slant_obj.predict( num_simulation=1, time_span_input = time_span)
    else:
        return slant_obj.predict( num_simulation=20,time_span_input=time_span)
        

def print_results( file_prefix, res ):

    print file_prefix
    print '-'*50,'MSE','-'*50
    print res['MSE']
    print '-'*50,'FR','-'*50
    print res['FR']
    print '-'*150


def main():
    #---------------------
    list_of_methods = ['cherrypick','robust_cherrypick','slant','huber_regression', 'robust_lasso',  'soft_thresholding'] # 'filtering', 'dense_err', 
    file_path, method, frac, reg, reg2, int_gen,time_span = parse_command_line_input( list_of_methods )
    file_prefix = file_path.split('/')[-1]
    w= load_data('w_v')[file_prefix]['w']
    v= load_data('w_v')[file_prefix]['v']
    #-----------------------------
    data = load_data(file_path)
    subset=get_subset( data, w, v, method, frac, reg, reg2 )
    res = get_slant_res(data, subset, w, v, reg, int_gen, time_span)
    print_results( file_prefix, res )
    
if __name__=='__main__':
    main()


                
