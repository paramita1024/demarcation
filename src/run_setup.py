import getopt
import datetime
import time 
import numpy as np
import sys
import os 

def parse_command_line_input( ):

    argv = sys.argv[1:]
    opts, args = getopt.getopt(argv, 'o:f:b:', ['option', 'file', 'brain'])

    # if len(opts) == 0 and len(opts) > 3:
    #     print ('usage: add.py -a <first_operand> -b <second_operand> -c tmp ')
    # else:
    option = ''
    br=''
    filename=''
    ind=0
    b=''
    for opt, arg in opts:
        if opt == '-o':
            option = arg
            
        if opt == '-f':
            filename = arg

        if opt == '-b':
            for baseline in list_of_baselines:
                if baseline.startswith(arg):
                    b = baseline

        if opt == '-br':
            br=arg
        if opt == '-i':
            ind=int(arg)

    return option, br, filename ,ind,b

def copy_directory():
    for br in [5]:#,6,7,12,14,15] :# , 6,7, 12,14,15]:
        for ind in range(6):
            # shutil.copy('CM', 'CM_'+ str(br) + '_' + str(ind)  )
            os.system('cp -r CM CM_subset_'+ str(ind))#+ '_' + str(ind))
            # os.system('rm -r  CM_'+ str(br) + '_' + str(ind))

def remove_directory():
    for br in [5,6,7,12,14,15] :# , 6,7, 12,14,15]:
        for ind in range(9):
            # shutil.copy('CM', 'CM_'+ str(br) + '_' + str(ind)  )
            os.system('rm -r CM_'+ str(br) + '_' + str(ind))
            # os.system('rm -r  CM_'+ str(br) + '_' + str(ind))

# def run_slant( br , filename, list_of_baselines, list_of_regularizers):
    
#     ind = 0
#     for baseline in list_of_baselines:
#         for r in list_of_regularizers:
#             os.chdir('CM_'+ br + '_' + str(ind))        
#             ind+=1
#             cwd = os.getcwd() 
#             print("Current working directory is:", cwd) 
#             if True: # baseline == 'robust_lasso': 
#                 print 'Running ',baseline
#                 os.system('nohup python baselines_slant.py -f ' + filename + ' -b ' + baseline[0] + ' -r ' + str(r) + ' &')
#             os.chdir('../')
#             #if ind==6:
#                 #return 

def run_slant( filename, baseline, br, ind, list_of_regularizers):
    for r in list_of_regularizers:
        os.chdir('CM_'+ br + '_' + str(ind))        
        ind+=1
        cwd = os.getcwd() 
        print("Current working directory is:", cwd) 
        if True: # baseline == 'robust_lasso': 
            print 'Running ',baseline
            os.system('nohup python baselines_slant.py -f ' + filename + ' -b ' + baseline[0] + ' -r ' + str(r) + ' &')
        os.chdir('../')
        
def run_subset(list_of_file_prefix):
    ind =0 
    for filename in list_of_file_prefix:
        if filename[0] in ['G', 'j', 'J', 'M', 'T']:
            os.chdir('CM_subset_'+ str(ind) )
            ind +=1
            cwd = os.getcwd()
            os.system('nohup python baselines.py -f ' + filename + ' &' )
            os.chdir('../')


def main():
    #---------------------
    list_of_file_prefix = ['barca', 'british_election', 'GTwitter','jaya_verdict', \
        'JuvTwitter',  'MsmallTwitter', 'real_vs_ju_703', \
                'Twitter' , 'VTwitter']  
    list_of_baselines = ['huber_regression', 'robust_lasso',  'soft_thresholding'] # 'filtering', 'dense_err', 
    list_of_regularizers=[ 0.6,0.8,1.0,1.2]
    option, br, filename , ind, baseline= parse_command_line_input( list_of_baselines)
    if option == 'copy':
        copy_directory()
    if option == 'remove':
        remove_directory()
    if option == 'slant':
        run_slant(filename, baseline, br, ind, list_of_regularizers )
    if option == 'subset':
        run_subset(list_of_file_prefix)
    



if __name__=='__main__':
	main()


				
