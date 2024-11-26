import os
from tabulate import tabulate
import pandas as pd
import csv
import numpy as np
import json
from six.moves import cPickle as pickle
from tkinter import *

from process_data import metadata_submetered

SAMPLES_FREQUENCY=30000
GRID_FREQUENCY=60
n_cycle = SAMPLES_FREQUENCY/GRID_FREQUENCY


plaid_path = "C:/YLX/NILM/Lenet/data/"
metadata_submetered_path = plaid_path +"metadata_submetered.json"
metadata_aggregated_path = plaid_path + "metadata_aggregated.json"
submetered_path = "C:/YLX/NILM/Lenet/data/submetered_new/"
aggregated_path = plaid_path + "aggregated/"
data_path = plaid_path
s_matrix_path = data_path + '/s_matrix/'
residue_path = data_path + '/residue/'
save_path = "data/"
save_path2 = "C:/YLX/NILM/nilm_with_plaid_data/data2"
save_path3 = "C:/YLX/NILM/nilm_with_plaid_data/data3"
save_path4 = "C:/YLX/NILM/nilm_with_plaid_data/data4"


# Load data from submetered metadata file
def metadata_submetered(metadata_file_submetered):
    f = open(metadata_file_submetered,'r')
    metadata = json.load(f)
    appliance_dict={}
    for file_number in metadata:
        appliance_dict[metadata[file_number]["appliance"]["type"]]=[]
    for file_number in metadata:
        if metadata[file_number]["appliance"]["type"] in appliance_dict:
            appliance_dict[metadata[file_number]["appliance"]["type"]].append(file_number)        
    f.close()
    
    return appliance_dict



def count_progress(number_of_files,count):
    if (count)%(int(np.ceil(number_of_files/1000)))==0:
        progress=np.round(count/number_of_files*100,decimals=1)
        print(f"Progress: {progress}%",end='\r')

# Calculate lag by number of samples
# Return index of array based on closest value
def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def lag_value_in_degrees(current,voltage,sample_frequency=30000,grid_frequency=60): 

    samples_per_cycle=int(sample_frequency/grid_frequency)
    i_cross_zero=find_nearest_index(current,0)
    v_cross_zero=find_nearest_index(voltage[i_cross_zero:i_cross_zero+int(samples_per_cycle/2)+1],0)+i_cross_zero
    if i_cross_zero-v_cross_zero<-samples_per_cycle/4:
        lag=-int(i_cross_zero+samples_per_cycle/2-v_cross_zero)*360/samples_per_cycle
    else:
        lag=-int(i_cross_zero-v_cross_zero)*360/samples_per_cycle
        
    return lag

def generate_rms(signal,mode=None,number_of_cycles=12,sample_frequency=30000,grid_frequency=60):
    n = len(signal)   
    samples_per_cycle=sample_frequency/grid_frequency
    duration=n/sample_frequency   
    time   = np.linspace(0,duration,n)
    signal_rms=np.array([])
    if mode=='half_cycle':
        resolution=samples_per_cycle/2
    elif mode=='full_cycle':
        resolution=samples_per_cycle
    else:
        resolution=number_of_cycles
    interv=np.arange(0,len(time),resolution)
    for i in interv:
        signal_pow=0                      
        if (i+resolution)<=(len(time)):
            signal_pow=[signal[j]**2 for j in range(int(i),int(i+(resolution)))]
            signal_pow=sum(signal_pow)
            i_rms=[np.sqrt(signal_pow/(resolution))]*int(resolution)
            signal_rms=np.concatenate((signal_rms, i_rms), axis=None)  
        else:
            signal_pow=[signal[j]**2 for j in range(int(i),int(len(time)-i))]
            signal_pow=sum(signal_pow)
            i_rms=[np.sqrt(signal_pow/(len(time)-i))]*int(len(time)-i)
            signal_rms=np.concatenate((signal_rms, i_rms), axis=None)  
    return signal_rms


def filter_harmonics(signal,highest_harmonic_order=None,sample_frequency=30000,grid_frequency=60):
    signal=np.array(signal,dtype=np.float32) 
    # Sample interval (s)
    dt=1/sample_frequency
    # Samples per cycle
    n_cycle=sample_frequency/grid_frequency
    # Number of samples
    n=len(signal)
    # Remainder samples of integer number of cycles
    remainder = n % n_cycle
    # Signal has to have integer number of cycles to do appropriate FFT 
    if remainder !=0:
        signal=signal[:-remainder]
        n = len(signal) 
    # Fast Fourier Transform (FFT) of signal
    fft_signal=np.fft.fft(signal,n)
    # Signal magnitude on frequency domain
    fft_signal_amp=np.abs(fft_signal)
    # Signal phase on frequency domain in radians
    fft_signal_phase=np.angle(fft_signal)
    # Frequency axes
    freq_axes = np.fft.fftfreq(n, d=dt)
    # Get only the odd harmonic frequencies
    harmonic_indices=[]
    if highest_harmonic_order==None:                   # If no odd order limit is given, gets indices 
        first_half=np.arange(-grid_frequency,min(freq_axes),-grid_frequency)   # through all range of frequencies, i.e. -fs/2 to fs/2                                     
        second_half=np.arange(grid_frequency,max(freq_axes),grid_frequency)    # in multiples of fn
        harmonic_indices=np.append(first_half,second_half,axis=0) 
    else:                                                               # Else, gets indices through range of -harmonic_order*fn
        first_half=np.arange(-grid_frequency,-grid_frequency*(highest_harmonic_order+1),-grid_frequency)    # to +harmonic_order*fn                                     
        second_half=np.arange(grid_frequency,grid_frequency*(highest_harmonic_order+1),grid_frequency)    
        harmonic_indices=np.append(first_half,second_half,axis=0)
    # Extract frequency indices around each harmonic order frequency
    ind_freq=[np.where((freq_axes >= (harmonic_indices[i]-grid_frequency/10)) & (freq_axes <= (harmonic_indices[i]+grid_frequency/10))) for i in range(len(harmonic_indices))]
    indices=[]
    for i in ind_freq:
        ind_max=np.where(fft_signal_amp == max(fft_signal_amp[i])) # get index where fft_signal is maximum
        ind=np.intersect1d(ind_max,i)    # discard rest of indices
        indices.append(ind)              # create list of indices selected
    # Create fft signal with only the harmonic values
    fft_signal_clean_amp=np.zeros(len(fft_signal_amp))
    fft_signal_clean_phase=np.zeros(len(fft_signal_phase))
    for i in indices:
        fft_signal_clean_amp[i]=fft_signal_amp[i]           # fft_signal clean = fft_signal at indices, 0 otherwise
        fft_signal_clean_phase[i]=fft_signal_phase[i]   

    return fft_signal_clean_amp,fft_signal_clean_phase # Returns magnitude and phase signal without noise in frequency domain

def reconstruct(signal_in_fft,highest_harmonic_order,length=None):
    dict_harmonics={}
    harmonic_pairs=[]
    harmonic_indices=np.where(abs(signal_in_fft)!=0)  # Gets all indices where signal in frequency domain isn't zero
    for i in range(len(harmonic_indices[0])//2):    # Construct list of indices pairs as tuples (negative and positive frequencies)
        harmonic_pairs.append((harmonic_indices[0][i],harmonic_indices[0][-i-1]))    
    mag_list=[]
    harmonic_number=1   # Starts with first harmonic
    for i,j in harmonic_pairs:  
        if harmonic_number<=highest_harmonic_order:
            harmonic_in_time_domain=np.zeros(len(signal_in_fft)).astype('complex128')  # Zero complex array in range of signal
            harmonic_in_time_domain[i]=signal_in_fft[i]          # First index of pair
            harmonic_in_time_domain[j]=signal_in_fft[j]          # Second index of pair 
            harmonic_in_time_domain=np.fft.ifft(harmonic_in_time_domain)        # Inverse Fourier Transform of pair
            mag_list.append(max(harmonic_in_time_domain.real))
            if length==None:
                dict_harmonics[harmonic_number]=harmonic_in_time_domain.real
            else:
                dict_harmonics[harmonic_number]=harmonic_in_time_domain.real[0:length]
            harmonic_number+=1          
    THD=sum(np.square(mag_list[1:]))**0.5/mag_list[0]           # Total Harmonic Distortion (THD) formula
    return dict_harmonics,THD           



def construct_harmonics_dict(signal_dict,highest_harmonic_order):
    
    # List of harmonic numbers
    harmonic_list = range(1,highest_harmonic_order+1)
    
    # Constructing harmonic_dict
    harmonic_dict={} 
    for appliance_name,value in signal_dict.items():   
        appliance_type = value.get('appliance_type')            
        harmonic_dict[appliance_type]={'appliance':{},'mean_lag':[],'mean_THD_current':[],'max_current':[],'first_harmonic_mag':[],'harmonics_proportions':{}} 
    
    n_appliances=len(signal_dict)
    
    count=0
    for appliance_name,value in signal_dict.items():        
        appliance_type = value.get('appliance_type')
        error_value = value.get('error_value') 
        harmonic_dict[appliance_type]['appliance'][appliance_name]={'error_value':error_value,'THD_current':None, 'THD_voltage':None,'harmonic_order':{}}
        count_progress(n_appliances,count)
        count+=1    
                   
        for harmonic_order in harmonic_list:
            harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][harmonic_order]={'current': [],'voltage':[]}                      
            harmonic_dict[appliance_type]['harmonics_proportions'][harmonic_order]=[]
        indices=value.get('indices')   
        current=value.get('current')
        voltage=value.get('voltage')
        voltage=voltage[indices[0]:indices[1]]
        current=current[indices[0]:indices[1]]

        current_fft_amp,current_fft_phase=filter_harmonics(current,highest_harmonic_order)
        current_fft=current_fft_amp*np.exp(current_fft_phase*1j)
        current_decomposed,THD_current=reconstruct(current_fft,21)
        harmonic_dict[appliance_type]['appliance'][appliance_name]['THD_current']=THD_current

        
        voltage_fft_amp,voltage_fft_phase=filter_harmonics(voltage,1)
        voltage_fft=voltage_fft_amp*np.exp(voltage_fft_phase*1j)
        voltage_decomposed,THD_voltage=reconstruct(voltage_fft,1)
        harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][1]['voltage']=(voltage_decomposed[1])
        harmonic_dict[appliance_type]['appliance'][appliance_name]['THD_voltage']=THD_voltage
        harmonic_dict[appliance_type]['first_harmonic_mag'].append(max(current_decomposed[1]))
        lag=lag_value_in_degrees(current,voltage)
        if harmonic_dict[appliance_type]['appliance'][appliance_name]['error_value']==0:
            harmonic_dict[appliance_type]['mean_lag'].append(lag)
            harmonic_dict[appliance_type]['mean_THD_current'].append(THD_current)
            harmonic_dict[appliance_type]['max_current'].append(max(current))

        for harmonic_order in current_decomposed:         
            harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][harmonic_order]['current']=(current_decomposed[harmonic_order])
            harmonic_dict[appliance_type]['harmonics_proportions'][harmonic_order].append(max(current_decomposed[harmonic_order])/max(current_decomposed[1]))           
           
    for appliance_type in harmonic_dict:
        harmonic_dict[appliance_type]['mean_lag']=int(np.mean(harmonic_dict[appliance_type]['mean_lag'])) 
        harmonic_dict[appliance_type]['mean_THD_current']=np.mean(harmonic_dict[appliance_type]['mean_THD_current'])
        harmonic_dict[appliance_type]['max_current']=max(harmonic_dict[appliance_type]['max_current'])
        for harmonic_order in harmonic_dict[appliance_type]['harmonics_proportions']:
            harmonic_dict[appliance_type]['harmonics_proportions'][harmonic_order]=np.mean(harmonic_dict[appliance_type]['harmonics_proportions'][harmonic_order])
    return harmonic_dict

def cal_lag(current,voltage):
# 假设这是你的瞬时电流和电压数据
# 请根据实际数据调整
    sample_frequency = SAMPLES_FREQUENCY
    # 进行快速傅立叶变换 (FFT)
    voltage_fft = np.fft.fft(voltage)
    current_fft = np.fft.fft(current)

    # 获取频率
    dt = 1/sample_frequency
    N = len(current)
    # 快速傅里叶变换
    
    freq = np.fft.fftfreq(N, d=dt)

    # 找到最大幅值对应的频率
    voltage_peak_idx = np.argmax(np.abs(voltage_fft))
    current_peak_idx = np.argmax(np.abs(current_fft))

    # 计算相位
    voltage_phase = np.angle(voltage_fft[voltage_peak_idx])
    current_phase = np.angle(current_fft[current_peak_idx])

    # 计算相位差
    phase_difference = current_phase - voltage_phase

    # 转换为度
    phase_difference_degrees = np.degrees(phase_difference)

    # 输出结果
    print(f'相位角（单位：度）: {phase_difference_degrees:.2f}°')

    return phase_difference_degrees




def get_indices(signal_rms,mode=None,sample_cycles=None,aggregated=0,sample_frequency=30000,grid_frequency=60):
    sample_dict={}
    sample_frag=[]
    n = len(signal_rms) 
    samples_per_cycle=sample_frequency/grid_frequency
    if mode=='half_cycle':
        resolution=samples_per_cycle/2
    else:
        resolution=samples_per_cycle
   
    #med_rms=np.mean(signal_rms)
    if sample_cycles==None:
        
        sample_cycles=12
        for k in range(int(n/(resolution)-sample_cycles*samples_per_cycle/resolution)+1):
            inf=int(k*resolution)
            sup=int(inf+sample_cycles*samples_per_cycle)        
            med=np.mean(signal_rms[inf:sup])     
            sample_dict[(inf,sup)]=np.var(signal_rms[inf:sup]/med)
        indices=min(sample_dict, key=sample_dict.get)
        indices=list(indices)
        return indices
    elif aggregated==0:                
        for k in range(int(n/(resolution)-sample_cycles*samples_per_cycle/resolution)+1):
            inf=int(k*resolution)
            sup=int(inf+sample_cycles*samples_per_cycle)        
            med=np.mean(signal_rms[inf:sup])                   
            flag=0            
            if med>0.03:
                if all(signal_rms[inf:sup]>0.9*med) and all(signal_rms[inf:sup]<1.1*med):
                    sample_dict[(inf,sup)]=np.var(signal_rms[inf:sup]/med)
        if sample_dict!={}:            
            indices=min(sample_dict, key=sample_dict.get)
            indices=list(indices)           
            return indices
        else:
            return None
    else:
        k=0
        while k<=int(n/(n_cycle/2)-sample_cycles):
            inf=int(k*n_cycle/2)
            sup=int(inf+n_cycle*sample_cycles)        
            med=np.mean(signal_rms[inf:sup])                   
            flag=0          
            if med>0.01:
                for j in range(2*sample_cycles):   
                    med_local=(signal_rms[inf+(j-1)*(int(n_cycle/2))]+signal_rms[inf+j*(int(n_cycle/2))])/2                         
                    if med_local>1.01*med or med_local<0.99*med:
                        flag=1
                        break
                if flag==0:
                    if sample_frag!=[]:
                        if sample_frag[-1][1]==inf:
                            sample_frag[-1][1]=sup
                        else:
                            sample_frag.append([inf,sup])
                    else:
                        sample_frag.append([inf,sup])
                    k+=2*sample_cycles-1
            k+=1
        print(sample_frag)
        return sample_frag
    


    
def steady_samples_submetered(submetered_file,data_dict):
    n_files=0
    for i in data_dict:
        n_files+=len(data_dict[i])
    signal_dict={}
    count=0
    for appliance_type in data_dict:             
        app_n=0
        for file in data_dict[appliance_type]:
            appliance_type=appliance_type.replace(" ","_")  
            app_n+=1
            count_progress(n_files,count)
            count+=1
            signal_dict[f"{appliance_type}_{app_n}_{file}"]={'appliance_type':appliance_type,'indices':None,'current':None,'current_rms':None,'voltage':None,'voltage_rms':None,'error_value':None}
            with open(submetered_file + file +'.csv') as csv_file:
                csv_reader = pd.read_csv(csv_file, header=None, names=(['current','voltage']))
                current=np.array(csv_reader['current'],dtype=np.float64)
                voltage=np.array(csv_reader['voltage'],dtype=np.float64)               
                sample_cycles=12
                error_image=0
                current_rms=generate_rms(current,mode='full_cycle')
                indices=get_indices(current_rms,mode='full_cycle',sample_cycles=sample_cycles)               
                if indices==None:                    
                    error_image=1
                    indices=get_indices(current_rms,None)                       
                signal_dict[f"{appliance_type}_{app_n}_{file}"]['current']=current                
                signal_dict[f"{appliance_type}_{app_n}_{file}"]['voltage']=voltage                
                signal_dict[f"{appliance_type}_{app_n}_{file}"]['indices']=indices
                signal_dict[f"{appliance_type}_{app_n}_{file}"]['error_value']=error_image
                
    return signal_dict

# Force lag between signals (lag in degrees)
def shift_phase(current,voltage,lag=0,sample_frequency=30000,grid_frequency=60): 
    samples_per_cycle=int(sample_frequency/grid_frequency)
    phase=int(lag_value_in_degrees(current,voltage)*samples_per_cycle/360)
    
    phase+=lag
      
    if int(phase)>0:
        current=current[int(phase):]
        voltage=voltage[:-int(phase)]        
    elif int(phase)<0:
        current=current[:int(phase)]
        voltage=voltage[int(abs(phase)):]  
    return current,voltage,phase


def harmonics_selection(harmonic_dict,highest_odd_harmonic_order,appliance_type,appliance_name,lag=None,odd=0):
    voltage=harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][1]['voltage']
    current=np.zeros(len(harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][1]['current']))

    if odd==0:
        for i in range(1,highest_odd_harmonic_order+1,1):
            harmonic=harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][i]['current']
            current+=harmonic
    else:
        for i in range(1,highest_odd_harmonic_order+1,2):
            harmonic=harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][i]['current']
            current+=harmonic
    if lag==1:
        current,voltage,phase=shift_phase(current,voltage,harmonic_dict[appliance_type]['mean_lag'])
    if lag==0:
        current,voltage,phase=shift_phase(current,voltage,lag)

    return current,voltage

def save_data(harmonic_dict,save_path,appliance_type,appliance_name,current,voltage):
    for appliance_type in harmonic_dict:          
            for appliance_name in harmonic_dict[appliance_type]['appliance']:      
                if not os.path.exists(f"{save_path}/{appliance_type}"):
                    os.makedirs(f"{save_path}/{appliance_type}")                    
                df2 = pd.DataFrame({
                    'current': current,
                    'voltage': voltage,   })
                df2 = df2.round(2)
                df2.to_csv(f"{save_path}/{appliance_type}/{appliance_name}.csv", sep=' ',header=False,index=False)
                
    return



def gen_numpy_data():
    available_options=np.arange(1,6)
    print("\nChoose an option below: \n\
            \n(1) Construct steady samples dictionary\
            \n(2) Construct harmonic dictionary from steady samples\
            \n(3) Generate V-I numpy data\
            \n(4) Generate VI-PIG numpy data\
            \n(5) Exit\n")

    while True:
        x = int(input("Option: "))
        if x in available_options:
            break
        print("Invalid option.")
    if x==1:
        # Get steady samples of current from submetered data and save in dictionary whose keys are appliances names
        metadata_dict=metadata_submetered(metadata_submetered_path)
        steady_samples_dict=steady_samples_submetered(submetered_path,metadata_dict)
        print("Saving dictionary...")
        if not os.path.exists(save_path):
            os.makedirs(save_path)  
        with open(save_path + '/steady_samples_dict.pkl', 'wb') as f: 
            pickle.dump(steady_samples_dict, f, pickle.HIGHEST_PROTOCOL)           
        gen_numpy_data()
    if x==2:
        # Construct harmonic dictionary from steady samples dictionary of submetered data 
        print("Loading steady samples dictionary...")
        with open(save_path + "steady_samples_dict.pkl", "rb") as f:
            signal_dict = pickle.load(f)        
            
        print("Constructing harmonics dictionary...")           
        harmonic_dict = construct_harmonics_dict(signal_dict,21) 
        print(type(harmonic_dict))
        print("Saving harmonics dictionary...") 
        if not os.path.exists(save_path):
            os.makedirs(save_path)  
        with open(save_path + '/harmonic_dict.pkl', 'wb') as f: 
            pickle.dump(harmonic_dict, f, pickle.HIGHEST_PROTOCOL)       
        gen_numpy_data()
    '''
    if x==3:

        with open(save_path + "harmonic_dict.pkl", "rb") as f:
            harmonic_dict = pickle.load(f)      
        
        for appliance_type in harmonic_dict:    
            mean_lag = harmonic_dict[appliance_type]['mean_lag']      
            print('meanlag = ',mean_lag)
            for appliance_name in harmonic_dict[appliance_type]['appliance']:

                voltage = harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][1]['voltage']
                
                current = harmonic_dict[appliance_type]['appliance'][appliance_name]['harmonic_order'][1]['current']
                
                if not os.path.exists(f"{save_path}/{appliance_type}"):
                    os.makedirs(f"{save_path}/{appliance_type}")                    
                df = pd.DataFrame({
                    'current': current,
                    'voltage': voltage,
                })
                df = df.round(2)
                df.to_csv(f"{save_path}/{appliance_type}/{appliance_name}.csv", sep=' ',header=False,index=False)

    gen_numpy_data()
'''
    if x==3:

        with open(save_path + "harmonic_dict.pkl", "rb") as f:
            harmonic_dict = pickle.load(f)      
        

        max_current=[]
        low_THD_appliance_type=[]

        for appliance_type in harmonic_dict:
            if harmonic_dict[appliance_type]['mean_THD_current']<0.05:
                low_THD_appliance_type.append(appliance_type)
                max_current.append(harmonic_dict[appliance_type]['max_current'])
        
        for appliance_type in harmonic_dict:          
            for appliance_name in harmonic_dict[appliance_type]['appliance']:                                    
               
                current,voltage=harmonics_selection(harmonic_dict,21,appliance_type,appliance_name,lag=0,odd=0)
                
                save_data(harmonic_dict,save_path,appliance_type,appliance_name,current,voltage)   

                current,voltage=harmonics_selection(harmonic_dict,21,appliance_type,appliance_name,lag=0,odd= 1)
                save_data(harmonic_dict,save_path2,appliance_type,appliance_name,current,voltage)

                current,voltage=harmonics_selection(harmonic_dict,21,appliance_type,appliance_name,lag=1,odd=0)
                save_data(harmonic_dict,save_path3,appliance_type,appliance_name,current,voltage)   

                current,voltage=harmonics_selection(harmonic_dict,21,appliance_type,appliance_name,lag=1,odd= 1)
                save_data(harmonic_dict,save_path4,appliance_type,appliance_name,current,voltage)

        print(f"V-I trajectories saved in '{save_path2}'\n")
       

    gen_numpy_data()

        
        
    if x==4:
        exit()



gen_numpy_data()

