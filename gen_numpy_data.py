import os
from tabulate import tabulate
import pandas as pd
import csv
import numpy as np
from six.moves import cPickle as pickle
from tkinter import *

from process_data import metadata_submetered

SAMPLES_FREQUENCY=30000
GRID_FREQUENCY=60
'''
Appliance= ['Air_Conditioner',
                   'Coffee_maker',
                     'Compact_Fluorescent_Lamp', 
                     'Fan',
                     'Fridge', 
                     'Hairdryer', 
                     'Heater', 
                     'Incandescent_Light_Bulb', 
                     'Laptop', 
                     'Microwave', 
                     'Vacuum', 
                     'Washing_Machine']
'''
plaid_path = "C:/YLX/NILM/Lenet/data/"
metadata_submetered_path = plaid_path +"metadata_submetered.json"
metadata_aggregated_path = plaid_path + "metadata_aggregated.json"
submetered_path = "C:/YLX/NILM/Lenet/data/submetered_new/"
aggregated_path = plaid_path + "aggregated/"
data_path = plaid_path
s_matrix_path = data_path + '/s_matrix/'
residue_path = data_path + '/residue/'
save_path = "data/"

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

            count+=1
            signal_dict[f"{appliance_type}_{app_n}_{file}"]={'appliance_type':appliance_type,'indices':None,'current':None,'current_rms':None,'voltage':None,'voltage_rms':None,'error_value':None}
            with open(submetered_file + file +'.csv') as csv_file:
                csv_reader = pd.read_csv(csv_file, header=None, names=(['current','voltage']))
                current=np.array(csv_reader['current'],dtype=np.float64)
                voltage=np.array(csv_reader['voltage'],dtype=np.float64)               

                    
                signal_dict[f"{appliance_type}_{app_n}_{file}"]['current']=current                
                signal_dict[f"{appliance_type}_{app_n}_{file}"]['voltage']=voltage                


                
    return signal_dict

def construct_harmonics_dict(signal_dict):
    
    harmonic_dict = {}
    
    for appliance_name, value in signal_dict.items():
        appliance_type = value.get('appliance_type')
        harmonic_dict[appliance_type] = {
            'appliance': {},
        }

    for appliance_name, value in signal_dict.items():
        appliance_type = value.get('appliance_type') 

        harmonic_dict[appliance_type]['appliance'][appliance_name] = {
        
            'current': [], 
            'voltage': [], 
        }

        current = value.get('current')
        voltage = value.get('voltage')

        #print   (third_harmonic_mag)
        if current is None:
            print("Current is None")
            continue

        #print("reactivate_power",reactivate_power)
        harmonic_dict[appliance_type]['appliance'][appliance_name]['current'] = current
        harmonic_dict[appliance_type]['appliance'][appliance_name]['voltage'] = voltage

    return harmonic_dict
    


def gen_numpy_data():
    available_options=np.arange(1,4)
    print("\nChoose an option below: \n\
            \n(1) Construct steady samples dictionary\
            \n(2) Construct harmonic dictionary from steady samples\
            \n(3) Generate V-I numpy data\
            \n(4) Exit\n")
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
        harmonic_dict = construct_harmonics_dict(signal_dict) 
        print(type(harmonic_dict))
        print("Saving harmonics dictionary...") 
        if not os.path.exists(save_path):
            os.makedirs(save_path)  
        with open(save_path + '/harmonic_dict.pkl', 'wb') as f: 
            pickle.dump(harmonic_dict, f, pickle.HIGHEST_PROTOCOL)       
        gen_numpy_data()
    if x==3:

        with open(save_path + "harmonic_dict.pkl", "rb") as f:
            harmonic_dict = pickle.load(f)      

        for appliance_type in harmonic_dict:          
            for appliance_name in harmonic_dict[appliance_type]['appliance']:
                current =  harmonic_dict[appliance_type]['appliance'][appliance_name]['current']
                voltage =  harmonic_dict[appliance_type]['appliance'][appliance_name]['voltage']
                
                if not os.path.exists(f"{save_path}/{appliance_type}"):
                    os.makedirs(f"{save_path}/{appliance_type}")                    
                df = pd.DataFrame({
                    'current': current,
                    'voltage': voltage,
                })
                df = df.round(2)
                df.to_csv(f"{save_path}/{appliance_type}/{appliance_name}.csv", sep=' ',header=False,index=False)

       
                
       


    gen_numpy_data()

        
    if x==4:
        exit()



gen_numpy_data()

