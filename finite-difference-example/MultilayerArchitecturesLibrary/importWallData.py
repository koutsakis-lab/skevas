# -*- coding: utf-8 -*-
def multilayerPropertiesArchitecture(iCoatCase):
    import numpy as np
    import pandas as pd

    folder_dir=r'./MultilayerArchitecturesLibrary/'
    fname     =r'Multilayer Architecture - Case '+str(iCoatCase)+'.xlsx'
    
    df=pd.read_excel(folder_dir+fname,sheet_name='Sheet1',header=None\
                            ,skiprows=[0,1],usecols='A,B,C,D,E')
    df.columns=['Name','ThermalConductivity','Density','SpecificHeatCapacity',
                'Thickness']
    
    return df