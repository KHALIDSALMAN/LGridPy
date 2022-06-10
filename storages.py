import numpy as np
import pandas as pd

class Storage():
    
    def __init__(self, name, p_nom, nom_capacity_MWh, min_capacity, stand_efficiency, discharge_efficiency, initial_state_of_charge, charge_efficiency=1, final_state_of_charge=None, cyclic_state_of_charge=False):
        self.name = name
        self.p_nom = p_nom
        self.nom_capacity = nom_capacity_MWh
        self.min_capacity = min_capacity * nom_capacity_MWh
        self.stand_ef = stand_efficiency
        self.discharge_ef = discharge_efficiency
        self.charge_ef = charge_efficiency
        self.initial_soc = initial_state_of_charge * nom_capacity_MWh
        self.cyclic_soc = cyclic_state_of_charge
        
        
        # If storage is not cyclic (initial soc = last soc), these final soc is set to 'None'
        if cyclic_state_of_charge:
            self.final_soc = None
        else:
            self.final_soc = final_state_of_charge