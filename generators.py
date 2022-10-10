import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

class Generator:
    
    def __init__(self, name, carrier, p_nom, p_min_pu=0., p_max_pu=1.):
        self.name = name
        self.carrier = carrier
        self.p_nom = p_nom
        self.p_min_pu = p_min_pu
        self.p_max_pu = p_max_pu
        
    def set_p_min_pu(self, load):
        # Instead of one value of p_min_pu, we use a len_load size array of p_min_pu (one for each snapshot)
        if not isinstance(self.p_min_pu, (list, np.ndarray)):
            self.p_min_pu = float(self.p_min_pu)*np.ones((1,len(load)))[0]
        return

class GasGenerator(Generator):
    
    def __init__(self, name, carrier, p_nom, p_min_pu=0., p_max_pu=1., min_uptime=0., min_downtime=0., ramp_up_limit=1., ramp_down_limit=1., start_up_cost=0., shut_down_cost=0., efficiency_curve=None, fuel_price=1., constant_efficiency=False, co2_per_mw=0.517, SFC=0.215, inertia_constant=3.2):
        # Parameters related to the efficiency curve, fuel consumption and co2 emissions
        super().__init__(name, 'gas', p_nom, p_min_pu, p_max_pu)
        
        self.min_uptime = min_uptime
        self.min_downtime = min_downtime
        self.ramp_up_limit = ramp_up_limit
        self.ramp_down_limit = ramp_down_limit
        self.start_up_cost = start_up_cost
        self.shut_down_cost = shut_down_cost
        self.efficiency_curve = efficiency_curve
        self.fuel_price = fuel_price
        self.constant_efficiency = constant_efficiency
        self.co2_per_mw = co2_per_mw
        self.SFC = SFC
        self.inertia_constant = inertia_constant
    
    def set_p_max_pu(self, load):
        # Instead of one value of p_max_pu, we use a len_load size array of p_max_pu (one for each snapshot)
        if self.carrier == 'gas':
            if not isinstance(self.p_max_pu, (list, np.ndarray)):
                self.p_max_pu = float(self.p_max_pu)*np.ones((1,len(load)))[0]
        return
    
    def set_efficiency_curve_parameters(self, preprocessment=False):
        # Calculate efficiency curve parameters
        x = np.asarray(self.efficiency_curve.iloc[:,0]) / 100
        y = np.asarray(self.efficiency_curve.iloc[:,1]) / 100
        
        # Constant fitting for efficiency curve
        ef_k = np.mean(y)

        # Quadratic fitting for efficiency curve
        def fitting_function(x, a, b):
            return a*x**2 + b*x
        
        ef_params, _ = curve_fit(fitting_function, x, y)
        
        # Set generator efficiency curve parameters
        self.ef_a = ef_params[0]
        self.ef_b = ef_params[1]
        self.ef_k = ef_k
        return
    
    def initialize(self, load):
        # After network is set (with its load), calculate the parameters that depends on the load
        self.set_p_min_pu(load)
        self.set_p_max_pu(load)
        self.set_efficiency_curve_parameters()
        return

class WindGenerator(Generator):
    
    def __init__(self, name, carrier, p_nom, p_min_pu=0., p_max_pu=1., wind_speed_array=[], wind_penetration=None, electromechanical_conversion_efficiency=None, number_of_turbines=1):
        super().__init__(name, 'wind', p_nom, p_min_pu, p_max_pu)
        # Parameters related to wind and power conversion
        self.p_nom = p_nom * number_of_turbines
        self.wind_speed_array = np.asarray(wind_speed_array)
        self.wind_penetration = wind_penetration
        self.em_conversion_efficiency = electromechanical_conversion_efficiency
        self.number_of_turbines = number_of_turbines
        
        # Efficiency is set to 100%
        self.ef_a = 0
        self.ef_b = 1.

    def set_p_max_pu(self, load):       
        # Update p_max_pu according to available wind, given wind penetration and power curve
        # Read data from sheet
        power_curve = pd.read_excel('input_data/15MW_power_curve.xlsx')
        x_pc = np.asarray(power_curve['wind speed (m/s)'])
        y_pc = np.asarray(power_curve['mechanical power (MW)'])

        # Calculate p_max_pu for each snapshot
        mechanical_power_array = []
        for ws in self.wind_speed_array:
            # Find index of first wind speed from the power curve greater than the input wind speed
            curve_ws_idx = np.argmax(x_pc > ws)
            
            # Calculate the mechanical power using a linear fit between two points of the power curve and the given wind speed
            # From the power curve:
                # 1. Find the 2 wind speed around the input wind speed and their respective mechanical power
            x_interp = [x_pc[curve_ws_idx-1], x_pc[curve_ws_idx-1]]
            y_interp = [y_pc[curve_ws_idx-1], y_pc[curve_ws_idx-1]]
            
                # 2. Get the linear interpolation on the power curve to the given wind speed
            mp = np.interp(ws, x_interp, y_interp)
            
            # Append the resulting mechanical power to the mechanical power array
            mechanical_power_array.append(mp)
        
        # Total mechanical power = one turbine mechanical power * number of turbines
        mechanical_power_array = np.asarray(mechanical_power_array) * self.number_of_turbines
        
        # Apply the electromechanical conversion efficiency to mechanical power array
        electrical_power = self.em_conversion_efficiency * mechanical_power_array
        p_max_mw_power_curve = electrical_power
        
        # Save total power generator for future curtailment 
        self.generated_power = electrical_power
           
        # Calculate the p_max_pu given for the available wind and wind turbine power curve
        p_max_pu_power_curve = p_max_mw_power_curve/self.p_nom

        # Calculate the p_max_pu given for the wind penetration on the load
        p_max_pu_penetration = np.asarray(load) * self.wind_penetration/self.p_nom
          
        # The p_max_pu at each snapshot is the minimum value of the respective p_max from power curve and penetration
        p_max_pu = np.minimum(p_max_pu_power_curve, p_max_pu_penetration)
        self.p_max_pu = p_max_pu
        return
        
    def initialize(self, load):
        # After network is set (with its load), calculate the parameters that depends on the load
        self.set_p_min_pu(load)
        self.set_p_max_pu(load)
        return
    