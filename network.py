from pyomo.environ import *
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import copy
import os
import time
from openpyxl.utils import get_column_letter
from pyomo.util.infeasible import log_infeasible_constraints

from generators import *
from storages import *

class Network:

    def __init__(self, name='Unknown'):
        self.name = name
        self.is_solved = False
        self.gas_generators = []
        self.wind_generators = []
        self.generators = []
        self.storages = []

    def add_load(self, load):
        self.load = load
        self.snapshots = np.asarray(range(len(load)))
        return
    
    def step_load(self, nominal_load=100, number_of_steps=1):
        # Return step load
        step = np.array([0., 1., 1.])
        step_load = np.concatenate(([0, 0, 0], step))
    
        for i in range(1, number_of_steps):
            step_load = np.concatenate((step_load, step+i))
            
        step_load = nominal_load * np.concatenate((step_load, [step[-1]+number_of_steps-1]))
        return list(step_load)

    def add_gas_generator(self, name, p_nom, p_min_pu, p_max_pu, min_uptime, min_downtime, ramp_up_limit, ramp_down_limit, start_up_cost, shut_down_cost, efficiency_curve, fuel_price, constant_efficiency=False, co2_per_mw=0.517, SFC=0.215, inertia_constant=3.2, **kwargs):
        # Create GasGenerator object with input parameters
        gas_generator = GasGenerator(name, 'gas', p_nom, p_min_pu, p_max_pu, min_uptime, min_downtime, ramp_up_limit, ramp_down_limit, start_up_cost, shut_down_cost, efficiency_curve=efficiency_curve, fuel_price=fuel_price, constant_efficiency=constant_efficiency, co2_per_mw=co2_per_mw, SFC=SFC, inertia_constant=inertia_constant)
        
        # Set gas generator according to the load
        gas_generator.initialize(self.load)
        
        # Include gas generator in network
        self.gas_generators.append(gas_generator)
        self.generators.append(gas_generator)
        return

    def add_wind_generator(self, name, p_nom, wind_speed_array, wind_penetration, electromechanical_conversion_efficiency, number_of_turbines):
        # Create WindGenerator object with input parameters
        wind_generator = WindGenerator(name, 'wind', p_nom, wind_speed_array=wind_speed_array, wind_penetration=wind_penetration, electromechanical_conversion_efficiency=electromechanical_conversion_efficiency, number_of_turbines=number_of_turbines)
        
        # Set wind generator according to the load
        wind_generator.initialize(self.load)
        
        # Include wind generator in network
        self.wind_generators.append(wind_generator)
        self.generators.append(wind_generator)
        return
    
    def add_storage(self, name, p_nom, nom_capacity_MWh, min_capacity, stand_efficiency, discharge_efficiency, initial_state_of_charge, charge_efficiency=1, final_state_of_charge=None, cyclic_state_of_charge=False):
        # Create Storage object with input parameters
        storage = Storage(name, p_nom, nom_capacity_MWh, min_capacity, stand_efficiency, discharge_efficiency, initial_state_of_charge, charge_efficiency=charge_efficiency, final_state_of_charge=final_state_of_charge, cyclic_state_of_charge=cyclic_state_of_charge)
        
        # Include storage in network
        self.storages.append(storage)
        return
    
    def plot_efficiency_curve(self, gas_gen_name):
        # Plot efficiency curve of selected gas generator and its fitting parameters
        gas_gen = [x for x in self.gas_generators if x.name == gas_gen_name][0]
        
        figure = plt.figure()
        p = np.linspace(0, 1, num=100)
        
        ef_a = gas_gen.ef_a
        ef_b = gas_gen.ef_b
        ef_k = gas_gen.ef_k
        
        if gas_gen.constant_efficiency:
            efficiency = ef_k * np.ones(100)
        else:
            efficiency = ef_a * p**2 + ef_b * p
        
        plt.plot(100*p, 100*efficiency, label='Fitting')
        
        ef_data_x = gas_gen.efficiency_curve.iloc[:,0].values/100
        ef_data_y = gas_gen.efficiency_curve.iloc[:,1].values/100
        
        plt.plot(100*ef_data_x, 100*ef_data_y, 'o', label='Data')
        
        p_min_pu = np.mean(gas_gen.p_min_pu)
        p_max_pu = np.mean(gas_gen.p_max_pu)
        
        plt.axvline(x=100*p_min_pu, color='gray', ls='--')
        plt.axvline(x=100*p_max_pu, color='gray', ls='--', label='Dispatch bounds')
        
        if gas_gen.constant_efficiency:
            max_error = np.max(np.abs(ef_k - ef_data_y))*100
            legend_text = f'\nef(p) = {ef_k:.4}\n\nError <= {ceil(max_error)}%'
        else:
            max_error = np.max(np.abs(ef_a*ef_data_x**2 + ef_b*ef_data_x - ef_data_y))*100
            legend_text = f'\nef(p) = {ef_a:.4} * p² + {ef_b:.4} * p\n\nError <= {ceil(max_error)}%'
            
        
        plt.plot([], [], ' ', label=legend_text)
        
        plt.grid()
        plt.title(gas_gen_name + ' efficiency curve')
        plt.xlabel('Dispatch [% Nominal Power]')
        plt.ylabel('Efficiency [%]')
        plt.legend(loc='best')
        plt.show()
    
    def set_optimization_model(self):
        # Initialize Pyomo model
        self.model = ConcreteModel(name = self.name)
        return
    
    def get_generators_attr(self, attribute, carrier=None):
        # Get given attribute from generators
        # 1. Get array of generators names
        if attribute == 'name':
            generators_names = []
            if carrier == 'gas':
                generators_names = [gen.name for gen in self.gas_generators]
            elif carrier == 'wind':
                generators_names = [gen.name for gen in self.wind_generators]
            else:
                generators_names = [gen.name for gen in self.generators]
            return generators_names

        # 2. Get array of generators indexes
        elif attribute == 'index':
            number_of_generators = 0
            if carrier == 'gas':
                number_of_generators = len(self.gas_generators)
            elif carrier == 'wind':
                number_of_generators = len(self.wind_generators)
            else:
                number_of_generators = len(self.generators)
            generators_indexes = range(number_of_generators)
            return generators_indexes
        return
    
    def get_storages_attr(self, attribute):
        # Get given attribute from storages
        # 1. Get array of storages names
        if attribute == 'name':
            storages_names = [storage.name for storage in self.storages]
            return storages_names

        # 2. Get array of storages indexes
        elif attribute == 'index':
            number_of_storages = len(self.storages)
            storages_indexes = range(number_of_storages)
            return storages_indexes
        return
    
    def set_optimization_variables(self):
        # Get generators variables first dimension = generators names
        gen_first_dimension = self.get_generators_attr('name')
        gas_gen_first_dimension = self.get_generators_attr('name', 'gas')

        # Set upper bound for all generators dispatches
        gen_ub = 0
        for gen in self.generators:
            p_max = max(gen.p_max_pu) * gen.p_nom
            gen_ub = max(gen_ub, p_max)
        gen_ub *= 2
        
        # Set upper bound for all start up and shut down costs
        sud_ub = 0
        for gen in self.gas_generators:
            su = gen.start_up_cost
            sd = gen.shut_down_cost
            sud_ub = max(sud_ub, su, sd)
        sud_ub *= 2

        # Set Pyomo generators variables
        self.model.generator_p = Var(gen_first_dimension, self.snapshots, within=Reals, bounds=(0, gen_ub))
        self.model.generator_status = Var(gen_first_dimension, self.snapshots, within=Binary)
        self.model.generator_start_up_cost = Var(gas_gen_first_dimension, self.snapshots, within=Reals, bounds=(0, sud_ub))
        self.model.generator_shut_down_cost = Var(gas_gen_first_dimension, self.snapshots, within=Reals, bounds=(0, sud_ub))
        
        # Get generators variables first dimension = generators names
        storage_first_dimension = self.get_storages_attr('name')
        
        # Set upper bound for all storages dispatches
        st_ub = 0
        for st in self.storages:
            st_ub = max(st_ub, st.p_nom)
        st_ub *= 2
        
        # Define rule for state of charge
        def storage_soc_rule(model, st, t):
            st_index = storage_first_dimension.index(st)
            storage_capacity = self.storages[st_index].nom_capacity
            max_capacity = storage_capacity
            min_capacity = self.storages[st_index].min_capacity
            return (min_capacity, max_capacity)
            
        # Set Pyomo storage variables
        self.model.storage_charge = Var(storage_first_dimension, self.snapshots, within=Reals, bounds=(0, st_ub))
        self.model.storage_discharge = Var(storage_first_dimension, self.snapshots, within=Reals, bounds=(0, st_ub))
        self.model.storage_soc = Var(storage_first_dimension, self.snapshots, within=Reals, bounds=storage_soc_rule)
        self.model.charge_status = Var(storage_first_dimension, self.snapshots, within=Binary)
        self.model.discharge_status = Var(storage_first_dimension, self.snapshots, within=Binary)
        
        # Set Pyomo fuel cost variable
        self.model.fuel_cost = Var(gas_gen_first_dimension, self.snapshots, within=Reals, bounds=(0, 1e16))
        
        return
    
    def set_gen_p_min_constraint(self):
        
        # Set rule for p_min constraint
        def p_min_rule(model, i, j):
            # Get index of generator named 'i'
            index = first_dimension.index(i)
            
            # Get p_min in MW
            p_min = self.generators[index].p_min_pu[j] * self.generators[index].p_nom
            
            # Set constraint equation
            return -model.generator_p[i,j] + p_min * model.generator_status[i,j] <= 0
        
        # Get model first dimension
        first_dimension = self.get_generators_attr('name')
        
        # Add constraint to model
        self.model.p_min = Constraint(first_dimension, self.snapshots, rule=p_min_rule)
        return
    
    def set_gen_p_max_constraint(self):
        
        # Set rule for p_max constraint
        def p_max_rule(model, i, j):
            # Get index of generator named 'i'
            index = first_dimension.index(i)
            
            # Get p_max in MW
            p_max = self.generators[index].p_max_pu[j] * self.generators[index].p_nom
            
            # Fix p_max = 0 to a low value
            if p_max == 0:
                p_max = 1e-8
            
            # Set constraint equation
            return model.generator_p[i,j] - p_max * model.generator_status[i,j] <= 0
        
        # Get model first dimension
        first_dimension = self.get_generators_attr('name')
        
        # Add constraint to model
        self.model.p_max = Constraint(first_dimension, self.snapshots, rule=p_max_rule)
        return
    
    def set_gen_ramp_up_constraint(self):
        
        # Set rule for ramp_up constraint
        def ramp_up_rule(model, i, j):
            # Get index of generator named 'i'
            index = first_dimension.index(i)
            
            # Get ramp_up_limit in MW
            rul = self.generators[index].ramp_up_limit * self.generators[index].p_nom
            
            # Set constraint equation
            return model.generator_p[i,j] - model.generator_p[i,j-1] <= rul * model.generator_status[i,j]
        
        # Get model first dimension
        first_dimension = self.get_generators_attr('name', carrier='gas')
        
        # Add constraint to model
        self.model.ramp_up_limit = Constraint(first_dimension, self.snapshots[1:], rule=ramp_up_rule)
        return
        
    def set_gen_ramp_down_constraint(self):
        
        # Set rule for ramp_down constraint
        def ramp_down_rule(model, i, j):
            # Get index of generator named 'i'
            index = first_dimension.index(i)
            
            # Get ramp_down_limit in MW
            rdl = self.generators[index].ramp_down_limit * self.generators[index].p_nom
            
            # Set constraint equation
            return -model.generator_p[i,j] + model.generator_p[i,j-1] <= rdl * model.generator_status[i,j-1]
        
        # Get model first dimension
        first_dimension = self.get_generators_attr('name', carrier='gas')
        
        # Add constraint to model
        self.model.ramp_down_limit = Constraint(first_dimension, self.snapshots[1:], rule=ramp_down_rule)
        return
    
    
    def set_gen_min_uptime_constraint(self):
        
        # Set rule for min_uptime constraint
        def min_uptime_rule(model, i, j):
            # Get index of generator named 'i'
            index = first_dimension.index(i)
            
            # Get min_uptime (notice that it is treated as an INTEGER value)
            mut = floor(self.generators[index].min_uptime)
            
            # If the min_uptime < 1, this constraint is skipped
            if mut == 0:
                return Constraint.Skip
            
            # Treat the upper limit of the sum (if the snapshot is close to the end, the upper limit is the remaining snapshots)
            sum_upper_limit = min(j+mut, len(self.snapshots)-1)
            
            # Activate the generator on the first snapshot
            if j == 0:
                return model.generator_status[i,j] == 1
            
            # Set constraint equation
            return -sum(model.generator_status[i,k] for k in range(j, sum_upper_limit)) + mut * (model.generator_status[i,j] - model.generator_status[i,j-1]) <= 0
        
        # Get model first dimension
        first_dimension = self.get_generators_attr('name', carrier='gas')
        
        # Add constraint to model (except for the last snapshot)
        self.model.min_uptime = Constraint(first_dimension, self.snapshots[:-1], rule=min_uptime_rule)
        return
    
    def set_gen_min_downtime_constraint(self):
        
        # Set rule for min_downtime constraint
        def min_downtime_rule(model, i, j):
            # Get index of generator named 'i'
            index = first_dimension.index(i)
            
            # Get min_downtime (notice that it is treated as an INTEGER value)
            mdt = floor(self.generators[index].min_downtime)
            
            # If the min_downtime < 1, this constraint is skipped
            if mdt == 0:
                return Constraint.Skip
            
            # Treat the upper limit of the sum (if the snapshot is close to the end, the upper limit is the remaining snapshots)
            sum_upper_limit = min(j+mdt, len(self.snapshots)-1)
            
            # Special equation for the first snapshot (generator_status at snapshot '-1' = 1)
            if j == 0:
                return sum(model.generator_status[i,k] for k in range(j, sum_upper_limit)) - mdt * (model.generator_status[i,j]) <= 0
            
            # Set constraint equation
            return sum(model.generator_status[i,k] for k in range(j, sum_upper_limit)) - mdt * (model.generator_status[i,j] - model.generator_status[i,j-1] + 1) <= 0
        
        # Get model first dimension
        first_dimension = self.get_generators_attr('name', carrier='gas')
        
        # Add constraint to model (except for the last snapshot)
        self.model.min_downtime = Constraint(first_dimension, self.snapshots[:-1], rule=min_downtime_rule)
        return
    
    def set_gen_start_up_cost_constraint(self):
        # Set rule for start_up constraint
        def start_up_cost_rule(model, i, j): 
            # Get index of generator named 'i'
            index = first_dimension.index(i)

            # Get start up cost in dollars
            su = self.generators[index].start_up_cost
            
            # Special equation for the first snapshot (generator_status at snapshot '-1' = 1)
            if j == 0:
                return -model.generator_start_up_cost[i,j] + su * model.generator_status[i,j] - su <= 0
            
            # Set constraint equation
            return -model.generator_start_up_cost[i,j] + su * (model.generator_status[i,j] - model.generator_status[i,j-1]) <= 0
        
        # Get model first dimension
        first_dimension = self.get_generators_attr('name', carrier='gas')
        
        # Add constraint to model 
        self.model.start_up_cost = Constraint(first_dimension, self.snapshots, rule=start_up_cost_rule)
        return
    
    def set_gen_shut_down_cost_constraint(self):
        # Set rule for shut_down constraint
        def shut_down_cost_rule(model, i, j):
            # Get index of generator named 'i'
            index = first_dimension.index(i)
            
            # Get start up cost in dollars
            sd = self.generators[index].shut_down_cost
            
            # Special equation for the first snapshot (generator_status at snapshot '-1' = 1)
            if j == 0:
                return -model.generator_shut_down_cost[i,j] - sd * model.generator_status[i,j] + sd <= 0
            
            # Set constraint equation
            return -model.generator_shut_down_cost[i,j] - sd * (model.generator_status[i,j] - model.generator_status[i,j-1]) <= 0
        
        # Get model first dimension
        first_dimension = self.get_generators_attr('name', carrier='gas')
        
        # Add constraint to model 
        self.model.shut_down_cost = Constraint(first_dimension, self.snapshots, rule=shut_down_cost_rule)
        return
    
    def set_storage_max_ch(self):
        # Set rule for maximum dispatch
        def max_ch_rule(model, i, j):
            # Get index of storage named 'i'
            index = first_dimension.index(i)

            # Get storage nominal power
            storage_p_nom = self.storages[index].p_nom
            
            # Return equation for maximum dispatch
            return model.storage_charge[i,j] <= model.charge_status[i,j] * storage_p_nom
        
        # Get model first dimension
        first_dimension = self.get_storages_attr('name')
        
        # Add constraint to model
        self.model.max_charge = Constraint(first_dimension, self.snapshots, rule=max_ch_rule)
        return
    
    def set_storage_max_disch(self):
        # Set rule for maximum dispatch
        def max_disch_rule(model, i, j):
            # Get index of storage named 'i'
            index = first_dimension.index(i)

            # Get storage nominal power
            storage_p_nom = self.storages[index].p_nom
            
            # Return equation for maximum dispatch
            return model.storage_discharge[i,j] <= model.discharge_status[i,j] * storage_p_nom
        
        # Get model first dimension
        first_dimension = self.get_storages_attr('name')
        
        # Add constraint to model
        self.model.max_discharge = Constraint(first_dimension, self.snapshots, rule=max_disch_rule)
        return
    
    def set_storage_soc_constraint(self):
        # Set rule for state_of_charge constraint
        def soc_rule(model, i, j):
            # Get index of storage named 'i'
            index = first_dimension.index(i)
            
            # Get all the storage efficiencies
            stand_ef = self.storages[index].stand_ef
            discharge_ef = self.storages[index].discharge_ef
            charge_ef = self.storages[index].charge_ef
            
            # Get initial state of chage
            initial_soc = self.storages[index].initial_soc
            
            # Get whether storage is cyclic or not
            is_cyclic = self.storages[index].cyclic_soc
            
            # Get index of last snapshot
            last_j = len(self.snapshots) - 1
            
            # Special equation for first snapshot
            if j == 0:
                # Treat cyclic storage
                if is_cyclic:
                    return model.storage_soc[i,j] == stand_ef*model.storage_soc[i,last_j] - discharge_ef*model.storage_discharge[i,j] + charge_ef*model.storage_charge[i,j]
                # Treat non-cyclic storage
                else:
                    return model.storage_soc[i,j] == stand_ef*initial_soc - discharge_ef*model.storage_discharge[i,j] + charge_ef*model.storage_charge[i,j]
                   
            # General equation for all remaining snapshots
            else:
                return model.storage_soc[i,j] == stand_ef*model.storage_soc[i,j-1] - (1/discharge_ef)*model.storage_discharge[i,j] + charge_ef*model.storage_charge[i,j]
    
        # Get model first dimension
        first_dimension = self.get_storages_attr('name')
        
        # Add constraint to model
        self.model.storages_soc = Constraint(first_dimension, self.snapshots, rule=soc_rule)
        return
    
    def set_storage_cyclic_constraint(self):
        # Set rule for cyclic or not cyclic storage
        def cyclic(model, i, j):
            # Get index of storage named 'i'
            index = first_dimension.index(i)
            
            # Get whether storage is cyclic and its initial and final soc
            is_cyclic = self.storages[index].cyclic_soc
            initial_soc = self.storages[index].initial_soc
            final_soc = self.storages[index].final_soc
            
            # If storage is cyclic
            if is_cyclic:
                return model.storage_soc[i,j] == initial_soc
            
            # Else, if a final state of charge is set
            elif final_soc is not None:
                return model.storage_soc[i,j] == final_soc
            
            # Else, skip this constraint
            else:
                return Constraint.Skip
            
        # Get model first dimension
        first_dimension = self.get_storages_attr('name')
        
        # Add constraint to model
        self.model.storages_cyclic = Constraint(first_dimension, [self.snapshots[-1]], rule=cyclic)
        return
    
    def set_storage_exclusive_behaviour_constraint(self):
        # Set rule for storage only charging or only discharging at each snapshot
        def exclusive(model, i, j):

            # At each snapshot, charging, discharging or both should be zero
            # So charging + discharging <= 1
            return model.charge_status[i,j] + model.discharge_status[i,j] <= 1
        
        # Get model first dimension
        first_dimension = self.get_storages_attr('name')
        
        # Add constraint to model
        self.model.storages_exclusive = Constraint(first_dimension, self.snapshots, rule=exclusive)
        return
    
    
    def set_power_balance_constraint(self):
        # Set rule for power_balance constraint
        def power_balance_rule(model, j):
            # Sum of all generators dispatch at snapshot j
            if model.find_component('generator_p'):
                gens_dispatch = sum(self.model.generator_p[:,j])
            else:
                gens_dispatch = 0
            
            # Sum of all storages discharge at snapshot j
            if model.find_component('storage_discharge'):
                storages_discharge = sum(self.model.storage_discharge[:,j])
            else:
                storages_discharge = 0
            
            # Sum of all storages charge at snapshot j
            if model.find_component('storage_charge'):
                storages_charge = sum(self.model.storage_charge[:,j])
            else:
                storages_charge = 0
            
            return  gens_dispatch + storages_discharge - storages_charge  == self.load[j]
        
        # Add constraint to model 
        self.model.power_balance = Constraint(self.snapshots, rule=power_balance_rule)
        return
    
    def set_fuel_cost_constraint(self):
        # Set rule for fuel_cost variable
        def fuel_cost_rule(model, i, j):
            # Get index of generator named 'i'
            index = first_dimension.index(i)
            
            # Get important parameters
            Pnom = self.generators[index].p_nom
            fuel_price = self.generators[index].fuel_price
            a = self.generators[index].ef_a
            b = self.generators[index].ef_b
            k = self.generators[index].ef_k
            
            # Get dispatch per unit and status variables
            p_var = model.generator_p[i,j] / Pnom 
            status_var = model.generator_status[i,j]
            
            if self.generators[index].constant_efficiency:
                # Equation for smooth cost function: cost(p=0) != 0
                smooth_fuel_cost = Pnom/k * p_var * fuel_price
                
            else:
                # Set 'x' auxiliary variable
                x = a/b * p_var
                
                # Equation for smooth cost function: cost(p=0) != 0
                smooth_fuel_cost = Pnom/b * (1 - x + x**2 - x**3) * fuel_price
            
            # Equation for non-smooth cost function: cost(p=0) = 0
            return model.fuel_cost[i,j] >= smooth_fuel_cost * status_var     
        
        # Get model first dimension
        first_dimension = self.get_generators_attr('name', carrier='gas')
        
        # Add constraint to model
        self.model.fuel_cost_constraint = Constraint(first_dimension, self.snapshots, rule=fuel_cost_rule)
        return

    def set_model_constraints(self):
        # Generators constraints
        self.set_gen_p_min_constraint()
        self.set_gen_p_max_constraint()
        self.set_gen_ramp_up_constraint()
        self.set_gen_ramp_down_constraint()
        self.set_gen_min_uptime_constraint()
        self.set_gen_min_downtime_constraint()
        self.set_gen_start_up_cost_constraint()
        self.set_gen_shut_down_cost_constraint()
        
        # Storages constraints
        self.set_storage_max_ch()
        self.set_storage_max_disch()
        self.set_storage_soc_constraint()
        self.set_storage_cyclic_constraint()
        self.set_storage_exclusive_behaviour_constraint()
        
        # Network constraints
        self.set_power_balance_constraint()
        
        # Fuel cost constraints
        self.set_fuel_cost_constraint()
        
        return
    
    def set_model_objective_function(self, model=None):
            
        # Set expression for the objective funtion
        def objective_function_expression(model):
            # Initialize expression with zero
            expression = 0
                    
            if self.gas_generators:
                # 1. Include fuel cost
                expression += sum(model.fuel_cost[:,:])
                
                # 2. Include start up cost
                expression += sum(model.generator_start_up_cost[:,:])
            
                # 3. Include shut down cost
                expression += sum(model.generator_shut_down_cost[:,:])
            
            
            return expression
    
        
        # Include objective to main model
        self.model.objective = Objective(rule=objective_function_expression, sense=minimize)


    def save_dispatches(self):
        if self.is_solved:
            # Initialize dispatch array
            generators_p = np.zeros((len(self.generators), len(self.snapshots)))
            
            # Iterate over the snapshots
            for j in self.snapshots:
                
                # Iterate over the generators
                for i, gen in enumerate(self.generators):
                    generators_p[i, j] = self.model.generator_p[gen.name, j].value
                    
            # Save dispatches
            generators_p = np.array(generators_p).T
            generators_names = self.get_generators_attr('name')
            self.generators_p = pd.DataFrame(generators_p, columns = generators_names)
            return
        return
    
    def save_status(self):
        if self.is_solved:
            # Initialize status array
            generators_status = np.zeros((len(self.generators), len(self.snapshots)))
            storages_ch_status = np.zeros((len(self.storages), len(self.snapshots)))
            storages_disch_status = np.zeros((len(self.storages), len(self.snapshots)))
            
            # Iterate over the snapshots
            for j in self.snapshots:
                
                # Iterate over the generators
                for i, gen in enumerate(self.generators):
                    generators_status[i, j] = self.model.generator_status[gen.name, j].value
                    
                # Iterate over the storages
                for i, st in enumerate(self.storages):
                    storages_ch_status[i,j] = self.model.charge_status[st.name, j].value
                    storages_disch_status[i,j] = self.model.discharge_status[st.name, j].value
                    
            # Save generators status
            generators_names = self.get_generators_attr('name')
            generators_status = np.array(generators_status).T
            self.generators_status = pd.DataFrame(generators_status, columns = generators_names)
            
            # Save storages status
            storages_names = self.get_storages_attr('name')
            storages_ch_status = np.array(storages_ch_status).T
            storages_disch_status = np.array(storages_disch_status).T
            self.discharge_status = pd.DataFrame(storages_disch_status, columns = storages_names)
            self.charge_status = pd.DataFrame(storages_ch_status, columns = storages_names)
            return
        return
    
    def save_costs(self):
        if self.is_solved:
            # Initialize cost array
            cost_array = np.zeros(len(self.snapshots))
            
            # Iterate over the snapshots
            for j in self.snapshots:
                # Initialize cost at each snapshot
                cost = 0
                
                # Iterate over the gas generators 
                for gen in self.gas_generators:
                    
                    gen_p = self.model.generator_p[gen.name,j].value
                    
                    # Sum fuel cost
                    if gen_p < 1e-5:
                        cost += 0
                    else:
                        cost += self.model.fuel_cost[gen.name,j].value
                    
                    # Sum start up cost
                    cost += self.model.generator_start_up_cost[gen.name,j].value
                    
                    # Sum shut down cost
                    if j == 0 and gen_p < 1e-5:
                        cost += 0
                    else:
                        cost += self.model.generator_shut_down_cost[gen.name,j].value
                
                # Update cost array
                cost_array[j] = cost
                
            # Save costs
            self.costs = pd.DataFrame(cost_array, columns = ['cost'])
            return
        return
    
    def save_emissions(self):
        # Emission in kg of CO2
        
        if self.is_solved:
            # Initialize emissions array
            emissions_array = np.zeros(len(self.snapshots))
            
            # Iterate over the snapshots
            for j in self.snapshots:
                # Initialize emissions at each snapshot
                emissions_sum = 0
                                
                # Iterate over gas generators:
                for gen in self.gas_generators:
                    # Get dispatch in MW for given generator
                    gen_p = self.model.generator_p[gen.name,j].value
                    
                    # Calculate dispatch per unit for given generator
                    gen_p_pu = gen_p / gen.p_nom
                    
                    # Get efficiency parameters to calculate efficiency at each snapshot
                    ef_a = gen.ef_a
                    ef_b = gen.ef_b
                    ef_k = gen.ef_k
                    
                    # Calculate efficiency at each snapshot
                    if gen.constant_efficiency:
                        efficiency = ef_k
                    else:
                        efficiency = ef_a * gen_p_pu**2 + ef_b * gen_p_pu
                    
                    # Calculate power provided by the fuel in MW
                    if efficiency < 1e-3:
                        fuel_p = 0
                    else:
                        fuel_p = gen_p/efficiency
                    
                    # Calculate emission
                    emissions = gen.co2_per_mw * fuel_p
                    
                    # Sum emissions of given generator to total emissions at snapshot j
                    emissions_sum += emissions
                    
                # Update emissions array with calculated emission
                emissions_array[j] = emissions_sum
            
            # Save emissions
            self.emissions = pd.DataFrame(emissions_array, columns = ['emission'])
            return
        return
    
    
    def save_fuel_consumption(self):
        # Fuel consumption in kg of natural gas
        
        if self.is_solved:
            # Initialize emissions array
            fuel_array = np.zeros(len(self.snapshots))
            
            # Iterate over the snapshots
            for j in self.snapshots:
                # Initialize emissions at each snapshot
                fuel_sum = 0
                
                # Iterate over gas generators:
                for gen in self.gas_generators:
                    # Get dispatch in MW for given generator
                    gen_p = self.model.generator_p[gen.name,j].value
                    
                    if gen_p < 1e-5:
                        gen_p = 0
                        
                    # Calculate fuel consumption
                    fuel = gen.SFC * gen_p
                    
                    # Sum fuel of given generator to total fuel at snapshot j
                    fuel_sum += fuel
                    
                # Update emissions array with calculated emission
                fuel_array[j] = fuel_sum
                
            # Save emissions
            self.fuel = pd.DataFrame(fuel_array, columns = ['fuel'])
            return
        return
    
    def save_state_of_charge(self):
        if self.is_solved:
            # Initialize state of charge array
            state_of_charge = np.zeros((len(self.storages), len(self.snapshots)))
            
            # Iterate over the snapshots
            for j in self.snapshots:
                
                # Iterate over the storages
                for i, st in enumerate(self.storages):
                    state_of_charge[i,j] = self.model.storage_soc[st.name, j].value
                    
            # Save state of charge
            storages_names = self.get_storages_attr('name')
            state_of_charge = np.array(state_of_charge).T
            self.state_of_charge = pd.DataFrame(state_of_charge, columns = storages_names)
            
            return
        return
    
    def save_storages_dispatch(self):
        if self.is_solved:
            # Initialize storage dispatch array
            st_dispatch = np.zeros((len(self.storages), len(self.snapshots)))
            
            # Iterate over the snapshots
            for j in self.snapshots:
                
                # Iterate over the storages
                for i, st in enumerate(self.storages):
                    st_dispatch[i,j] = self.model.storage_discharge[st.name, j].value - self.model.storage_charge[st.name, j].value
                    
            # Save storage dispatch
            storages_names = self.get_storages_attr('name')
            st_dispatch = np.array(st_dispatch).T
            self.storage_p = pd.DataFrame(st_dispatch, columns = storages_names)
            
            return
        return
    
    def save_rocof(self):
        if self.is_solved:
            # Initialize storage dispatch array
            rocof = np.zeros((len(self.snapshots)))
            inertia = np.zeros((len(self.snapshots)))
            
            # Define constant values
            f0 = 60
            H = 3.2
            
            # Get only gas generators dispatches
            gas_gen_names = self.get_generators_attr('name', carrier='gas')
            gas_gen_P = []
            gas_gen_u = []
            
            # Iterate over gas generators
            for gen_name in gas_gen_names:
                gas_gen_P_row = []
                gas_gen_u_row = []
                
                # Iterate over snapshots
                for j in self.snapshots:
                    gen_p = self.model.generator_p[gen_name,j].value
                    gas_gen_P_row.append(gen_p)
                    
                    gen_u = self.model.generator_status[gen_name,j].value
                    gas_gen_u_row.append(gen_u)
                    
                gas_gen_P.append(gas_gen_P_row)
                gas_gen_u.append(gas_gen_u_row)
                
            gas_gen_P = pd.DataFrame(np.asarray(gas_gen_P).T, columns=gas_gen_names)
            gas_gen_u = pd.DataFrame(np.asarray(gas_gen_u).T, columns=gas_gen_names)
            
            
            # Get maximum dispatch among the active generators
            Pg_max = gas_gen_P.max(axis=1).to_numpy().T
                
            # Get sum of dispatches
            Pg_total = gas_gen_P.sum(axis=1).to_numpy().T
            
            # Get sum of statuses
            sum_status = gas_gen_u.sum(axis=1).to_numpy().T
            
            # Get load
            load = self.load
            
            # Iterate over the snapshots
            for j in self.snapshots:
                
                # Calculate Rated Power and Equivalent Inertia at each snapshot
                row = gas_gen_u.iloc[[j]]
                active_gens = row.columns[(row == 1.0).iloc[0]].array
                
                Srated = 0
                H_eq = 0
                for gen in self.gas_generators:
                    if gen.name in active_gens:
                        Srated += gen.p_nom
                        H_eq += gen.inertia_constant
                
                inertia[j] = H_eq
                rocof[j] = f0/(2*H_eq) * ((Pg_total[j] - Pg_max[j]) - load[j]) / Srated
                
            
            # Save ROCOF
            self.rocof = pd.DataFrame(rocof, columns = ['ROCOF [Hz/s]'])
            
            # Save Available Inertia
            self.inertia = pd.DataFrame(inertia, columns = ['Available Inertia [s]'])
            
            return
        return
        
        
    def solve(self, show_complete_info=True):
        
        # Initialize model
        self.set_optimization_model()
        
        # Include model cold variables
        self.set_optimization_variables()
        
        # Include model constraints
        self.set_model_constraints()        
        
        # Include model objective function
        self.set_model_objective_function()
        
        # Solve the model
        print('Solving main network...\n')
        
        start_time = time.time()
        res = SolverFactory('mindtpy').solve(self.model,
                                            #  strategy='GOA',
                                             mip_solver='gurobi',
                                             nlp_solver='ipopt',
                                             tee=show_complete_info
                                             )
        
        self.optimization_time = time.time() - start_time
        
        # Print termination conditions
        if (res.solver.status == SolverStatus.ok) and (res.solver.termination_condition == TerminationCondition.optimal):
            self.is_solved = True
            print('Termination condition: Feasible and Optimal\n')
            print('Time of optimization: ' + str(self.optimization_time) + '\n')
        elif res.solver.termination_condition == TerminationCondition.infeasible:
            self.is_solved = False
            print('Termination condition: Infeasible\n')
            log_infeasible_constraints(self.model, 
                                       log_expression=True, 
                                       log_variables=True)
        elif (res.solver.status == SolverStatus.ok) and (res.solver.termination_condition == TerminationCondition.feasible):
            self.is_solved = True
            print('Termination condition: Feasible\n')
            print('Time of optimization: ' + str(self.optimization_time) + '\n')
        else:
            print(str(res.solver))
                    
        # Save results
        self.save_dispatches()
        self.save_status()
        self.save_costs()
        self.save_emissions()
        self.save_fuel_consumption()
        self.save_state_of_charge()
        self.save_storages_dispatch()
        self.save_rocof()
            
        # Model info (Variables, Constraints and Objective Function)
        if show_complete_info:
            self.model.pprint()
            print(res)
        return
    
    def export_results_to_xlsx(self, filename, include_means=True, include_status=False, compact_format=True):

        self.export_unit_commitment(filename=filename, include_means=include_means, include_status=include_status, compact_format=compact_format)
        if self.storages:
            self.export_storage(filename=filename, include_means=include_means, include_status=include_status, compact_format=compact_format)
        
        return
    
    def export_unit_commitment(self, filename, sheet_name='Unit commitment', include_means=True, include_status=False, compact_format=True):
        if self.is_solved:
            # Create empty data container
            data = {}
            
            # Add Load to data container
            data['Load [MW]'] = self.load
            
            # Add Dispatches to data container
            for gen in self.generators:
                data[gen.name + ' [MW]'] = list(self.generators_p[gen.name])
            
            # Add Cost and Total cost to data container
            data['Cost [USD]'] = list(self.costs['cost'])
            data['Total cost [USD]'] = np.sum(data['Cost [USD]'])
            
            # Add Emission and Total emission to data container
            data['Emission [kg]'] = list(self.emissions['emission'])
            data['Total emission [kg]'] = np.sum(data['Emission [kg]'])
            
            # Add Fuel and Total fuel to data container
            data['Fuel [kg]'] = list(self.fuel['fuel'])
            data['Total fuel [kg]'] = np.sum(data['Fuel [kg]'])
            
            # Add ROCOF to data container
            data['ROCOF [Hz/s]'] = list(self.rocof['ROCOF [Hz/s]'])
            
            # Add Available Inertia to data container
            data['Available Inertia [s]'] = list(self.inertia['Available Inertia [s]'])
            
            # Add Status to data container
            if include_status:
                for gen in self.generators:
                    data[gen.name + ' Status'] = list(self.generators_status[gen.name])
        
            # Convert data container into DataFrame
            data = pd.DataFrame(data)
            
            # Include Means line at the end of DataFrame
            if include_means:
                data.loc['Mean'] = data.mean(axis=0)
            
            # Save Excel sheet
            writer = pd.ExcelWriter('results/'+filename, engine="xlsxwriter")
            if compact_format:
                data.to_excel(writer, sheet_name=sheet_name, float_format='%.2f')
            else:
                data.to_excel(writer, sheet_name=sheet_name)
            # Auto-adjust columns' width
            for column in data:
                if compact_format:
                    column_width = max(data[column].map('{:,.2f}'.format).astype(str).map(len).max(), len(column))
                    column_width = max(column_width, 11)
                else:
                    column_width = max(data[column].astype(str).map(len).max(), len(column))
                col_idx = data.columns.get_loc(column)+1
                writer.sheets[sheet_name].set_column(col_idx, col_idx, column_width)
            writer.save()                   
        return

    def export_storage(self, filename, sheet_name='Storages', include_means=True, include_status=False, compact_format=True):
        if self.is_solved:
            # Create empty data container
            data = {}
            
            # Add Load to data container
            data['Load [MW]'] = self.load
        
            # Add Gas Power to data container
            gas_gen_names = self.get_generators_attr('name', carrier='gas')
            gas_dispatch = self.generators_p.filter(items=gas_gen_names)
            gas_dispatch = gas_dispatch.sum(axis='columns')
            data['Gas Power [MW]'] = list(gas_dispatch)

            # Add Wind Power to data container
            wind_gen_names = self.get_generators_attr('name', carrier='wind')
            wind_dispatch = self.generators_p.filter(items=wind_gen_names)
            wind_dispatch = wind_dispatch.sum(axis='columns')
            data['Wind Power [MW]'] = list(wind_dispatch)
            
            # Add Wind Curtailment to data container
            wind_total_power = np.zeros((len(self.snapshots)))
            for wind_gen in self.wind_generators:
                wind_total_power += wind_gen.generated_power
            curtailment = wind_total_power - wind_dispatch
            curtailment[curtailment < 0] = 0
            data['Curtailment [MW]'] = list(curtailment)
                        
            # Add Storage Power to data container and State of Charge
            for st in self.storages:
                data[st.name + ' [MW]'] = list(self.storage_p[st.name])
                data[st.name + ' SOC [MWh]'] = list(self.state_of_charge[st.name])
            
            # Convert data container into DataFrame
            data = pd.DataFrame(data)
            
            # Include Means line at the end of DataFrame
            if include_means:
                data.loc['Mean'] = data.mean(axis=0)
                
            # Save Excel sheet
            writer = pd.ExcelWriter('results/'+filename, engine="openpyxl", mode='a')
            if compact_format:
                data.to_excel(writer, sheet_name=sheet_name, float_format='%.2f')
            else:
                data.to_excel(writer, sheet_name=sheet_name)
            # Auto-adjust columns' width
            for column in data:
                if compact_format:
                    column_width = max(data[column].map('{:,.2f}'.format).astype(str).map(len).max(), len(column))
                    column_width = max(column_width, 15)
                else:
                    column_width = max(data[column].astype(str).map(len).max(), len(column))
                col_idx = data.columns.get_loc(column)+2
                writer.sheets[sheet_name].column_dimensions[get_column_letter(col_idx)].width = column_width
            writer.save()                   
        return
            
    def plot_results(self, display_plot=True, save_plot=False, plot_dpi=400, load_color='#263481', gas_gen_colors=['#805722', '#996d37', '#b38756', '#cca680', '#d9b99a'], wind_gen_colors=['#3d8236', '#539b4c', '#70b467', '#94cd8b', '#a9daa1'], st_colors=['#7d0e79', '#de66d4', '#e890e2', '#d86ceb', '#d93bb1']):
        
        self.plot_unit_commitment(display_plot=display_plot, save_plot=save_plot, filename='Unit commitment.png', plot_dpi=plot_dpi, load_color=load_color, gas_gen_colors=gas_gen_colors, wind_gen_colors=wind_gen_colors)   
        if self.storages:
            self.plot_storages(display_plot=display_plot, save_plot=save_plot, filename='Storages.png', plot_dpi=plot_dpi, load_color=load_color, gas_gen_color=gas_gen_colors[0], wind_gen_color=wind_gen_colors[0], st_colors=st_colors)
        return
      
    def plot_unit_commitment(self, display_plot=True, save_plot=False, filename='Unit commitment.png', plot_dpi=400, load_color='#263481', gas_gen_colors=['#805722', '#996d37', '#b38756', '#cca680', '#d9b99a'], wind_gen_colors=['#3d8236', '#539b4c', '#70b467', '#94cd8b', '#a9daa1']):
        if self.is_solved:
            # SET CONFIGS
            x_axis = range(len(self.load))

            fig = plt.figure(figsize=(1200/100, 800/100), dpi=100)
            
            # PLOT LOAD
            plt.fill_between(x_axis, self.load, color=load_color, label='Load', alpha=0.3)
            plt.plot(x_axis, self.load, color=load_color)

            # PLOT GAS GENERATORS
            gas_generators = [gen for gen in self.gas_generators if gen.carrier == 'gas']
            for index, gen in enumerate(gas_generators):
                y_axis = list(self.generators_p[gen.name])
                plt.fill_between(x_axis, y_axis, color=gas_gen_colors[index], label=gen.name, alpha=0.3)
                plt.plot(x_axis, y_axis, color=gas_gen_colors[index])

            # PLOT WIND GENERATORS
            wind_generators = [gen for gen in self.wind_generators if gen.carrier == 'wind']
            for index, gen in enumerate(wind_generators):
                y_axis = list(self.generators_p[gen.name])
                plt.fill_between(x_axis, y_axis, color=wind_gen_colors[index], label=gen.name, alpha=0.3)
                plt.plot(x_axis, y_axis, color=wind_gen_colors[index])

            # SET CONFIGS
            plt.legend(loc='upper left', ncol=7)

            # plt.title('LGridPy\nUnit commitment and optimal dispatch')
            plt.title('Unit commitment and Optimal Dispatch of\nGas Turbines and Wind Turbine')
            plt.xlabel('Snapshots [h]')
            plt.ylabel('Dispatch [MW]')
            
            fig.tight_layout()
            plt.grid()        

            if save_plot:
                print('Saving plots results...\n')
                plt.savefig('results/'+filename, dpi=plot_dpi)
                
            if not self.storages:
                plt.show()   
                plt.close()
            
            return
            
    def plot_storages(self, display_plot=True, save_plot=False, filename='Storages.png', plot_dpi=400, load_color='#263481', gas_gen_color='#805722', wind_gen_color='#3d8236', st_colors=['#7d0e79', '#de66d4', '#e890e2', '#d86ceb', '#d93bb1']):
        if self.is_solved:
            # SET CONFIGS
            x_axis = range(len(self.load))

            fig, ax = plt.subplots(2, figsize=(1200/100, 800/100), dpi=100, sharex=True)
            
            # PLOT STATE OF CHARGE
            for i, st in enumerate(self.storages):
                y_axis = np.asarray(self.state_of_charge[st.name])
                y_axis = y_axis / st.nom_capacity * 100
                ax[0].plot(x_axis, y_axis, label=st.name, color=st_colors[i])
                
            # SET STATE OF CHARGE PLOT CONFIGS
            ax[0].set_ylabel('State of charge [%]')
            ax[0].legend(loc='upper left', ncol=7)
            ax[0].grid()
            
            # PLOT LOAD
            ax[1].plot(x_axis, self.load, color=load_color)
            ax[1].fill_between(x_axis, self.load, label='Load', color=load_color, alpha=0.3)
            
            # PLOT GAS GENERATION
            gas_gen_names = self.get_generators_attr('name', carrier='gas')
            gas_dispatch = self.generators_p.filter(items=gas_gen_names)
            gas_dispatch = list(gas_dispatch.sum(axis='columns'))
            ax[1].plot(x_axis, gas_dispatch, color=gas_gen_color)
            ax[1].fill_between(x_axis, gas_dispatch, label='Gas', color=gas_gen_color, alpha=0.3)
            
            # PLOT WIND GENERATION
            wind_gen_names = self.get_generators_attr('name', carrier='wind')
            wind_dispatch = self.generators_p.filter(items=wind_gen_names)
            wind_dispatch = list(wind_dispatch.sum(axis='columns'))
            ax[1].plot(x_axis, wind_dispatch, color=wind_gen_color)
            ax[1].fill_between(x_axis, wind_dispatch, label='Wind', color=wind_gen_color, alpha=0.3)
            
            # PLOT WIND CURTAILMENT
            wind_total_power = np.zeros((len(self.snapshots)))
            for wind_gen in self.wind_generators:
                wind_total_power += wind_gen.generated_power
            curtailment = wind_total_power - wind_dispatch
            curtailment[curtailment < 0] = 0
            y_axis = curtailment
            plt.fill_between(x_axis, y_axis, color='y', label='Curtailment', alpha=0.3)
            plt.plot(x_axis, y_axis, color='y')
            
            # PLOT STORAGES DISPATCHES
            for i, st in enumerate(self.storages):
                storage_p = list(self.storage_p[st.name])
                ax[1].plot(x_axis, storage_p, color=st_colors[i])
                ax[1].fill_between(x_axis, storage_p, label=st.name, color=st_colors[i], alpha=0.3)
                
            
            ax[1].set_ylabel('Power [MW]')
            ax[1].set_xlabel('Snapshots [h]')
            
            ax[1].legend(loc='upper left', ncol=7)
            ax[1].grid()
            
            # fig.suptitle('LGridPy\nStorages')
            fig.suptitle('Storage State of Charge and Unit Commitment of\nGas Turbines, Wind Turbine and Storage')
            
            fig.tight_layout()
            
            if save_plot:
                plt.savefig('results/'+filename, dpi=plot_dpi)
                
            if display_plot:
                plt.show()   
            plt.close()
            print('Plotting finished')
            