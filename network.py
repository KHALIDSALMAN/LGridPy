from pyomo.environ import *
from numpy import inf
import matplotlib.pyplot as plt
from pprint import pprint
import json
import copy
import os
import time

from generators import *

class Network:

    def __init__(self, name='Unknown'):
        self.name = name
        self.is_solved = False
        self.gas_generators = []
        self.wind_generators = []
        self.generators = []

    def add_load(self, load):
        self.load = load
        self.snapshots = range(len(load))
        return

    def add_gas_generator(self, name, p_nom, p_min_pu, p_max_pu, min_uptime, min_downtime, ramp_up_limit, ramp_down_limit, start_up_cost, shut_down_cost, efficiency_curve, efficiency_type, fuel_price, co2_per_mw=0.517, SFC=0.215, **kwargs):
        # Create GasGenerator object with input parameters
        gas_generator = GasGenerator(name, 'gas', p_nom, p_min_pu, p_max_pu, min_uptime, min_downtime, ramp_up_limit, ramp_down_limit, start_up_cost, shut_down_cost, efficiency_curve=efficiency_curve, efficiency_type=efficiency_type, fuel_price=fuel_price, co2_per_mw=co2_per_mw, SFC=SFC)
        
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
    
    def set_optimization_variables(self, initialize=None):
        # Get model first dimension = generators names
        first_dimension_all = self.get_generators_attr('name')
        first_dimension_gas = self.get_generators_attr('name', 'gas')

        # Set Pyomo variables
        self.model.generator_p = Var(first_dimension_all, self.snapshots, within=Reals)
        self.model.generator_status = Var(first_dimension_all, self.snapshots, within=Binary)
        self.model.generator_start_up_cost = Var(first_dimension_gas, self.snapshots, within=Reals)
        self.model.generator_shut_down_cost = Var(first_dimension_gas, self.snapshots, within=Reals)
        
        return
    
    def set_p_min_constraint(self):
        
        # Set rule for p_min constraint
        def p_min_rule(model, i, j):
            # Get index of generator named 'i'
            index = first_dimension.index(i)
            
            # Get p_min in MW
            p_min = self.generators[index].p_min_pu[j] * self.generators[index].p_nom
            
            # Set constraint equation
            return -self.model.generator_p[i,j] + p_min * self.model.generator_status[i,j] <= 0
        
        # Get model first dimension
        first_dimension = self.get_generators_attr('name')
        
        # Add constraint to model
        self.model.p_min = Constraint(first_dimension, self.snapshots, rule=p_min_rule)
        return
    
    def set_p_max_constraint(self):
        
        # Set rule for p_max constraint
        def p_max_rule(model, i, j):
            # Get index of generator named 'i'
            index = first_dimension.index(i)
            
            # Get p_max in MW
            p_max = self.generators[index].p_max_pu[j] * self.generators[index].p_nom
            
            # Set constraint equation
            return self.model.generator_p[i,j] - p_max * self.model.generator_status[i,j] <= 0
        
        # Get model first dimension
        first_dimension = self.get_generators_attr('name')
        
        # Add constraint to model
        self.model.p_max = Constraint(first_dimension, self.snapshots, rule=p_max_rule)
        return
    
    def set_ramp_up_constraint(self):
        
        # Set rule for ramp_up constraint
        def ramp_up_rule(model, i, j):
            # Get index of generator named 'i'
            index = first_dimension.index(i)

            if self.generators[index].carrier == 'wind':
                return Constraint.Skip
            
            # Get ramp_up_limit in MW
            rul = self.generators[index].ramp_up_limit * self.generators[index].p_nom
            
            # Set constraint equation
            return self.model.generator_p[i,j] - self.model.generator_p[i,j-1] <= rul * self.model.generator_status[i,j]
        
        # Get model first dimension
        first_dimension = self.get_generators_attr('name')
        
        # Add constraint to model
        self.model.ramp_up_limit = Constraint(first_dimension, self.snapshots[1:], rule=ramp_up_rule)
        return
        
    def set_ramp_down_constraint(self):
        
        # Set rule for ramp_down constraint
        def ramp_down_rule(model, i, j):
            # Get index of generator named 'i'
            index = first_dimension.index(i)

            if self.generators[index].carrier == 'wind':
                return Constraint.Skip
            
            # Get ramp_down_limit in MW
            rdl = self.generators[index].ramp_down_limit * self.generators[index].p_nom
            
            # Set constraint equation
            return -self.model.generator_p[i,j] + self.model.generator_p[i,j-1] <= rdl * self.model.generator_status[i,j-1]
        
        # Get model first dimension
        first_dimension = self.get_generators_attr('name')
        
        # Add constraint to model
        self.model.ramp_down_limit = Constraint(first_dimension, self.snapshots[1:], rule=ramp_down_rule)
        return
    
    
    def set_min_uptime_constraint(self):
        
        # Set rule for min_uptime constraint
        def min_uptime_rule(model, i, j):
            # Get index of generator named 'i'
            index = first_dimension.index(i)

            if self.generators[index].carrier == 'wind':
                return Constraint.Skip
            
            # Get min_uptime (notice that it is treated as an INTEGER value)
            mut = floor(self.generators[index].min_uptime)
            
            # If the min_uptime < 1, this constraint is skipped
            if mut == 0:
                return Constraint.Skip
            
            # Treat the upper limit of the sum (if the snapshot is close to the end, the upper limit is the remaining snapshots)
            sum_upper_limit = min(j+mut, len(self.load)-1)
            
            # Activate the generator on the first snapshot
            if j == 0:
                return model.generator_status[i,j] == 1
            
            # Set constraint equation
            return -sum(self.model.generator_status[i,k] for k in range(j, sum_upper_limit)) + mut * (self.model.generator_status[i,j] - self.model.generator_status[i,j-1]) <= 0
        
        # Get model first dimension
        first_dimension = self.get_generators_attr('name')
        
        # Add constraint to model (except for the last snapshot)
        self.model.min_uptime = Constraint(first_dimension, self.snapshots[:-1], rule=min_uptime_rule)
        return
    
    def set_min_downtime_constraint(self):
        
        # Set rule for min_downtime constraint
        def min_downtime_rule(model, i, j):
            # Get index of generator named 'i'
            index = first_dimension.index(i)

            if self.generators[index].carrier == 'wind':
                return Constraint.Skip
            
            # Get min_downtime (notice that it is treated as an INTEGER value)
            mdt = floor(self.generators[index].min_downtime)
            
            # If the min_downtime < 1, this constraint is skipped
            if mdt == 0:
                return Constraint.Skip
            
            # Treat the upper limit of the sum (if the snapshot is close to the end, the upper limit is the remaining snapshots)
            sum_upper_limit = min(j+mdt, len(self.load)-1)
            
            # Special equation for the first snapshot (generator_status at snapshot '-1' = 1)
            if j == 0:
                return sum(self.model.generator_status[i,k] for k in range(j, sum_upper_limit)) - mdt * (self.model.generator_status[i,j]) <= 0
            
            # Set constraint equation
            return sum(self.model.generator_status[i,k] for k in range(j, sum_upper_limit)) - mdt * (self.model.generator_status[i,j] - self.model.generator_status[i,j-1] + 1) <= 0
        
        # Get model first dimension
        first_dimension = self.get_generators_attr('name')
        
        # Add constraint to model (except for the last snapshot)
        self.model.min_downtime = Constraint(first_dimension, self.snapshots[:-1], rule=min_downtime_rule)
        return
    
    def set_start_up_cost_constraint(self):
        # Set rule for start_up constraint
        def start_up_cost_rule(model, i, j): 
            # Get index of generator named 'i'
            index = first_dimension.index(i)

            if self.generators[index].carrier == 'wind':
                return Constraint.Skip
            
            # Get start up cost in dollars
            su = self.generators[index].start_up_cost
            
            # Special equation for the first snapshot (generator_status at snapshot '-1' = 1)
            if j == 0:
                return -self.model.generator_start_up_cost[i,j] + su * self.model.generator_status[i,j] - su <= 0
            
            # Set constraint equation
            return -self.model.generator_start_up_cost[i,j] + su * (self.model.generator_status[i,j] - self.model.generator_status[i,j-1]) <= 0
        
        # Get model first dimension
        first_dimension = self.get_generators_attr('name')
        
        # Add constraint to model (except for the last snapshot)
        self.model.start_up_cost = Constraint(first_dimension, self.snapshots, rule=start_up_cost_rule)
        return
    
    def set_shut_down_cost_constraint(self):
        # Set rule for shut_down constraint
        def shut_down_cost_rule(model, i, j):
            # Get index of generator named 'i'
            index = first_dimension.index(i)

            if self.generators[index].carrier == 'wind':
                return Constraint.Skip
            
            # Get start up cost in dollars
            sd = self.generators[index].shut_down_cost
            
            # Special equation for the first snapshot (generator_status at snapshot '-1' = 1)
            if j == 0:
                return -self.model.generator_shut_down_cost[i,j] - sd * self.model.generator_status[i,j] + sd <= 0
            
            # Set constraint equation
            return -self.model.generator_shut_down_cost[i,j] - sd * (self.model.generator_status[i,j] - self.model.generator_status[i,j-1]) <= 0
        
        # Get model first dimension
        first_dimension = self.get_generators_attr('name')
        
        # Add constraint to model (except for the last snapshot)
        self.model.shut_down_cost = Constraint(first_dimension, self.snapshots, rule=shut_down_cost_rule)
        return
    
    def set_power_balance_constraint(self):
        # Set rule for power_balance constraint
        def power_balance_rule(model, j):
            return sum(self.model.generator_p[:,j]) - self.load[j] == 0
        
        # Get model first dimension
        first_dimension = self.get_generators_attr('name')
        
        # Add constraint to model (except for the last snapshot)
        self.model.power_balance = Constraint(self.snapshots, rule=power_balance_rule)
    
    def set_model_constraints(self):
        self.set_p_min_constraint()
        self.set_p_max_constraint()
        self.set_ramp_up_constraint()
        self.set_ramp_down_constraint()
        self.set_min_uptime_constraint()
        self.set_min_downtime_constraint()
        self.set_start_up_cost_constraint()
        self.set_shut_down_cost_constraint()
        self.set_power_balance_constraint()
        return
    
    def set_model_objective_function(self, preprocessment=False, model=None):
        
        # Get preprocessment linear cost for objective function
        preproc_costL = []
        if preprocessment:
            for gen in self.gas_generators:
                costL = gen.set_efficiency_curve_parameters(preprocessment=True)
                preproc_costL.append(costL)
            
        # Set expression for the objective funtion
        def objective_function_expression(model):
            # Initialize expression with zero
            expression = 0
            
            # If preprocessment, declare a counter on preproc_costL
            preproc_c = 0
            
            # 1. Include fuel cost
            for i, gen in enumerate(self.generators):
                
                # Include only gas generators
                if gen.carrier == 'gas':
                    # Calculate the quadratic coefficient of the cost
                    costQ = gen.ef_a * gen.fuel_price / gen.p_nom
                    
                    # Calculate the linear coefficient of the cost
                    costL = gen.ef_b * gen.fuel_price
                    
                    # If preprocessment, calculate the linear expression of cost
                    if preprocessment:
                        # Calculate the linear coefficient of the cost
                        costL = preproc_costL[preproc_c] * gen.fuel_price
                        
                         # Calculate the cost expression for each snapshot
                        for j in self.snapshots:
                            expression += costL * model.generator_p[gen.name,j]
                            
                        # Update our preproc counter
                        preproc_c += 1
                    
                    # Otherwise, proceed with quadratic calculation of cost at each snapshot
                    else:
                        for j in self.snapshots:
                            expression += costQ * model.generator_p[gen.name,j]**2 + costL * model.generator_p[gen.name,j]

                    
            
            # 2. Include start up cost
            expression += sum(model.generator_start_up_cost[:,:])
            
            # 3. Include shut down cost
            expression += sum(model.generator_shut_down_cost[:,:])
            
            return expression
        
        # If in preprocessment, do not add objective to main model
        if preprocessment:
            model.objective = Objective(rule=objective_function_expression, sense=minimize)
        
        # Else, include objective to main model
        else:
            self.model.objective = Objective(rule=objective_function_expression, sense=minimize)

    def preprocessment(self, instance):
        
        # Include preprocessment objective function to model instance
        self.set_model_objective_function(preprocessment=True, model=instance)
        
        # Run preprocessment results
        threads = os.cpu_count()
        
        preproc = SolverFactory('mindtpy').solve(instance, 
                                                 mip_solver='gurobi',
                                                 nlp_solver='ipopt',
                                                 threads=threads)
        
        # Save preprocessment data to network
        # Initialize dispatch array
        generators_p = np.zeros((len(self.generators), len(self.snapshots)))
        
        # Iterate over the snapshots
        for j in self.snapshots:
            
            # Iterate over the generators
            for i, gen in enumerate(self.generators):
                generators_p[i, j] = instance.generator_p[gen.name, j].value
        
        generators_p = generators_p.T
        
        # Save preprocessment results to network
        self.preprocessment = generators_p
        return
                
    def update_model_variables(self):
        # Initialize variables with warmstart
        first_dimension_all = self.get_generators_attr('name')

        # Get preprocessment results
        initialize = self.preprocessment

        # Make initial guess to generators dispatches
        for index, gen in enumerate(first_dimension_all):
            for t in self.snapshots:
                self.model.generator_p[gen, t] = initialize[t][index]
                
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
            
            # Iterate over the snapshots
            for j in self.snapshots:
                
                # Iterate over the generators
                for i, gen in enumerate(self.generators):
                    generators_status[i, j] = self.model.generator_status[gen.name, j].value
                    
            # Save dispatches
            generators_status = np.array(generators_status).T
            generators_names = self.get_generators_attr('name')
            self.generators_status = pd.DataFrame(generators_status, columns = generators_names)
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
                for i, gen in enumerate(self.generators):
                    
                    # Select gas generators ONLY
                    if gen.carrier == 'gas':
                        # Get simpler name for the generator dispatch
                        gen_p = float(self.model.generator_p[gen.name, j].value)
                        
                        # Quadratic coefficient of the cost
                        costQ = gen.ef_a * gen.fuel_price / gen.p_nom
                        
                        # Linear coefficient of the cost
                        costL = gen.ef_b * gen.fuel_price
                        
                        # Constant coefficient of the cost
                        costC = gen.ef_c * gen.fuel_price * gen.p_nom
                        
                        # Calculate fuel cost at respective snapshot
                        cost += costQ * (gen_p**2) + costL * gen_p + costC

                        # Include start up cost and shut down cost
                        cost += self.model.generator_start_up_cost[gen.name,j].value
                        cost += self.model.generator_shut_down_cost[gen.name,j].value
                    else:
                        break
                    
                cost_array[j] = cost
                
            # Save costs
            self.costs = pd.DataFrame(cost_array, columns = ['cost'])
            return
        return
    
    def save_emissions(self):
        # Emission in tons of CO2
        
        if self.is_solved:
            # Initialize emissions array
            emissions_array = np.zeros((len(self.generators), len(self.snapshots)))
            
            # Iterate over all generators:
            for index, gen in enumerate(self.generators):
                # Get dispatch in MW for given generator
                gen_p = np.asarray(self.generators_p[gen.name])
                
                # Calculate dispatch per unit for given generator
                gen_p_pu = gen_p / gen.p_nom
                
                # Get efficiency parameters to calculate efficiency at each snapshot
                ef_a = gen.ef_a
                ef_b = gen.ef_b
                ef_c = gen.ef_c
                
                # Calculate efficiency at each snapshot
                efficiency = gen_p_pu / (ef_a * gen_p_pu**2 + ef_b * gen_p_pu + ef_c)
                
                # All efficiency values lower than 1% are corrected to zero (to avoid division mistakes on the next step)
                efficiency[np.abs(efficiency) < 0.01] = 0
                
                # Calculate power provided by the fuel in MW
                fuel_p = np.divide(gen_p, efficiency)
                
                # Calculate emissions
                if gen.carrier == 'gas':
                    emissions = gen.co2_per_mw * fuel_p
                elif gen.carrier == 'wind':
                    emissions = 0 * fuel_p
                
                # If the generator is OFF, the efficiency will be zero
                # In this case, we divide the dispatch by zero, so the emission value will be 'inf'
                # We have to update all the 'inf' values to zero
                emissions[emissions == -inf] = 0
                emissions[emissions == inf] = 0
                
                # Update emissions array with calculated emission
                emissions_array[index] = emissions

            # The emissions array has 2 dimensions: generators x snapshots
            # We want the accumulated emissions by all generators, so the array should have 1 dimension: snapshots
            # We must sum the elements in the same column (sum the emissions by all gens at one snapshot)
            emissions_array = emissions_array.sum(axis=0)
            
            # Save emissions
            self.emissions = pd.DataFrame(emissions_array, columns = ['emission'])
            return
        return
    
    
    def save_fuel_consumption(self):
        # Fuel consumption in tons of natural gas
        
        if self.is_solved:
            # Initialize emissions array
            fuel_array = np.zeros((len(self.generators), len(self.snapshots)))
            
            # Iterate over all generators:
            for index, gen in enumerate(self.generators):
                # Get dispatch in MW for given generator
                gen_p = np.asarray(self.generators_p[gen.name])
                
                # Calculate fuel consumption
                if gen.carrier == 'gas':
                    fuel = gen.SFC * gen_p
                elif gen.carrier == 'wind':
                    fuel = 0 * gen_p
                
                # Update emissions array with calculated emission
                fuel_array[index] = fuel

            # The fuel array has 2 dimensions: generators x snapshots
            # We want the accumulated fuel by all generators, so the array should have 1 dimension: snapshots
            # We must sum the elements in the same column (sum the fuel by all gens at one snapshot)
            fuel_array = fuel_array.sum(axis=0)
            
            # Save emissions
            self.fuel = pd.DataFrame(fuel_array, columns = ['fuel'])
            return
        return
        
    def solve(self, preprocessment=True, show_complete_info=True):
        
        # Initialize model
        self.set_optimization_model()
        
        # Include model cold variables
        self.set_optimization_variables()
        
        # Include model constraints
        self.set_model_constraints()
        
        if preprocessment:
            print('Performing preprocessment...\n')
               
            # Do preprocessment
            start_time = time.time()
            instance = copy.deepcopy(self.model)
            self.preprocessment(instance)
            
            # Include model preprocessed variables (only dispatches)
            self.update_model_variables()
            print('Preprocessment time: ' + str(time.time() - start_time) + '\n')
        else:
            print('Not performing preprocessment\n')
        
        # Include model objective function
        self.set_model_objective_function()

        # Solve the model
        print('Solving main network...\n')
        threads = os.cpu_count()
        
        start_time = time.time()
        res = SolverFactory('mindtpy').solve(self.model, 
                                            mip_solver='gurobi',
                                            nlp_solver='ipopt',
                                            tee=show_complete_info,
                                            threads=threads)
        
        self.optimization_time = time.time() - start_time
        
        # Save JSON file with termination conditions and time of optimization
        self.model.solutions.store_to(res)
        res.write(filename='termination_status.json', format='json')
        file_ = open('termination_status.json')
        termination_status = json.load(file_)
        self.termination_condition = termination_status['Solver'][0]['Termination condition']
        file_.close()

        if self.termination_condition == 'infeasible':
            self.is_solved = False
        elif self.termination_condition == 'optimal':
            print('Termination condition: ' + self.termination_condition)
            self.is_solved = True
            
        print('\nTime of optimization: ' + str(self.optimization_time) + '\n')
        
        # Save results
        self.save_dispatches()
        self.save_status()
        self.save_costs()
        self.save_emissions()
        self.save_fuel_consumption()
            
        # Model info (Variables, Constraints and Objective Function)
        if show_complete_info:
            self.model.pprint()
            print(res)
        return
    


    def export_results_to_xlsx(self, filename, sheet_name='LGridPy Simulation Result', include_means=True, include_status=False, compact_format=True):
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
            
            # Add Status to data container
            if include_status:
                for gen in self.generators:
                    data[gen.name + ' Status'] = list(self.generators_status[gen.name])
        
            # Convert data container into DataFrame
            data = pd.DataFrame(data)
            
            # Include Means line at the end of DataFrame
            if include_means:
                data.loc['Mean'] = data.mean()
            
            # Save Excel sheet
            writer = pd.ExcelWriter(filename, engine="xlsxwriter")
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
        else:
            print('Network is not solved.\n')
        return


    def plot_results(self, save_plot=False, filename='LGridPy_opf_result.png', plot_dpi=400, load_color='#263481', gas_gen_colors=['#805722', '#996d37', '#b38756', '#cca680', '#d9b99a'], wind_gen_colors=['#3d8236', '#539b4c', '#70b467', '#94cd8b', '#a9daa1']):
        if self.is_solved:
            # ARRANGE ORDER OF AREA PLOTS
            plot_order = list(self.generators_p.max().sort_values(ascending=False).index)
            ordered_generators = len(plot_order)*[0]

            for i, gen_name in enumerate(plot_order):
                gen = next((gen_ for gen_ in self.generators if gen_.name == gen_name), None)
                ordered_generators[i] = gen
            
            # SET CONFIGS
            x_axis = range(len(self.load))

            fig = plt.figure(figsize=(1200/100, 800/100), dpi=100)
            
            # PLOT LOAD
            plt.fill_between(x_axis, self.load, color=load_color, label='Load', zorder=0)

            # PLOT GAS GENERATORS
            gas_generators = [gen for gen in ordered_generators if gen.carrier == 'gas']
            for index, gen in enumerate(gas_generators):
                y_axis = list(self.generators_p[gen.name])
                idx = plot_order.index(gen.name)
                plt.fill_between(x_axis, y_axis, color=gas_gen_colors[index], label=gen.name, zorder=idx+1)

            # PLOT WIND GENERATORS
            wind_generators = [gen for gen in ordered_generators if gen.carrier == 'wind']
            for index, gen in enumerate(wind_generators):
                y_axis = list(self.generators_p[gen.name])
                idx = plot_order.index(gen.name)
                plt.fill_between(x_axis, y_axis, color=wind_gen_colors[index], label=gen.name, zorder=idx+1)

            # SET CONFIGS
            plt.legend(loc='upper right')

            plt.title('LGridPy\nUnit commitment and optimal dispatch')
            plt.xlabel('Snapshots [h]')
            plt.ylabel('Dispatch [MW]')            

            if save_plot:
                print('Saving plots results...\n')
                plt.savefig(filename, dpi=plot_dpi)

            print('Showing plots...\n')
            plt.show()
            plt.close()
            print('Plotting finished')

        else:
            print('Network is not solved.\n')