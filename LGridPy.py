from network import *

# Reading data from sheets
dem_load=pd.read_excel(r'input_data/Elisabetta load P.U.xlsx')
# load_data=dem_load['Y'][140:644] # three weeks of data

# load_data = (30 + 2*33.3)/49 * np.asarray(range(50))
load_data = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0, 0, 0, 0, 0, 0, 5, 10, 15]

lm2500_efcurve = pd.read_excel('input_data/ef_curve_lm2500.xlsx')

network = Network(name='My Network')

# load = list(90*load_data)
load = load_data

network.add_load(load)

network.add_gas_generator('GT1',
                        p_nom=33.3,
                        p_min_pu=0.2,
                        p_max_pu=0.7,
                        min_uptime=1/6, # 10 min
                        min_downtime=0.5, # 30 min
                        ramp_up_limit=0.15,
                        ramp_down_limit=0.15,
                        start_up_cost=70.,
                        shut_down_cost=70.,
                        fuel_price=1.,
                        efficiency_curve=lm2500_efcurve
                        )

network.add_gas_generator('GT2',
                        p_nom=33.3,
                        p_min_pu=0.2,
                        p_max_pu=0.7,
                        min_uptime=1/6, # 10 min
                        min_downtime=0.5, # 30 min
                        ramp_up_limit=0.15,
                        ramp_down_limit=0.15,
                        start_up_cost=70.,
                        shut_down_cost=70.,
                        fuel_price=1.,
                        efficiency_curve=lm2500_efcurve
                        )

network.add_gas_generator('GT3',
                        p_nom=33.3,
                        p_min_pu=0.2,
                        p_max_pu=0.7,
                        min_uptime=1/6, # 10 min
                        min_downtime=0.5, # 30 min
                        ramp_up_limit=0.15,
                        ramp_down_limit=0.15,
                        start_up_cost=70.,
                        shut_down_cost=70.,
                        fuel_price=1.,
                        efficiency_curve=lm2500_efcurve
                        )

network.add_gas_generator('GT4',
                        p_nom=33.3,
                        p_min_pu=0.2,
                        p_max_pu=0.7,
                        min_uptime=1/6, # 10 min
                        min_downtime=0.5, # 30 min
                        ramp_up_limit=0.15,
                        ramp_down_limit=0.15,
                        start_up_cost=70.,
                        shut_down_cost=70.,
                        fuel_price=1.,
                        efficiency_curve=lm2500_efcurve
                        )

wind_speed_data = pd.read_csv('input_data/150m one year wind speed.txt', delimiter = "\n", header=None)

# Average Wind Speed
mean_wind_speed = np.mean(wind_speed_data)
wind_speed = np.ones(len(load))
# wind_speed *= mean_wind_speed[0]
wind_speed *= 15

# Variable Wind Speed
# wind_speed = np.array(wind_speed_data[:len(load)]).T[0]

wind_penetration = 0.35

network.add_wind_generator('WT1',
                            p_nom=15,
                            number_of_turbines=2,
                            wind_speed_array=wind_speed,
                            wind_penetration=wind_penetration,
                            # electromechanical_conversion_efficiency=0.965
                            electromechanical_conversion_efficiency=1.
                            )
    
network.add_storage('S1',
                    p_nom=5,
                    nom_capacity_MWh=5,
                    min_capacity=0.2,
                    stand_efficiency=1 - 0.02/30/24,
                    discharge_efficiency=0.92,
                    initial_state_of_charge=1,
                    cyclic_state_of_charge=True
                    )

# network.add_storage('S2',
#                     p_nom=5,
#                     nom_capacity_MWh=5,
#                     min_capacity=0.2,
#                     stand_efficiency=1 - 0.02/30/24,
#                     discharge_efficiency=0.92,
#                     initial_state_of_charge=1,
#                     cyclic_state_of_charge=True
#                     )

# print(network.wind_generators[0].p_max_pu*30)

network.solve(show_complete_info=False)

# network.model.fuel_cost.pprint()
# network.model.fuel_cost_constraint.pprint()

# For high quality images, use plot_dpi = 2000
# For quick simulation times, use plot_dpi = 400
network.plot_results(display_plot=True,
                    save_plot=True, 
                    plot_dpi=400, 
                    gas_gen_colors=list(['orangered', 'purple', 'lightcoral', 'black']),
                    st_colors=list(['#7d0e79', 'r'])
                    )

network.export_results_to_xlsx('results.xlsx', 
                            include_status=True, 
                            include_means=True, 
                            compact_format=True
                            )
            
del network