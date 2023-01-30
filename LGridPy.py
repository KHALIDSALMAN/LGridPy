from network import *
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Reading data from sheets
# dem_load=pd.read_excel(r'input_data/Elisabetta load P.U.xlsx')
# load_data=dem_load['Y'][560:570]
# load_data=dem_load['Y']

used_rows = list(range(100))

dem_load = pd.read_csv(r'input_data/525600 mins - Elisbetta load.csv', header=None, index_col=0, skiprows=used_rows[0], nrows=len(used_rows))
load_data = np.asarray(dem_load).T[0]

lm2500_efcurve = pd.read_excel('input_data/ef_curve_lm2500.xlsx')[['0-xaxis', '0-yaxis']]
# ef25= pd.read_excel('input_data/constant_efficiency.xlsx', sheet_name='1')
# ef50= pd.read_excel('input_data/constant_efficiency.xlsx', sheet_name='2')
# ef75= pd.read_excel('input_data/constant_efficiency.xlsx', sheet_name='3')
# ef100= pd.read_excel('input_data/constant_efficiency.xlsx', sheet_name='4')

# ef25 = pd.read_excel('input_data/constant_efficiency.xlsx')


network = Network(name='My Network',
                  frequency=60, # Hz
                  rocof_limit=-2.0, # Hz/s
                  contingency_frequency=54, # Hz
                  timebase='minutes'
                  )

increase = 0        # Percentage increase

load = list((1+increase/100) * 85*load_data)

# load = network.step_load(nominal_load=25, number_of_steps=4)

network.add_load(load)

network.add_gas_generator('GT1',
                        p_nom=25.,
                        p_min_pu=0.4,
                        p_max_pu=0.9,
                        min_uptime=10, # minutes 1/6 hours
                        min_downtime=30, # minutes 1/2 hours
                        ramp_up_limit=0.12,
                        ramp_down_limit=0.12,
                        start_up_cost=70.,
                        shut_down_cost=70.,
                        fuel_price=10.,
                        efficiency_curve=lm2500_efcurve,
                        constant_efficiency=False,
                        inertia_constant=3.2
                        )

network.add_gas_generator('GT2',
                        p_nom=25.,
                        p_min_pu=0.4,
                        p_max_pu=0.9,
                        min_uptime=10, # minutes 1/6 hours
                        min_downtime=30, # minutes 1/2 hours
                        ramp_up_limit=0.12,
                        ramp_down_limit=0.12,
                        start_up_cost=70.,
                        shut_down_cost=70.,
                        fuel_price=10.,
                        efficiency_curve=lm2500_efcurve,
                        constant_efficiency=False,
                        inertia_constant=3.2
                        )

network.add_gas_generator('GT3',
                        p_nom=25.,
                        p_min_pu=0.4,
                        p_max_pu=0.9,
                        min_uptime=10, # minutes 1/6 hours
                        min_downtime=30, # minutes 1/2 hours
                        ramp_up_limit=0.12,
                        ramp_down_limit=0.12,
                        start_up_cost=70.,
                        shut_down_cost=70.,
                        fuel_price=10.,
                        efficiency_curve=lm2500_efcurve,
                        constant_efficiency=False,
                        inertia_constant=3.2
                        )

network.add_gas_generator('GT4',
                        p_nom=25.,
                        p_min_pu=0.4,
                        p_max_pu=0.9,
                        min_uptime=10, # minutes 1/6 hours
                        min_downtime=30, # minutes 1/2 hours
                        ramp_up_limit=0.12,
                        ramp_down_limit=0.12,
                        start_up_cost=70.,
                        shut_down_cost=70.,
                        fuel_price=10.,
                        efficiency_curve=lm2500_efcurve,
                        constant_efficiency=False,
                        inertia_constant=3.2
                        )


# network.plot_efficiency_curve('GT1')

wind_speed_data = pd.read_csv('input_data/150m one year wind speed.txt', header=None)

# Average Wind Speed
# mean_wind_speed = np.mean(wind_speed_data)
# wind_speed = np.ones(len(load))
# wind_speed *= mean_wind_speed[0]
# wind_speed *= 15

# Variable Wind Speed
wind_speed = np.array(wind_speed_data[:len(load)]).T[0]

wind_penetration = 0.35

# network.add_wind_generator('WT1',
#                             p_nom=15,
#                             number_of_turbines=2,
#                             wind_speed_array=wind_speed,
#                             wind_penetration=wind_penetration,
#                             # electromechanical_conversion_efficiency=0.965
#                             electromechanical_conversion_efficiency=1.
#                             )

# network.add_wind_generator('WT2',
#                             p_nom=15,
#                             number_of_turbines=1,
#                             wind_speed_array=wind_speed,
#                             wind_penetration=wind_penetration,
#                             # electromechanical_conversion_efficiency=0.965
#                             electromechanical_conversion_efficiency=1.
#                             )


# network.add_storage('ST1',
#                     p_nom=1,
#                     nom_capacity_MWh=1,
#                     min_capacity=0.2,
#                     stand_efficiency=1 - 0.02/30/24,
#                     discharge_efficiency=0.92,
#                     initial_state_of_charge=1.,
#                     cyclic_state_of_charge=True
#                     # final_state_of_charge=0.0
#                     ) 

# network.add_storage('ST2',
#                     p_nom=1,
#                     nom_capacity_MWh=1,
#                     min_capacity=0.2,
#                     stand_efficiency=1 - 0.02/30/24,
#                     discharge_efficiency=0.92,
#                     initial_state_of_charge=1.,
#                     cyclic_state_of_charge=True
#                     # final_state_of_charge=0.0
#                     ) 

network.solve(show_complete_info=False)

# network.model.display()

# For high quality images, use plot_dpi = 2000
# For quick simulation times, use plot_dpi = 400
network.plot_results(display_plot=True,
                    save_plot=False, 
                    plot_dpi=400, 
                    gas_gen_colors=list(['orangered', 'purple', 'lightcoral', 'black']),
                    st_colors=list(['#7d0e79', 'r'])
                    )

network.export_results_to_xlsx('minutes2.xlsx', 
                            include_status=True, 
                            include_means=True, 
                            compact_format=True
                            )
            
del network