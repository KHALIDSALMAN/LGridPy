from network import *
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
    
# Reading data from sheets
dem_load=pd.read_excel(r'input_data/Elisabetta load P.U.xlsx')
load_data=dem_load['Y'][140:644] # three weeks of data

# load_data = (60+4*33.3)/49 * np.asarray(range(50))

lm2500_efcurve = pd.read_excel('input_data/ef_curve_lm2500.xlsx')

network = Network(name='My Network')

# load = list(90*load_data)
# load = load_data

load = network.step_load(nominal_load=25, number_of_steps=4)

network.add_load(load)

network.add_gas_generator('GT1',
                        p_nom=33.4,
                        p_min_pu=0.,
                        p_max_pu=1.,
                        min_uptime=0., # 10 min
                        min_downtime=0., # 30 min
                        ramp_up_limit=0.75,
                        ramp_down_limit=0.75,
                        start_up_cost=70.,
                        shut_down_cost=70.,
                        fuel_price=1.,
                        efficiency_curve=lm2500_efcurve
                        )

network.add_gas_generator('GT2',
                        p_nom=33.4,
                        p_min_pu=0.,
                        p_max_pu=1.,
                        min_uptime=0., # 10 min
                        min_downtime=0., # 30 min
                        ramp_up_limit=0.75,
                        ramp_down_limit=0.75,
                        start_up_cost=70.,
                        shut_down_cost=70.,
                        fuel_price=1.,
                        efficiency_curve=lm2500_efcurve
                        )

network.add_gas_generator('GT3',
                        p_nom=33.4,
                        p_min_pu=0.,
                        p_max_pu=1.,
                        min_uptime=0., # 10 min
                        min_downtime=0., # 30 min
                        ramp_up_limit=0.75,
                        ramp_down_limit=0.75,
                        start_up_cost=70.,
                        shut_down_cost=70.,
                        fuel_price=1.,
                        efficiency_curve=lm2500_efcurve
                        )

# network.add_gas_generator('GT4',
#                         p_nom=33.3,
#                         p_min_pu=0.2,
#                         p_max_pu=0.7,
#                         min_uptime=1/6, # 10 min
#                         min_downtime=0.5, # 30 min
#                         ramp_up_limit=0.15,
#                         ramp_down_limit=0.15,
#                         start_up_cost=0.,
#                         shut_down_cost=0.,
#                         fuel_price=1.,
#                         efficiency_curve=lm2500_efcurve
#                         )

wind_speed_data = pd.read_csv('input_data/150m one year wind speed.txt', header=None)

# Average Wind Speed
# mean_wind_speed = np.mean(wind_speed_data)
# wind_speed = np.ones(len(load))
# wind_speed *= mean_wind_speed[0]
# wind_speed *= 15

# Variable Wind Speed
wind_speed = np.array(wind_speed_data[:len(load)]).T[0]

wind_penetration = .35

# network.add_wind_generator('WT1',
#                             p_nom=15,
#                             number_of_turbines=4,
#                             wind_speed_array=wind_speed,
#                             wind_penetration=wind_penetration,
#                             electromechanical_conversion_efficiency=0.965
#                             # electromechanical_conversion_efficiency=1.
#                             )

round_trip_efficiencies = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
capacities = [1., 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# network.add_storage('S1',
#                     p_nom=100,
#                     nom_capacity_MWh=capacities[9]*3*25*(1+2+3+4),
#                     min_capacity=0.,
#                     # stand_efficiency=1 - 0.02/30/24,
#                     stand_efficiency=1.,
#                     # discharge_efficiency=0.92,
#                     discharge_efficiency=1.,
#                     initial_state_of_charge=1.,
#                     cyclic_state_of_charge=False
#                     )

# network.add_storage('S2',
#                     p_nom=5,
#                     nom_capacity_MWh=5,
#                     min_capacity=0.2,
#                     stand_efficiency=1 - 0.02/30/24,
#                     discharge_efficiency=0.92,
#                     initial_state_of_charge=1,
#                     cyclic_state_of_charge=True
#                     )

network.solve(show_complete_info=False)

# For high quality images, use plot_dpi = 2000
# For quick simulation times, use plot_dpi = 400
network.plot_results(display_plot=True,
                    save_plot=True, 
                    plot_dpi=400, 
                    gas_gen_colors=list(['orangered', 'purple', 'lightcoral', 'black']),
                    st_colors=list(['#7d0e79', 'r'])
                    )

network.export_results_to_xlsx('Test2.xlsx', 
                            include_status=True, 
                            include_means=True, 
                            compact_format=True
                            )
            
del network