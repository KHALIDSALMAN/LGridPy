from network import *

# Reading data from sheets
dem_load=pd.read_excel(r'Elisabetta load P.U.xlsx')
load_data=dem_load['Y'][240:340]

lm2500_efcurve = pd.read_excel('ef_curve_lm2500.xlsx')

network = Network(name='My Network')

load = list(70*load_data)

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
                        fuel_price=100.,
                        efficiency_curve=lm2500_efcurve,
                        efficiency_type='quadratic' # constant/quadratic
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
                        fuel_price=200.,
                        efficiency_curve=lm2500_efcurve,
                        efficiency_type='quadratic' # constant/quadratic
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
                        fuel_price=300.,
                        efficiency_curve=lm2500_efcurve,
                        efficiency_type='quadratic' # constant/quadratic
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
                        fuel_price=400.,
                        efficiency_curve=lm2500_efcurve,
                        efficiency_type='quadratic' # constant/quadratic
                        )

wind_speed_data = pd.read_csv('150m one year wind speed.txt', delimiter = "\n", header=None)

# Average Wind Speed
mean_wind_speed = np.mean(wind_speed_data)
wind_speed = np.ones(len(load))
wind_speed *= mean_wind_speed[0]

# Variable Wind Speed
wind_speed = np.array(wind_speed_data[:len(load)]).T[0]

wind_penetration = 0.35

network.add_wind_generator('WT1',
                            p_nom=15,
                            number_of_turbines=4,
                            wind_speed_array=wind_speed,
                            wind_penetration=wind_penetration,
                            electromechanical_conversion_efficiency=0.965
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

network.solve(preprocessment=False,
              show_complete_info=False
              )

# For high quality images, use plot_dpi = 2000
# For quick simulation times, use plot_dpi = 400
network.plot_results(display_plot=True,
                     save_plot=True, 
                     plot_dpi=400, 
                     gas_gen_colors=list(['orangered', 'purple', 'lightcoral', 'black']),
                     st_colors=list(['#7d0e79'])
                     )

network.export_results_to_xlsx('resultados.xlsx', 
                               include_status=False, 
                               include_means=True, 
                               compact_format=True
                               )
            
del network