"""Driver for constrained attitude control

"""
import spacecraft
import plotting

# attitude stabilization without adaptive update law
sc_noadapt = spacecraft.SpaceCraft(
                        scenario_switch='multiple',
                        avoid_switch=True,
                        dist_switch=True,
                        adaptive_switch=False,
                        time_varying_switch=False)

# atttitude stabilization with the adaptive update law
sc_adapt = spacecraft.SpaceCraft(
                        scenario_switch='multiple',
                        avoid_switch=True,
                        dist_switch=True,
                        adaptive_switch=True,
                        time_varying_switch=False)

# attitude stabilization with a time varying disturbance
sc_timevarying = spacecraft.SpaceCraft(
                        scenario_switch='single',
                        avoid_switch=True,
                        dist_switch=True,
                        time_varying_switch=True,
                        adaptive_switch=True)

sc_experiment = spacecraft.SpaceCraft(
                        scenario_switch='single',
                        avoid_switch=True,
                        dist_switch=True,
                        adaptive_switch=True,
                        time_varying_switch=False,
                        experiment_switch=True)
# now integrate and plot them all
sc_noadapt.integrate(10)
# plotting.plot_outputs(sc_noadapt)
