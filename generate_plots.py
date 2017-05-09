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
# experimental results
sc_experiment = spacecraft.SpaceCraft(
                        scenario_switch='single',
                        avoid_switch=True,
                        dist_switch=True,
                        adaptive_switch=True,
                        time_varying_switch=False,
                        experiment_switch=True)
# now integrate and plot them all
sc_noadapt.integrate(10)
sc_adapt.integrate(10)
sc_timevarying.integrate(10)
sc_experiment.load_experiment()
# now plot everything all awesome like
# input('<Enter> to Plot no adaptive control')
plotting.plot_outputs(sc_noadapt, fname_suffix='noadapt', wscale=0.3333, hscale=0.2, pgf_save=True)
# input('<Enter> to Plot adaptive control')
plotting.plot_outputs(sc_adapt, fname_suffix='adapt', wscale=0.4, hscale=0.2, pgf_save=True)
# input('<Enter> to Plot time varying disturbance')
plotting.plot_outputs(sc_timevarying, fname_suffix='timevarying', wscale=0.3333, hscale=0.2, pgf_save=True)
# input('<Enter> to Plot experimental data')
plotting.plot_outputs(sc_experiment, fname_suffix='exp', wscale=0.3333, hscale=0.2, pgf_save=True)
