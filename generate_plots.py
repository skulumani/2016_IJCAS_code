#!/usr/bin/env python
"""Driver for constrained attitude control

"""
import argparse

import spacecraft
import plotting
import log_barrier

def main(pgf_save=False):
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
    input('Press <Enter> to plot the configuration error function visualization')
    log_barrier.plot_error_function(fwidth=0.4, pgf_save=pgf_save)

    input('Press <Enter> to Plot no adaptive control')
    plotting.plot_outputs(sc_noadapt, fname_suffix='noadapt', wscale=0.3333, hscale=0.2, pgf_save=pgf_save)

    input('Press <Enter> to Plot adaptive control')
    plotting.plot_outputs(sc_adapt, fname_suffix='adapt', wscale=0.4, hscale=0.2, pgf_save=pgf_save)

    input('Press <Enter> to Plot time varying disturbance')
    plotting.plot_outputs(sc_timevarying, fname_suffix='timevarying', wscale=0.3333, hscale=0.175, pgf_save=pgf_save)

    input('Press <Enter> to Plot experimental data')
    plotting.plot_outputs(sc_experiment, fname_suffix='exp', wscale=0.3333, hscale=0.2, pgf_save=pgf_save)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate plots for IJCAS Paper titled: Constrained Geometric Attitude Control on SO(3)')
    parser.add_argument('-s', '--save', help="Save the figures to both PDF and PGF formats", 
                        action='store_true')

    args = parser.parse_args()

    if args.save:
        print("Will save figures to disk")
        
    main(args.save)
