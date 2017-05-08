"""Driver for constrained attitude control

"""
import spacecraft
import plotting

# instatiate all the various versions of the simulation
sc = spacecraft.SpaceCraft()
# attitude stabilization without adaptive update law
sc_noadapt = spacecraft.SpaceCraft(
                        scenario='multiple',
                        avoid_switch=True,
                        dist_switch=True,
                        adaptive_switch=False)
# sc_noadapt.integrate(10)

# plotting.plot_outputs(sc_noadapt)

