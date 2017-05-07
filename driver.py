"""Driver for constrained attitude control

"""
import matplotlib.pyplot as plt
import spacecraft

sc = spacecraft.SpaceCraft()
# simulate the rigid body
sc.integrate(10)

# parse out the simulation results and recompute the error functions for 
# each state value

fig, ax = plt.subplots(1,1)
ax.plot(sc.time, sc.state[:,-1])
plt.show()
