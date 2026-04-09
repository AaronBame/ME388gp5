# ME388gp5



Purpose: Design a 1D plane thermal reactor that:

* produces 1 MW/m^2 and can operate without refueling for 15 years
* has a core+reflector+shield that is shorter than the flatbed of a tractor-trailer
* has a total neutron flux that is less than 10^6 outside the shield
* Could be operated remotely or autonomously
* Show how the inherent feedback mechanisms are very strongly negative to enable remote operation and compute what would happen to the reactor if it lost all cooling to the environment. Plot the power distribution, average fuel temperature, and reactivity in the system until it returns to steady state



Overall project requirements:

1. Detailed core design composed of clad pins in assemblies, with a core reflector, and bio-shield
2. Has a mechanism for reactivity control (rods in the core, drums in the reflector, chemical shim, burnable poisons, etc.)
3. Is initially subcritical (keff<0.95) at room temperature when the reactivity controls are inserted
4. Can be defined as critical (keff {0.999,1.001}) when the reactor is at full power (reactivity controls can be anywhere)
5. Has a relatively flat power distribution, where the peak-to-average power distribution (F\_Q) is less than 2.0 (initially)
6. Has an initial isothermal coefficient of reactivity that is negative
7. Can remain critical (keff>1.0) for the entire fuel cycle
8. Can meet the safety/design targets as specified in the specific problem statement
9. Not that control rods in 1D are unreasonable strong, so be prepared

