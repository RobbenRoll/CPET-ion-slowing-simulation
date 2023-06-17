import numpy as np
from scipy.interpolate import interp2d
def apply_ramp_corr(energies, V_nest_eff, species="23Na"):
    """Correct energies for cooling due to endcap-voltage ramp

    V_nest_eff : float
        Effective nest depth w.r.t. ground potential.
    """
    corrected_energies = []
    ramp_corrs = {}
    eff_nest_depths = np.arange(28, 41, 2)
    for V_nest_eff in eff_nest_depths:
        ramp_corr = pd.read_csv("phase-averaged_ion_energies_after_extraction_"+str(species)+"_"+str(V_nest_eff)+"V_eff_nest_depth_0V_RFA_barrier.csv")
        ramp_corr.set_index("Energy before extraction (eV)", inplace=True)

        # Add in data at V_nest eff -0.01 eV / +0.025eV
        for V_nest_eff in eff_nest_depths:
            for E_init in [V_nest_eff - 0.01, V_nest_eff + 0.025]:
                if not E_init in ramp_corr.index.values:
                    E_below = V_nest_eff - (V_nest_eff % 5)
                    E_above = E_below + 5
                    if  E_below in ramp_corr.index.values and E_above in ramp_corr.index.values:
                        interp_row = ramp_corr.loc[E_below] + (ramp_corr.loc[E_above] - ramp_corr.loc[E_below])*(V_nest_eff % 5)/(E_above - E_below)
                    elif E_below not in ramp_corr.index.values:
                        E_below = V_nest_eff - (V_nest_eff % 5) + 0.025
                        E_above = E_below + 4.975
                        interp_row = ramp_corr.loc[E_below] + (ramp_corr.loc[E_above] - ramp_corr.loc[E_below])*(V_nest_eff % 5 - 0.025)/(E_above - E_below)
                    elif E_above not in ramp_corr.index.values:
                        E_below = V_nest_eff - (V_nest_eff % 5)
                        E_above = np.round(E_below + 4.99, 3)
                        interp_row = ramp_corr.loc[E_below] + (ramp_corr.loc[E_above] - ramp_corr.loc[E_below])*(V_nest_eff % 5 + 0.010)/(E_above - E_below)
                    ramp_corr.loc[E_init] = interp_row
                ramp_corr = ramp_corr.sort_index()
        if not 30. in ramp_corr.index.values:
            E_below = 29.99
            E_above = 30.025
            interp_row = ramp_corr.loc[E_below] + (ramp_corr.loc[E_above] - ramp_corr.loc[E_below])*(0.010)/(E_above - E_below)
            ramp_corr.loc[30.] = interp_row
            ramp_corr = ramp_corr.sort_index()
        if not 40. in ramp_corr.index.values:
            E_below = 39.99
            E_above = 40.025
            interp_row = ramp_corr.loc[E_below] + (ramp_corr.loc[E_above] - ramp_corr.loc[E_below])*(0.010)/(E_above - E_below)
            ramp_corr.loc[40.] = interp_row
            ramp_corr = ramp_corr.sort_index()
        corrected_energies.append(ramp_corr["Mean energy after extraction (eV)"])
    corrected_energies = np.array(corrected_energies)

    final_energies = interp2d(ramp_corr.index.values.transpose(),
                              eff_nest_depths, corrected_energies, kind="linear")
    return final_energies(np.array(energies), np.atleast_1d(V_nest_eff))
apply_ramp_corr = np.vectorize(apply_ramp_corr)