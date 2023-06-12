import numpy as np

'''
add time
[0] Sumo__Plant__Influent1__param__T {influent_input
[1] Sumo__Plant__Influent1__param__TCOD {influent_input
[2] Sumo__Plant__Influent1__param__TKN
[3] Sumo__Plant__Influent1__param__TP
[4] Sumo__Plant__Influent1__param__Q
[5] Sumo__Plant__EnergyCenter__Pel_aeration
[6] Sumo__Plant__CFP__CFPoffgas_GN2O_mainstream
[7] Sumo__Plant__Effluent1__TN
[8] Sumo__Plant__Effluent1__TP
[9] Sumo__Plant__Effluent1__TCOD

'''


def calculate_reward(observation):

    AE = observation[5]
    CO2_f_e = 0.337 #kg CO2 eq / kWh
    CO2_em_AE = CO2_f_e * AE
    N2O_emitted = observation[6] # CO2 eq already in sumo for N2O em
    CO2_f_N2O = 265 # not using right now
    CO2_em_N2O = N2O_emitted  # using the SUMO variable directly now
    total_CO2_em = CO2_em_AE + CO2_em_N2O

    TN = observation[7]
    TP = observation[8]
    TCOD = observation[9]

    if TN < 10 and TP < 1 and TCOD < 50:
        w = 1
    else:
        w = 3
        print('EFFLUENT QUALITY PENALTY!')

    reward = (-1 * w) * total_CO2_em

    reward = float(reward)
    return reward