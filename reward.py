import random

'''
[0] Sumo__Plant__Influent1__param__T
[1] Sumo__Plant__Influent1__param__TCOD
[2] Sumo__Plant__Influent1__param__TKN
[3] Sumo__Plant__Influent1__param__TP
[4] Sumo__Plant__Influent1__param__Q
[5] Sumo__Plant__EnergyCenter__Pel_aeration
[6] Sumo__Plant__CFP__dEmoffgas_GN2O_mainstream_dt
[7] Sumo__Plant__Effluent1__TN
[8] Sumo__Plant__Effluent1__TP
[9] Sumo__Plant__Effluent1__TCOD
[10] Sumo__Plant__CSTR3__SN2O
[11] Sumo__Plant__CSTR5__SN2O
[12] Sumo__Plant__CSTR4__SO2
[13] Sumo__Plant__CSTR5__SO2
'''


def calculate_reward(observation):

    AE = observation[5] * 0.25 # from kW to kWh 
    CO2_f_e = 0.337 #kg CO2 eq / kWh
    CO2_em_AE = CO2_f_e * AE # CO2 eq
    CO2_f_n = 265 # kg CO2eq per kg N2O
    CO2_em_N2O = (observation[6] / 96) * CO2_f_n # from kg N2O/d to kg N2O / 1h
    total_CO2_em = CO2_em_AE + CO2_em_N2O

    TN = observation[7]
    TP = observation[8]
    TCOD = observation[9]
    DO1 = observation[12]
    DO2 = observation[13]

    # Color codes
    COLOR_RED = '\033[91m'
    COLOR_GREEN = '\033[92m'
    COLOR_YELLOW = '\033[93m'
    COLOR_RESET = '\033[0m'

    print(COLOR_YELLOW + 'PRINTING STATS:' + COLOR_RESET)

    if TN > 10 or TP > 1 or TCOD > 60:
        p = 0
        print(COLOR_RED + 'EFFLUENT QUALITY PENALTY!' + COLOR_RESET)

    elif TN > 8:
        p = 50
        print(COLOR_YELLOW + 'N DANGER ZONE (TN > 8.0)' + COLOR_RESET)

    else:
        p = 100


    reward = (p - total_CO2_em)

    reward = float(reward)

    # adjust random print frequency (1 == 100% change)
    if random.randint(1, 1) == 1:

        # Print statements with color
        print('CO2-eq from Aeration / Emission:', COLOR_GREEN + str(round(CO2_em_AE, 2)) + COLOR_RESET, '/', COLOR_GREEN + str(round(CO2_em_N2O, 2)) + COLOR_RESET)
        print('Current DO concentration (1 / 2):', COLOR_GREEN + str(round(DO1, 2)) + COLOR_RESET, '/', COLOR_GREEN + str(round(DO2, 2)) + COLOR_RESET)
        print('Current effluent concentration (TN / TP / TCOD):', COLOR_GREEN + str(round(TN, 2)) + COLOR_RESET, '/', COLOR_GREEN + str(round(TP, 2)) + COLOR_RESET, '/', COLOR_GREEN + str(round(TCOD, 2)) + COLOR_RESET)
    
    
    print('reward:',COLOR_GREEN + str(round(reward, 2)) + COLOR_RESET)



    return reward