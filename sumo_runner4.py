import time
import os
import pandas as pd
import numpy as np
import dynamita.scheduler as ds
import dynamita.tool as dtool

state_filename = "temp_state1.xml"
filepath_csv = "model_output_1.csv"

def run_simulation(influent_input, do_setpoint, reset_state):

    # load influent parameters to respective SUMO parameter
    influent_input = np.round(np.array(influent_input, dtype=np.float32), 3)
    print(influent_input)

    # transform action to DOsp and SUMO-acceptable format
    do_setpoint = np.round(np.array(do_setpoint, dtype=np.float32), 3)

    sumo_names = ["Sumo__Plant__Influent1__param__T", "Sumo__Plant__Influent1__param__TCOD",
                  "Sumo__Plant__Influent1__param__TKN", "Sumo__Plant__Influent1__param__TP",
                  "Sumo__Plant__Influent1__param__Q", "Sumo__Plant__CSTR4__param__DOSP",
                  "Sumo__Plant__CSTR5__param__DOSP"]

    # Reshape the arrays to be a single row
    influent_input = influent_input.reshape(1, -1)
    do_setpoint = do_setpoint.reshape(1, -1)

    # Create a DataFrame from the reshaped arrays
    temp_input_df = pd.DataFrame(np.concatenate((influent_input, do_setpoint), axis=1), columns=sumo_names)

    # Add a new column at the first position
    temp_input_df.insert(0, "Sumo__Time", 0)

    # Export the DataFrame as a TSV file
    temp_input_df.to_csv("input.tsv", sep="\t", index=False)


    model = "sumoproject.dll"

    if reset_state == True:
        state = 'init.xml'
        print('state was reset')
    else:
        state = state_filename
        print('prior state was loaded')

    # Set up callbacks for Sumo to call
    ds.sumo.message_callback = msg_callback
    ds.sumo.datacomm_callback = data_callback

    # Set the number of parallel simulations (6 cores, so 6 runs)
    ds.sumo.setParallelJobs(6)

    # Schedule the Sumo job
    job = ds.sumo.schedule(model,
        commands=[
            f"load {state}",
            f"loadtsv input.tsv;",
            f"set Sumo__StopTime {1 * dtool.hour};",
            f"set Sumo__DataComm {1 * dtool.hour};",
            "mode dynamic;",
            "start;",
        ],
        variables=[
            "Sumo__Time",
            "Sumo__Plant__EnergyCenter__Pel_aeration",
            "Sumo__Plant__CFP__dEmoffgas_GN2O_mainstream_dt",
            "Sumo__Plant__Effluent1__TN",
            "Sumo__Plant__Effluent1__TP",
            "Sumo__Plant__Effluent1__TCOD",
            "Sumo__Plant__CSTR3__SN2O",
            "Sumo__Plant__CSTR5__SN2O",
            "Sumo__Plant__CSTR4__SO2",
            "Sumo__Plant__CSTR5__SO2"
        ],
        jobData={
            "finished": False,
        },
    )
    jobData = ds.sumo.getJobData(job)
    while(not jobData["finished"]):
        time.sleep(0.1)

def data_callback(job, data):
    #save the current data as last data (this will get overwritten in all datacomm)
    jobData = ds.sumo.getJobData(job)
    jobData["last_data"] = data

def msg_callback(job, msg):
    # In case of simulation finished sumocore message and end simulation
    print("MSG #" + str(job) + ": '" + msg + "'")
    jobData = ds.sumo.getJobData(job)
    if (ds.sumo.isSimFinishedMsg(msg)):
        jobData["finished"] = True
        
        # save last data
        save_data_to_csv(jobData["last_data"])
        ds.sumo.sendCommand(job, f"save {state_filename}")

    if msg.startswith("530045") and jobData["finished"]:
        ds.sumo.finish(job)

def save_data_to_csv(data):
    # Create a DataFrame with the data
    df = pd.DataFrame([data])

    # Drop the column with name "Sumo__time"
    df = df.drop(columns=["Sumo__Time"])

    # Write the DataFrame to CSV, overwrite existing file
    df.to_csv(filepath_csv, mode="w", index=False)

