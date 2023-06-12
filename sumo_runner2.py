import time
import pandas as pd
import numpy as np
import dynamita.scheduler as ds
import dynamita.tool as dtool

class SumoSimulation:
    def __init__(self, influent_input, do_setpoint, n_run):
        self.influent_input = influent_input
        self.do_setpoint = do_setpoint
        self.n_run = n_run
        self.model = "ZDAM.dll"
        self.initial_state = "init_steadystate.xml"

        # Set up callbacks for Sumo to call
        ds.sumo.message_callback = self.msg_callback
        ds.sumo.datacomm_callback = self.data_callback

        # Set the number of parallel simulations (6 cores, so 6 runs)
        ds.sumo.setParallelJobs(6)

        # Schedule the Sumo job
        self.job = ds.sumo.schedule(
            self.model,
            commands=[
                f"load {self.initial_state}",
                f"set Sumo__StopTime {1 * dtool.hour};",
                f"set Sumo__DataComm {5 * dtool.minute};",
                "mode dynamic;",
            ],
            variables=[
                "Sumo__Time",
                "Sumo__Plant__EnergyCenter__Pel_aeration",
                "Sumo__Plant__CFP__CFPoffgas_GN2O_mainstream",
                "Sumo__Plant__Effluent1__TN",
                "Sumo__Plant__Effluent1__TP",
                "Sumo__Plant__Effluent1__TCOD",
            ],
            jobData={
                ds.sumo.persistent: True,
                "results": {},
            },
        )

    def run_simulation(self):
        # load influent parameters to respective SUMO parameter
        influent_input = np.array(self.influent_input, dtype=np.float32)
        ds.sumo.sendCommand(self.job, f"set Sumo__Plant__Influent1__param__T {influent_input[0]}")
        ds.sumo.sendCommand(self.job, f"set Sumo__Plant__Influent1__param__TCOD {influent_input[1]}")
        ds.sumo.sendCommand(self.job, f"set Sumo__Plant__Influent1__param__TKN {influent_input[2]}")
        ds.sumo.sendCommand(self.job, f"set Sumo__Plant__Influent1__param__TP {influent_input[3]}")
        ds.sumo.sendCommand(self.job, f"set Sumo__Plant__Influent1__param__Q {influent_input[4]}")

        # transform action to DOsp and SUMO-acceptable format
        do_setpoint = np.array(self.do_setpoint, dtype=np.float32)
        ds.sumo.sendCommand(self.job, f"set Sumo__Plant__CSTR4__param__DOSP {do_setpoint}")
        ds.sumo.sendCommand(self.job, f"set Sumo__Plant__CSTR5__param__DOSP {do_setpoint}")

        # start run
        ds.sumo.sendCommand(self.job, "start")

    def data_callback(self, job, data):
        # Check Sumo timestep
        t = data["Sumo__Time"]
        
        if t == dtool.hour:
            self.save_data_to_csv(data)
            print('saving output...')
            time.sleep(0.3)

    def msg_callback(self, job, msg):
        print("MSG #" + str(job) + ": '" + msg + "'")
        if ds.sumo.isSimFinishedMsg(msg):
            ds.sumo.finish(job)

    def save_data_to_csv(self, data):
        filepath = f"model_output_{self.n_run}.csv"

        # Create a DataFrame with the data
        df = pd.DataFrame([data])

        # Drop the column with name "Sumo__time"
        df = df.drop(columns=["Sumo__Time"])

        # Write the DataFrame to CSV, overwrite existing file
        df.to_csv(filepath, mode="w", index=False)

