import yaml
from lmarspy.fps.data import FpsData

class Flag():

    def __init__(self, fps: FpsData, file_name=None, yaml_data=None):

        if file_name != None and yaml_data != None:
            fps.raise_error(f' Must choose one of file_name and yaml_data. file_name != None and yaml_data != None')
             
        if file_name != None:
            with open(file_name, "r") as file:
                data = yaml.safe_load(file)

        elif yaml_data != None:
            data = yaml_data
        
        else:
            data = {}
                
        data0 = {"times":{   "YYYY": 2000, "MM": 1, "DD": 1,
                            "days": 0, "hours": 0, "minutes": 0, "seconds": 1080,
                            "dt_atmos": 1}, 
                "dims" :{   "global_npx": 201, "global_npy": 2, "npz": 300, "ng": 3},
                "fpses"  :{   "px":1, "py":1},
                "ics"  :{   "ic_type": 21},
                "ios"  :{   "do_output": True, "nc_parallel": True, "out_fre": 10, "do_diag": True},
                "dyns" :{   "k_split": 5, "n_split": 25, "rk": 333, 
                            "dyn_core": "eul", "lim": False, "lim_deg": 0.6}}
            
        for name1, value1 in data0.items():
            fps.main_print(f"    {name1}:")
            for name2, value2 in value1.items():
                if name1 in data and name2 in data[name1]:
                    setattr(self, name2, data[name1][name2])
                    fps.main_print(f"        {name2}: {data[name1][name2]} (input)")
                else:
                    setattr(self, name2, value2)
                    fps.main_print(f"        {name2}: {value2} (default)")

        return
