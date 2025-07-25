import threading
import time
import yaml
from pymavlink import mavutil
from enum import IntFlag

class MODE(IntFlag):
    ANGLE       = 0
    ANGLE_RATE  = 1
    MIX         = 2
    DISARM      = 3
    ARM         = 4
    MANUAL      = 5
    OFFBOARD    = 6

def load_rc_config(yaml_file):
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)

rc_data_lock = threading.Lock()
rc_data = {
    "ch1": None,
    "ch2": None,
    "ch3": None,
    "ch4": None,
    "ch5": None,
    "ch6": None,
    "ch7": None,
    "ch8": None,
}

rc_command = {
    "roll": 0.0,
    "pitch": 0.0,
    "yaw": 0.0,
    "throttle": 0.0,
    "ANGLE": 1,          # 0: angel, 1: angle_rate, 2: not defined
    "ARM": 0,            # 0: disarmed, 1: armed
    "OFFBOARD": 0,       # 0: manunal,  1: offboard
}

def clamp(min_val, max_val, value):
    return max(min_val, min(value, max_val))

def mavlink_receive_thread(device="/dev/ttyUSB0", baudrate=2000000, rc_config=None):
    connection = mavutil.mavlink_connection(device, baud=baudrate)
    
    print(f"Connected to {device} at baudrate {baudrate}")
    
    while True:
        try:
            msg = connection.recv_match(blocking=True)
            
            if msg is None:
                continue

            if msg.get_type() == "RC_CHANNELS_RAW":
                with rc_data_lock:
                    rc_data.update({        # interval [1000;2000] for THROTTLE and [-500;+500] for ROLL/PITCH/YAW
                        "ch1": msg.chan1_raw,
                        "ch2": msg.chan2_raw,
                        "ch3": msg.chan3_raw,
                        "ch4": msg.chan4_raw,
                        "ch5": msg.chan5_raw,
                        "ch6": msg.chan6_raw,
                        "ch7": msg.chan7_raw,
                        "ch8": msg.chan8_raw,
                    })
                    # print("RC Data:", rc_data)
                    rc_command["roll"] =      clamp(-1, 1, (rc_data[rc_config.get("ROLL", "ch1")] - 1500) / 500)      # scale to range [-1.0, 1.0]
                    rc_command["pitch"] =     clamp(-1, 1, (rc_data[rc_config.get("PITCH", "ch2")] - 1500) / 500)
                    rc_command["yaw"] =       clamp(-1, 1, (rc_data[rc_config.get("YAW", "ch3")] - 1500) / 500)
                    rc_command["throttle"] =  clamp(0, 1, (rc_data[rc_config.get("throttle", "ch4")] - 1000) / 1000)
                    temp_angle = rc_data[rc_config.get("ANGLE", "ch6")]
                    temp_arm = rc_data[rc_config.get("ARM", "ch5")]
                    temp_offboard = rc_data[rc_config.get("OFFBOARD", "ch8")]
                    
                    rc_command["ANGLE"] = 0 if temp_angle < 1400 else 1 if temp_angle <= 1700 else 0  
                    rc_command["ARM"] = 1 if temp_arm > 1400 else 0          
                    rc_command["OFFBOARD"] = 0 if temp_offboard < 1400 else 1     

                    # print("RC Command:", rc_command)

        except KeyboardInterrupt:
            print("Exiting...")
            break


def start_mavlink_receive_thread():
    rc_config = load_rc_config("config/sim_env/flight.yaml")
    t = threading.Thread(target=mavlink_receive_thread, args=(rc_config["USB_path"], rc_config["baudrate"], rc_config), daemon=True)
    t.start()

if __name__ == "__main__":

    start_mavlink_receive_thread()
    while True:
        time.sleep(0.1)
