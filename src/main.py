
import imu_sim
import mavlink
import time


def env_init():

    # TODO 增加场景初始化代码

    imu_sim = imu_sim.IMU_sim(
        entity = drone, 
        env_num = 1,
        yaml_path = "config/IMU_sim_param"
    ) 

if __name__ == "__main__":

    mavlink.start_mavlink_receive_thread()
    while True:
        time.sleep(0.1)
