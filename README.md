### Run demos
- make sure your PC has one GPU at least
- enter your conda env
- install dependencies by: `pip install -e`

#### Use RC control FPV in Genesis
- flash HEX file in `./modified_BF_firmware/betaflight_4.4.0_STM32H743_forRC.hex` to your FCU (for STM32H743)
- use Type-c to power the FCU, and connect UART port (on FCU) and USB port (on PC) through USB2TTL module, like:
- <img src="./doc/1.png"  width="300" /> <br>
- connect the FC and use mavlink to send FC_data from FCU to PC
- use `ls /dev/tty*` to check the port id and modified param `USB_path` in `./config/flight.yaml`
- do this since the default mavlink frequence for rc_channle is too low
- connect the FC and use mavlink to send FC_data from FCU to PC
- run demo by `python main.py`
- use FC to control the sim drone by `python scripts/eval/back2nt_eval.py`

#### An tracking RL task
- config in `config/rl_task`
- `python scripts/train/track_rsl.py` 

#### An obstacle avoidance task
- `python scripts/eval/back2nt_eval.py` 

### Reference
- [gazebo-vegetation](https://github.com/kubja/gazebo-vegetation)
- [OmniPerception](https://github.com/aCodeDog/OmniPerception)
- [rsl-rl](https://github.com/leggedrobotics/rsl_rl.git)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3.git)
- [Back to Newton](https://github.com/HenryHuYu/DiffPhysDrone)