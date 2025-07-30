### Run demos
- Make sure your PC has one GPU at least
- Enter your conda env
- Install dependencies by: `pip install -e`

#### Use RC control FPV in Genesis
- Flash HEX file in `./modified_BF_firmware/betaflight_4.4.0_STM32H743_forRC.hex` to your FCU (for STM32H743)
- Use Type-c to power the FCU, and connect UART port (on FCU) and USB port (on PC) through USB2TTL module, like:
- <img src="./docs/1.png"  width="300" /> <br>
- Connect the FC and use mavlink to send FC_data from FCU to PC
- Use `ls /dev/tty*` to check the port id and modified param `USB_path` in `./config/flight.yaml`
- Do this since the default mavlink frequence for rc_channle is too low
- Connect the FC and use mavlink to send FC_data from FCU to PC
- Run demo by `python main.py`
- Use FC to control the sim drone by :
    ```
    python scripts/eval/back2nt_eval.py
    ```

#### An tracking task
- Train the model
    ```
    python scripts/train/track_rsl.py 
    ```
#### An obstacle avoidance task
- Evaluate the exsisting model
    ```
    python scripts/eval/back2nt_eval.py
    ``` 

### Reference
- [gazebo-vegetation](https://github.com/kubja/gazebo-vegetation)
- [OmniPerception](https://github.com/aCodeDog/OmniPerception)
- [rsl-rl](https://github.com/leggedrobotics/rsl_rl.git)
- [stable-baselines3](https://github.com/DLR-RM/stable-baselines3.git)
- [Back to Newton](https://github.com/HenryHuYu/DiffPhysDrone)