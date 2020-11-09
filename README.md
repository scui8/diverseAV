# Diverse AV Project
## Learning by Cheating Agent
The code of this project is based on the learning-by-cheating agent 2020 CARLA challenge submission code. Found the repo here `https://github.com/bradyz/2020_CARLA_challenge`. Sample command, `python3 leaderboard/leaderboard/leaderboard_evaluator.py --track=SENSORS --scenarios=leaderboard/data/fi_front_accidient.json --agent=image_agent.py --agent-config=epoch24.ckpt --routes=leaderboard/data/routes_fi/route_highway.xml --port=2000 --trafficManagerSeed=0`

## Setup
Follow `https://github.com/saurabhjha1/DiverseAV` for setup CARLA and learning-by-cheating agent. To setup the custom agent in this folder, merge the content in this folder with the content already in `agents/2020_CARLA_challenge`, do not replace the folder as the current repo needs additional libraries fromt the original repo to run. 

## Dual Agent
The project change the original code of 2020 CARLA challenge and learning-by-cheating official implementation, and add dual agents support that duplicates the ADS agent for better fault tolarence and reliability. To run ADS with two agents, use the flag `--dual_agent` when running the `leaderboard_evaluator.py`.


## Scenarios
The code currently provide several scenarios for fault-injection evaluation. These scenarios are much shorter than the 2020 CARLA Challenge scenarios and more scalable for running experiments. The scenarios supported are
1. Leading vehicle slowing down 
2. Ghost-cut-in from the left
3. Accident in front that involves other cars
4. ...

To run scenarios, choose scenarios under the folder `leaderboard/data/` and look for those start with the `fi` prefix. you need to pass the scenario json file using the scenario flag for example `--scenarios=leaderboard/data/fi_*.json`. When running these scenarios with the `fi` prefix, make sure you run it with `--routes=leaderboard/data/routes_fi/route_highway.xml`, other combinations are not tested.

## Logging
The user can also dump the control output from the ADS(s) for analysis. To dump the output, use `--control_log_path=path/to/dump`, the path should points to a valid folder. The control output will be dumped into a CSV file. To disable logging simply do not set this flag. 
