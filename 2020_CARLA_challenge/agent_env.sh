#!/bin/bash
source /auto/bin/docker_env.sh
python3 -m pip install ephem
python3 -m pip install tabulate 
python3 -m pip install -r  ./carla_project/requirements.txt
python3 -m pip install pytorchfi
export PORT=2000                                                    # change to port that CARLA is running on
export ROUTES=leaderboard/data/routes_testing/route_00.xml  # change to desired route
export TEAM_AGENT=image_agent.py                                    # no need to change
# export TEAM_AGENT=auto_pilot.py
export TEAM_CONFIG=epoch24.ckpt                                      # change path to checkpoint
# export TEAM_CONFIG=sample_date
export HAS_DISPLAY=1

