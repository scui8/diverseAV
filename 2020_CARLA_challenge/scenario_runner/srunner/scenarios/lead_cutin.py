#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# modified by Shengkun Cui

"""
Ghost Cut In:

The scenario realizes a common driving behavior, in which the
user-controlled ego vehicle follows a lane at constant speed and
an npc suddenly cut into the lane from the left while slowing down.
THe user-controlled ego vechicle should break and stop if necessary
to avoid a crash.
"""

import random

import py_trees

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      KeepVelocity,
                                                                      StopVehicle,
                                                                      WaypointFollower,
                                                                      ChangeAutoPilot,
                                                                      LaneChange)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToVehicle,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               DriveDistance,
                                                                               StandStill)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_waypoint_in_distance


class LeadCutIn(BasicScenario):

    """
    This class holds everything required for a simple "Follow a leading vehicle"
    scenario involving two vehicles.  (Traffic Scenario 2)

    This is a single ego vehicle scenario
    """

    timeout = 120            # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario

        If randomize is True, the scenario parameters are randomized
        """

        self._map = CarlaDataProvider.get_map()
        self._first_vehicle_location = 25
        self._first_vehicle_speed = 22
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._other_actor_max_brake = 1.0
        self._other_actor_stop_in_front_intersection = 20
        self._other_actor_transform = None
        # Timeout of scenario in seconds
        self.timeout = timeout

        super(LeadCutIn, self).__init__("FollowVehicle",
                                                   ego_vehicles,
                                                   config,
                                                   world,
                                                   debug_mode,
                                                   criteria_enable=criteria_enable)

        if randomize:
            self._ego_other_distance_start = random.randint(4, 8)

            # Example code how to randomize start location
            # distance = random.randint(20, 80)
            # new_location, _ = get_location_in_distance(self.ego_vehicles[0], distance)
            # waypoint = CarlaDataProvider.get_map().get_waypoint(new_location)
            # waypoint.transform.location.z += 39
            # self.other_actors[0].set_transform(waypoint.transform)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        first_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        self._other_actor_transform = carla.Transform(
            carla.Location(first_vehicle_waypoint.transform.location.x + 3.2,
                           first_vehicle_waypoint.transform.location.y - 5,
                           first_vehicle_waypoint.transform.location.z + 0.1),
            first_vehicle_waypoint.transform.rotation)
        first_vehicle_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x + 3.2,
                           self._other_actor_transform.location.y - 5,
                           self._other_actor_transform.location.z - 500),
            self._other_actor_transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor('vehicle.audi.a2',
                                                            first_vehicle_transform)
        first_vehicle.set_simulate_physics(enabled=True)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        The scenario defined after is a "follow leading vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then make the other actor to drive until reaching
        the next intersection. Finally, the user-controlled vehicle has to be close
        enough to the other actor to end the scenario.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        # to avoid the other actor blocking traffic, it was spawed elsewhere
        # reset its pose to the required one
        start_transform = ActorTransformSetter(self.other_actors[0], self._other_actor_transform)

        # let the other actor drive and catch up, and perform a dangerous merge lane
        driving_to_next_intersection = py_trees.composites.Parallel("Driving forward and chagne lane",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        driving_to_next_intersection_first = py_trees.composites.Sequence("Start Driving")
        driving_to_next_intersection_first.add_child(InTriggerDistanceToVehicle(self.other_actors[0],
                                                                          self.ego_vehicles[0],
                                                                          distance=50,
                                                                          name="Distance"))
        driving_to_next_intersection_first.add_child(ChangeAutoPilot(self.other_actors[0], True, 
                                                                     parameters={"max_speed": self._first_vehicle_speed}))
        driving_to_next_intersection_first.add_child(KeepVelocity(self.other_actors[0], self._first_vehicle_speed))

        driving_to_next_intersection_second = py_trees.composites.Sequence("Merge Lane")
        driving_to_next_intersection_second.add_child(InTriggerDistanceToVehicle(self.other_actors[0],
                                                                          self.ego_vehicles[0],
                                                                          distance=7,
                                                                          name="Distance"))
        driving_to_next_intersection_second.add_child(LaneChange(self.other_actors[0],
                                                                 direction="right",
                                                                 distance_same_lane=0,
                                                                 distance_other_lane=100,
                                                                 distance_lane_change=15,
                                                                 speed=10))


        # construct scenario
        driving_to_next_intersection.add_child(driving_to_next_intersection_first)
        driving_to_next_intersection.add_child(driving_to_next_intersection_second)

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        endcondition_part = StandStill(self.ego_vehicles[0], name="StandStill", duration=15)
        endcondition.add_child(endcondition_part)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(start_transform)
        sequence.add_child(driving_to_next_intersection)
        sequence.add_child(endcondition)
        sequence.add_child(ActorDestroy(self.other_actors[0]))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
