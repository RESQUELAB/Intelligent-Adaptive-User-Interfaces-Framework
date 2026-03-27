import os
import json
from enum import Enum, auto
import numpy as np
from ui_adapt.envs.user import User
from ui_adapt.envs.environment import Environment
from ui_adapt.envs.platform_ import Platform
from ui_adapt.envs.uidesign import UIDesign

class Config():

    def __init__(self, config_data="config.json") -> None:

        if isinstance(config_data, str) and config_data.endswith(".json"):
            current_folder = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_folder, config_data)
            self.config = self.read_config(config_path)
        else:
            self.config = config_data

        # Create Enums for USER, PLATFORM, ENVIRONMENT, and UIDESIGN
        self.user_enums = self.create_enums_from_config(
            self.config["USER"])
        self.platform_enums = self.create_enums_from_config(
            self.config["PLATFORM"])
        self.environment_enums = self.create_enums_from_config(
            self.config["ENVIRONMENT"])
        self.uidesign_enums = self.create_enums_from_config(
            self.config["UIDESIGN"])
        if "PREFERENCES" in self.config:
            self.user_preferences = self.config["PREFERENCES"]
        # Get the actions from the config file
        self.actions = self.config["ACTIONS"]
        # print("\tI'm in the UIADAPTATION CONFIG SETUP")
        # print(self.config)

        if "REWARD" in self.config:
            self.reward_mode = self.config["REWARD"]["MODE"]
        # If using the API MODE, to connect to other apps,
        # Set API_CONNECTION in your config file.
        if "API_CONNECTION" in self.config:
            self.api_connection = self.config["API_CONNECTION"]

    def create_enum_class(self, enum_dict):
        return Enum(enum_dict['name'], {value: auto() for value in enum_dict['values']})

    def create_enums_from_config(self, config):
        enums = {}
        for enum_name, enum_values in config.items():
            enums[enum_name] = self.create_enum_class({"name": enum_name, "values": enum_values})
        return enums

    def read_config(self, file_path):
        with open(file_path, 'r') as file:
            config = json.load(file)
        return config

def get_combinations(aspects, enum_type, prev_combinations=[]):
    combinations = prev_combinations.copy()
    for aspect in aspects:
        possible_values = [elem.name for elem in enum_type[aspect]]
        combinations.append(len(possible_values))
    return combinations

def get_actions(config):
    actions = {}
    for action_id in config.actions:
        if action_id.upper() == "MODE":
            continue
        actions[int(action_id)] = config.actions[action_id]
    sorted_ids = sorted(list(actions))
    if sorted_ids != list(range(min(sorted_ids), max(sorted_ids) + 1)):
        raise ValueError(
            "Action IDs in the config file must be consecutive and incrementally ordered."
            )
    return actions

def get_random_aspect(enum_class, aspect_name):
    
    if aspect_name in enum_class:
        possible_values = enum_class[aspect_name]
        if len(possible_values) == 1:
            return possible_values(1).name
        random = np.random.choice(enum_class[aspect_name])
        return random.name
    else:
        print(f"Aspect '{aspect_name}' not found in the config.")
        return None

def get_random_ui(config, uiDesignInfo= None, aspects=[], mode='human', client_ws=None,server_ws=None):
    if not aspects:
        aspects = [aspect for aspect in config.uidesign_enums]
    user_interface = {}
    for aspect in aspects:
        if uiDesignInfo:
            user_interface[aspect.lower()] = uiDesignInfo["UIDESIGN"][aspect.lower()]
        else:
            user_interface[aspect.lower()] = get_random_aspect(config.uidesign_enums ,aspect)
    combinations = get_combinations(aspects, config.uidesign_enums)
    if mode != 'human':
        return user_interface, combinations
    return UIDesign(config, user_interface, combinations, client_socket=client_ws, server_socket=server_ws)


def get_random_user(config, userinfo= None, aspects=[]):
    if not aspects:
        aspects = [aspect for aspect in config.user_enums]
    user = {}
    for aspect in aspects:
        if userinfo:
            user[aspect.lower()] = userinfo["USER"][aspect.lower()]
        else:
            user[aspect] = get_random_aspect(config.user_enums ,aspect)
    # preferences, combinations_pref = get_random_ui(config, mode='user_preferences')
    # combinations = get_combinations(aspects, config.user_enums, combinations_pref)
    combinations = get_combinations(aspects, config.user_enums)
    return User(config, user, {}, combinations)
    return User(config, user, preferences, combinations)


def get_random_environment(config, envinfo= None, aspects=[]):
    if not aspects:
        aspects = [aspect for aspect in config.environment_enums]
    environment = {}
    for aspect in aspects:
        if envinfo:
            environment[aspect.lower()] = envinfo["ENVIRONMENT"][aspect.lower()]
        else:
            environment[aspect] = get_random_aspect(config.environment_enums ,aspect)
    combinations = get_combinations(aspects, config.environment_enums)
    return Environment(config, environment, combinations)

def get_random_platform(config, platforminfo=None, aspects=[]):
    if not aspects:
        aspects = [aspect for aspect in config.platform_enums]
    platform = {}
    for aspect in aspects:
        if platforminfo:
            platform[aspect.lower()] = platforminfo["PLATFORM"][aspect.lower()]
        else:
            platform[aspect] = get_random_aspect(config.platform_enums ,aspect)
    combinations = get_combinations(aspects, config.platform_enums)
    return Platform(config, platform, combinations)

class StopImage():
    
    def __init__(self) -> None:
        pass
    