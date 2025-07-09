import requests
import json
import time

class UIDesign:
    """
    Represents a User Interface design with its attributes.

    Parameters:
    - attributes (dict): A dictionary containing user attributes.

    Example:
    >>> config = Config()
    >>> random_ui = UIDesign(utils.get_random_ui(config))
    >>> print(random_ui)
    """

    def __init__(self, config, attributes, combinations, 
                 server_socket=None, client_socket=None):
        self.config = config
        self.mode = self.config.actions["MODE"]
        if "API" in self.mode and self.config.api_connection:
            host_name = self.config.api_connection["HOST"]
            port = self.config.api_connection["PORT"]
            self.url = host_name + ":" + str(port)
            self.parameters = {'change': ''}
            if self.config.api_connection["RENDER_RESOURCE"]:
                self.render_resource = self.config.api_connection["RENDER_RESOURCE"]
        
        self.attributes = {}
        for aspect_name in attributes:
            value = attributes[aspect_name]
            setattr(self, aspect_name.lower(), value)
            self.attributes[aspect_name.lower()] = value
        self.combinations = combinations

        if server_socket:
            self.server_socket = server_socket
        if client_socket:
            self.client_socket = client_socket
        self.image=None
        self.prev_image = None

    def __str__(self):
        uidesign_str = "UIDesign:\n"
        for aspect_name, aspect_value in self.attributes.items():
            uidesign_str += f"\t{aspect_name}: {aspect_value}\n"

        return uidesign_str

    def update(self, target, value, api_call= ""):
        # Dynamically set the attribute in the UIDesign class based on the target
        # print("UPDATING THE UI")
        target = target.lower()
        value = value.lower()
        err_msg = (
            f"{target.lower()!r} is not an attribute, use: "
            f"{self.attributes}. Check Config file."
        )
        if "pass" not in target:
            assert hasattr(self, target), err_msg
            possible_values = [elem.name for elem in self.config.uidesign_enums[target.upper()]]
            err_msg = f"{value!r} is not a valid value, use: {possible_values}. Check Config file."
            assert value in possible_values, err_msg

        if self.mode == "API":
            assert api_call, "Api call not defined"
            api_call_split = api_call.split(" ")
            assert len(api_call_split) == 2, "api_call shoud have 2 parameters '<Resource> <Value>'"
            # Handle API request to update UI
            api_resource = api_call_split[0]
            api_value    = api_call_split[1]
            api_response = self.make_api_call(api_resource, api_value)
            if api_response != "success":
                err_msg = f"API request failed: {api_response}"
                return err_msg
        if self.mode == "WEBSOCKET":
            msg_type, msg_target, msg_value = api_call.split(" ")
            message = {
                "type": msg_type,
                "target": msg_target,
                "value": msg_value
            }
            message = json.dumps(message)
            self.server_socket.send_message(self.client_socket, message)
        if target == "pass":
            return
        setattr(self, target, value)
        self.attributes[target] = value

    def askForImage(self):
        message = {
            "type": 'image'
        }
        self.server_socket.send_message(self.client_socket, message)


    def updateSockets(self, clientWS, serverWS):
        self.client_socket = clientWS
        self.server_socket = serverWS
    
    def getImage(self, clear=False, getPrevious=False):
        if getPrevious:
            retImage = self.prev_image
        else:
            retImage = self.image
        if clear:
            self.clearImage()
        return retImage
    
    def setImage(self, image):
        self.image = image
    
    def clearImage(self):
        self.prev_image = self.image
        self.image = None

    def make_api_call(self, resource, value):
        full_url = self.url+resource
        self.parameters["change"] = value
        response = requests.get(url = full_url, params = self.parameters)
        return response

    def picture(self, mode=""):
        if mode != "API":
            return
        response = self.make_api_call(self.render_resource, '')
        return response.json()

    def render(self, render_mode='ansii'):
        print(self)

    def get_state(self):
        state = {}
        # Add attributes from 'attributes'
        for aspect_name, aspect_value in self.attributes.items():
            aspect_name_upper = aspect_name.upper()
            value = (self.config.uidesign_enums[aspect_name_upper][aspect_value].value) - 1 
            state[aspect_name] = value
        return {"uidesign": state}
    