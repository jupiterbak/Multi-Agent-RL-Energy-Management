from threading import Thread
import copy
import logging
from datetime import datetime
import time
from math import sin
import sys

from opcua import ua, uamethod, Server
import opcua
from enum import Enum

class EFLEXAgentType(Enum):
    CONSUMER = 1
    PRODUCER = 2
    STORAGE = 3


class EFLEXAgent():
    def __init__(self, eflex_type=EFLEXAgentType.CONSUMER, port=4840, name="EFLEX Agent"):
        self.eflex_type = eflex_type
        self.name = name
        self.port = port
        self.server = Server()
        self.configure_server(self.server)

    def __str__(self):
        return "EFLEX AGENT OPC UA Server: %s" % self.port

    def configure_server(self, server):
        """
        Configure the agent OPC UA Server
        """
        server.set_endpoint("opc.tcp://0.0.0.0:%s/EFlex-Agent" % self.port)
        server.set_server_name("%s" % self.name)
        server.set_security_policy([
            ua.SecurityPolicyType.NoSecurity,
            ua.SecurityPolicyType.Basic256Sha256_SignAndEncrypt,
            ua.SecurityPolicyType.Basic256Sha256_Sign])
        server.set_application_uri("urn:fau:faps:OPCUA-EFLEXAgent:server")
        objects = server.get_objects_node()
        server.import_xml("./IM/Opc.Ua.Di.NodeSet2.xml")
        server.import_xml("./IM/Opc.Ua.Plc.NodeSet2.xml")
        server.import_xml("./IM/packmltest.xml")
        server.import_xml("./IM/packml_eflex_modified.xml")
        uri = "http://faps.fau.de/OPCUA-EFLEXAgent"
        idx = server.register_namespace(uri)
        idx_eflex = server.get_namespace_index("http://siemens.com/PackML_Eflex/")

        # Configure the method calls
        method_node = server.get_node(ua.NodeId(9850), idx_eflex)

    def start(self):
        """
        Start the agent OPC UA Server
        """
        self.server.start()

    def stop(self):
        """
        Stop the agent OPC UA Server
        """
        self.server.stop()

