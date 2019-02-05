from enum import Enum

from EFLEXAgent.EFLEXAgent import EFLEXAgentType


class EFLEXAgentState(Enum):
    RunningState = 0
    ClearedState = 1
    Aborted = 2
    Aborting = 3
    Stopped = 4
    Clearing = 5
    Stopping = 6
    PowerOff = 7
    PoweringOff = 8
    LoadChange = 9
    StartingUp = 10
    StandBy = 11
    StandingBy = 12
    StartedUp = 13
    Resetting = 14
    Idle = 15
    Starting = 16
    Execute = 17
    Completing = 18
    Completed = 19
    Holding = 20
    Held = 21
    UnHolding = 22
    Suspending = 23
    Suspended = 24
    UnSuspending = 25


class EFLEXAgentTransition(Enum):
    SC = 0
    Abort = 1
    Clear = 2
    Reset = 3
    Stop = 4
    ChangeLoad = 5
    Hold = 6
    PowerOn = 7
    PowerOff = 8
    Standby = 9
    Start = 10
    Suspend = 11
    UnHold = 12
    Unsuspend = 13


class EFLEXAgentStateMachine():
    def __init__(self, eflex_type=EFLEXAgentType.CONSUMER):
        self.eflex_type = eflex_type
        self.machine_load = 0.0

    def __str__(self):
        return "EFLEX State Machine: %s" % self.eflex_type

    def SetTargetLoad(self, load=0.0):
        """
        SetTargetLoad
        """
        self.machine_load = load
