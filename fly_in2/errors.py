"""Error types shared across the Fly-in pipeline."""


class FlyInError(Exception):
    """Base class for every error the program reports to the user."""


class ParseError(FlyInError):
    """A map file violates the format; the message has line and cause."""


class SimulationError(FlyInError):
    """The map is unrunnable or no valid plan could be produced."""
