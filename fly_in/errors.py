"""Custom exceptions for the Fly-in project."""


class FlyInError(Exception):
    """Base class for all Fly-in errors."""


class ParseError(FlyInError):
    """Raised when a map file cannot be parsed."""


class SimulationError(FlyInError):
    """Raised when the simulation reaches an invalid state."""
