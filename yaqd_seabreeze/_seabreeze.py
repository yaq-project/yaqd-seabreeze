__all__ = ["Seabreeze"]

import asyncio
from typing import Dict, Any, List

from seabreeze.spectrometers import Spectrometer  # type: ignore
from yaqd_core import Sensor, logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Seabreeze(Sensor):
    _kind = "seabreeze"
    traits: List[str] = ["is-sensor"]
    defaults: Dict[str, Any] = {}

    def __init__(self, name, config, config_filepath):
        super().__init__(name, config, config_filepath)
        # Perform any unique initialization

        self.spec = Spectrometer.from_serial_number(config.get("serial"))
        self._correct_dark_counts = config.get("correct_dark_counts", False)
        self._correct_nonlinearity = config.get("correct_nonlinearity", False)

        self.channel_names = ["wavelengths", "intensities"]
        self.channel_units = {"wavelengths": "nm", "intensities": None}
        self.channel_shapes = {
            "wavelengths": (self.spec.pixels,),
            "intensities": (self.spec.pixels,),
        }

    def _load_state(self, state):
        """Load an initial state from a dictionary (typically read from the state.toml file).

        Must be tolerant of missing fields, including entirely empty initial states.

        Parameters
        ----------
        state: dict
            The saved state to load.
        """
        super()._load_state(state)
        self._integration_time_micros = state.get("integration_time_micros", 0)

    def get_state(self):
        state = super().get_state()
        state["integration_time_micros"] = self._integration_time_micros
        return state

    async def _measure(self):
        return {
            "wavelengths": self.spec.wavelengths(),
            "intensities": self.spec.intensities(
                self._correct_dark_counts, self._correct_nonlinearity
            ),
        }

    def set_integration_time_micros(self, micros: int) -> None:
        """Set the integration time in microseconds"""
        self._integration_time_micros = micros

    def get_integration_time_micros(self) -> int:
        """Get the integration time in microseconds"""
        return self._integration_time_micros
