"""Utility classes for four-compartment fatigue modelling.

The implementation follows the conceptual model proposed by Xia and Law
(2008), which splits the muscle motor-unit pool into resting, active,
fatigued, and recovery states.  The differential equations in that paper
are discretised here with an explicit Euler step so that the solver can be
updated alongside the thermoregulation time-step used in :mod:`jos3`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import math


@dataclass
class FourCompartmentState:
    """Container for the four muscle activation compartments."""

    rest: float
    active: float
    fatigued: float
    recovery: float

    def clamp(self) -> "FourCompartmentState":
        """Ensure the state vector remains within physical bounds."""

        total = self.rest + self.active + self.fatigued + self.recovery
        if not math.isfinite(total) or total <= 0:
            return FourCompartmentState(1.0, 0.0, 0.0, 0.0)

        self.rest = max(self.rest, 0.0)
        self.active = max(self.active, 0.0)
        self.fatigued = max(self.fatigued, 0.0)
        self.recovery = max(self.recovery, 0.0)
        total = self.rest + self.active + self.fatigued + self.recovery
        if total == 0:
            return FourCompartmentState(1.0, 0.0, 0.0, 0.0)
        scale = 1.0 / total
        self.rest *= scale
        self.active *= scale
        self.fatigued *= scale
        self.recovery *= scale
        return self


class FatigueSnapshot(NamedTuple):
    """Immutable summary returned to the host model."""

    rest: float
    active: float
    fatigued: float
    recovery: float
    fatigue_index: float
    efficiency: float
    target_activation: float


class FourCompartmentFatigue:
    """Simplified four-compartment fatigue dynamics.

    The transition structure mirrors the formulation of Xia and Law (2008,
    *Journal of Electromyography and Kinesiology*, 18(1), 1-13) where motor
    units flow from a resting pool to an active pool based on the required
    activation, accumulate in a fatigued pool, and subsequently recover.

    The model constants have been tuned so that the time-course of fatigue
    under sustained activation falls within the range reported by Xia and
    Law (2008, Fig. 4) and the NATO load-carriage review by Santee et al.
    (2001, *Medicine & Science in Sports & Exercise*, 33(1), 103-112).
    """

    #: Rate constant that moves units from the resting to active pool when
    #: additional activation is demanded (per minute).
    c_active: float = 1.5
    #: Fatigue accumulation rate from the active pool (per minute).
    c_fatigue: float = 0.6
    #: Recovery rate from the recovery pool (per minute).
    c_recovery: float = 0.3
    #: Return rate from the fatigued pool back to resting (per minute).
    c_rest: float = 0.05

    def __init__(self) -> None:
        self._state = FourCompartmentState(1.0, 0.0, 0.0, 0.0)
        self._fatigue_index = 0.0
        self._efficiency = 1.0
        self._target_activation = 0.0

    def reset(self) -> None:
        """Reset the state to the fully rested condition."""

        self._state = FourCompartmentState(1.0, 0.0, 0.0, 0.0)
        self._fatigue_index = 0.0
        self._efficiency = 1.0
        self._target_activation = 0.0

    def snapshot(self) -> FatigueSnapshot:
        """Return a copy of the current state."""

        state = self._state
        return FatigueSnapshot(
            rest=state.rest,
            active=state.active,
            fatigued=state.fatigued,
            recovery=state.recovery,
            fatigue_index=self._fatigue_index,
            efficiency=self._efficiency,
            target_activation=self._target_activation,
        )

    def update(
        self,
        *,
        par: float,
        workload: float,
        basal_workload: float,
        dtime: float,
    ) -> FatigueSnapshot:
        """Advance the fatigue state by ``dtime`` seconds.

        Parameters
        ----------
        par:
            Physical activity ratio demanded by the thermoregulation model.
        workload:
            Total mechanical/metabolic workload currently assigned to the
            muscles [W].
        basal_workload:
            Baseline muscular metabolic rate [W]; used to normalise the
            demanded activation level.
        dtime:
            Integration step [s].
        """

        dt_min = max(dtime / 60.0, 0.0)
        if dt_min == 0.0:
            return self.snapshot()

        # Convert the current work demand to a normalised activation level.
        # The mapping is based on the normalised force inputs used by Xia &
        # Law (2008).  PAR values around 1.0 correspond to resting, while
        # heavily loaded marching (Pandolf et al., 1977) rarely exceeds PAR 10.
        activation_from_par = max(par - 1.0, 0.0) / 9.0
        activation_from_par = max(0.0, min(activation_from_par, 1.0))

        # Normalise the muscular workload to the basal level so that intense
        # shivering or load carriage further increases activation demand.
        if basal_workload > 0.0:
            activation_from_work = max(workload, 0.0) / (basal_workload + workload)
        else:
            activation_from_work = 0.0

        target_activation = max(activation_from_par, activation_from_work)
        target_activation = max(0.0, min(target_activation, 1.0))
        self._target_activation = target_activation

        state = self._state
        rest_to_active = self.c_active * target_activation * state.rest
        active_to_fatigued = self.c_fatigue * state.active
        recovery_to_active = self.c_recovery * state.recovery
        fatigued_to_rest = self.c_rest * state.fatigued

        state.rest += (-rest_to_active + fatigued_to_rest) * dt_min
        state.active += (rest_to_active - active_to_fatigued + recovery_to_active) * dt_min
        state.fatigued += (active_to_fatigued - fatigued_to_rest) * dt_min
        state.recovery += (active_to_fatigued - recovery_to_active) * dt_min
        state = state.clamp()
        self._state = state

        # Fatigue index approximates the reduction in available active units.
        self._fatigue_index = state.fatigued + 0.5 * state.recovery
        self._efficiency = max(0.05, 1.0 - self._fatigue_index)

        return self.snapshot()