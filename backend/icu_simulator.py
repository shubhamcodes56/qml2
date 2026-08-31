import time
import math
import random

class ICUSimulator:
    def __init__(self):
        self.start_time = time.time()
        self.is_critical = False
        
        # Baselines (Normal Healthy Adult)
        self.baselines = {
            "hr": 75.0,     # Heart Rate (bpm)
            "spo2": 98.0,   # Oxygen Saturation (%)
            "rr": 16.0,     # Respiratory Rate (breaths/min)
            "temp": 37.0    # Core Temperature (°C)
        }
        
        # Current state
        self.current = self.baselines.copy()
        
    def _add_noise(self, value, variance):
        """Add biological noise/fluctuation to a reading."""
        return value + random.gauss(0, variance)

    def trigger_anomaly(self):
        """Trigger the micro-drift anomaly."""
        self.is_critical = True

    def get_live_vitals(self):
        """Generate the next tick of vital signs."""
        elapsed = time.time() - self.start_time
        
        # Sine wave breathing/heart patterns for realistic fluctuation
        respiratory_cycle = math.sin(elapsed * (self.current["rr"] / 60) * 2 * math.pi)
        cardiac_cycle = math.sin(elapsed * (self.current["hr"] / 60) * 2 * math.pi)

        if self.is_critical:
            # The Micro-Drift (Drifting towards critical, but still in "Normal" bounds)
            # A normal monitor won't trigger an alarm here.
            # HR goes up slightly
            target_hr = 88.0
            # SpO2 goes down slightly
            target_spo2 = 95.0
            # RR goes up slightly (compensating for lower oxygen)
            target_rr = 22.0
            # Temp goes up slightly (infection flare-up)
            target_temp = 37.8
            
            # Drift rate
            drift_factor = 0.05
            
            self.current["hr"] += (target_hr - self.current["hr"]) * drift_factor
            self.current["spo2"] += (target_spo2 - self.current["spo2"]) * drift_factor
            self.current["rr"] += (target_rr - self.current["rr"]) * drift_factor
            self.current["temp"] += (target_temp - self.current["temp"]) * drift_factor
        else:
            # Stay near baseline
            self.current = self.baselines.copy()

        # Add physical noise
        hr_reading = self._add_noise(self.current["hr"], 1.5) + (respiratory_cycle * 2.0)
        spo2_reading = self._add_noise(self.current["spo2"], 0.2)
        rr_reading = self._add_noise(self.current["rr"], 0.5)
        temp_reading = self._add_noise(self.current["temp"], 0.05)

        # Ensure bounds
        spo2_reading = min(100.0, max(0.0, spo2_reading))

        return {
            "timestamp": time.time(),
            "is_anomaly_injected": self.is_critical,
            "vitals": {
                "Heart Rate": round(hr_reading, 1),
                "SpO2": round(spo2_reading, 1),
                "Resp Rate": round(rr_reading, 1),
                "Temperature": round(temp_reading, 2)
            }
        }
