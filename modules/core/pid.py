import warnings
from typing import Optional

class PIDController:
    """
    Enhanced PID Controller with advanced features:
    - Derivative filtering
    - Anti-windup mechanisms
    - Output clamping
    - Multiple derivative modes
    - Safe time handling
    - Runtime adjustments
    """
    
    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        setpoint: float = 0.0,
        *,
        output_limits: tuple = (-float('inf'), float('inf')),
        derivative_tau: float = 0.0,
        anti_windup: str = 'back_calculation',
        derivative_mode: str = 'measurement',
        time_sample: float = 0.01
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._setpoint = setpoint
        
        # Configuration parameters
        self.output_min, self.output_max = output_limits
        self.derivative_tau = derivative_tau
        self.anti_windup = anti_windup.lower()
        self.derivative_mode = derivative_mode.lower()
        self.time_sample = time_sample
        
        # State variables
        self._integral = 0.0
        self._last_error: Optional[float] = None
        self._last_measurement: Optional[float] = None
        self._last_derivative = 0.0
        self._last_output = 0.0
        
        # Input validation
        self._validate_parameters()

    def _validate_parameters(self):
        """Ensure configuration parameters are valid"""
        if self.anti_windup not in {'back_calculation', 'clamp', 'none'}:
            raise ValueError("Invalid anti-windup method")
        if self.derivative_mode not in {'measurement', 'error'}:
            raise ValueError("Invalid derivative mode")
        if self.output_max <= self.output_min:
            raise ValueError("Output limits must be ascending")

    def update(self, measurement: float, dt: Optional[float] = None) -> float:
        """
        Calculate PID output with enhanced features
        :param measurement: Current process variable
        :param dt: Time delta since last update (optional)
        :return: PID control output
        """
        dt = dt if dt is not None else self.time_sample
        
        # Handle invalid time intervals
        if dt <= 0:
            warnings.warn(f"Invalid dt={dt}, using default {self.time_sample}", RuntimeWarning)
            dt = self.time_sample

        error = self.setpoint - measurement
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with clamping
        i_term = self._update_integral(error, dt)
        
        # Derivative term with filtering and mode selection
        d_term = self._update_derivative(measurement, error, dt)
        
        # Calculate raw output
        output = p_term + i_term + d_term
        
        # Apply output clamping
        clamped_output = max(self.output_min, min(output, self.output_max))
        
        # Handle anti-windup
        self._handle_anti_windup(output, clamped_output, p_term, d_term)
        
        # Store state for next iteration
        self._last_error = error
        self._last_measurement = measurement
        self._last_output = clamped_output
        
        return clamped_output

    def _update_integral(self, error: float, dt: float) -> float:
        """Update integral term with anti-windup protection"""
        self._integral += error * dt
        
        # Integral clamping
        if self.anti_windup == 'clamp':
            max_i = (self.output_max - self.kp * error) / self.ki if self.ki != 0 else 0
            min_i = (self.output_min - self.kp * error) / self.ki if self.ki != 0 else 0
            self._integral = max(min(self._integral, max_i), min_i)
            
        return self.ki * self._integral

    def _update_derivative(self, measurement: float, error: float, dt: float) -> float:
        """Calculate derivative term with filtering and mode selection"""
        if self.derivative_mode == 'measurement':
            if self._last_measurement is None:
                raw_deriv = 0.0
            else:
                raw_deriv = (measurement - self._last_measurement) / dt
        else:  # Error-based
            if self._last_error is None:
                raw_deriv = 0.0
            else:
                raw_deriv = (error - self._last_error) / dt

        # Apply low-pass filtering
        if self.derivative_tau > 0 and dt > 0:
            alpha = dt / (self.derivative_tau + dt)
            filtered_deriv = alpha * raw_deriv + (1 - alpha) * self._last_derivative
        else:
            filtered_deriv = raw_deriv

        self._last_derivative = filtered_deriv
        return self.kd * (-filtered_deriv if self.derivative_mode == 'measurement' else filtered_deriv)

    def _handle_anti_windup(self, raw_output: float, clamped_output: float, 
                           p_term: float, d_term: float):
        """Apply anti-windup corrections"""
        if self.anti_windup == 'back_calculation' and raw_output != clamped_output:
            if self.ki != 0:
                self._integral = (clamped_output - p_term - d_term) / self.ki

    def reset(self):
        """Reset controller state"""
        self._integral = 0.0
        self._last_error = None
        self._last_measurement = None
        self._last_derivative = 0.0
        self._last_output = 0.0

    @property
    def setpoint(self) -> float:
        return self._setpoint

    @setpoint.setter
    def setpoint(self, value: float):
        """Setpoint change with bump-less transfer"""
        if self._last_output is not None:
            error = value - (self.setpoint - self._last_error) if self._last_error else 0
            self._integral += error * self.time_sample
        self._setpoint = value

    @property
    def components(self) -> tuple:
        """Return individual PID components (P, I, D)"""
        return (
            self.kp * (self.setpoint - self._last_measurement) if self._last_measurement else 0,
            self.ki * self._integral,
            self._last_derivative
        )