import numpy as np


#Two split liquids which start at the bottom and push upwards for 2 seconds with force dissipating over time
def bottom_inflow(time, point):
    time_decay = np.maximum(2.0 - 0.5 * time, 0.0)
    return time_decay * np.where(
        (
            (point[0] > 0.4) & (point[0] < 0.6) &
            (point[1] > 0.1) & (point[1] < 0.3)
        ),
        np.array([0.0, 1.0]),
        np.array([0.0, 0.0])
    )

def left_to_right(time, point):
    # Time decay function
    time_decay = np.maximum(2.0 - 0.5 * time, 0.0)

    # Define the regions for the four quadrants (top-left, top-right, bottom-left, bottom-right)
    region_tl = (point[0] > 0.0) & (point[0] < 0.2) & (point[1] > 0.4) & (point[1] < 0.6)
    region_tr = (point[0] > 0.8) & (point[0] < 1.0) & (point[1] > 0.4) & (point[1] < 0.6)
    region_bl = (point[0] > 0.0) & (point[0] < 0.2) & (point[1] > 0.0) & (point[1] < 0.2)
    region_br = (point[0] > 0.8) & (point[0] < 1.0) & (point[1] > 0.0) & (point[1] < 0.2)

    # Apply the forcing function with a time decay and force in the defined regions
    return time_decay * np.where(
        region_tl | region_tr | region_bl | region_br,
        np.array([0.0, 1.0]),
        np.array([0.0, 0.0])
    )
def four_flows(time, point):
    # Time decay function
    time_decay = np.maximum(2.0 - 0.5 * time, 0.0)

    # Define the regions for the four quadrants (top-left, top-right, bottom-left, bottom-right)
    region_tl = (point[0] > 0.4) & (point[0] < 0.6) & (point[1] > 0.1) & (point[1] < 0.3)  # Bottom
    region_tr = (point[0] > 0.4) & (point[0] < 0.6) & (point[1] > 0.8) & (point[1] < 1.0)  # Top
    region_bl = (point[0] > 0.0) & (point[0] < 0.2) & (point[1] > 0.4) & (point[1] < 0.6)  # Left
    region_br = (point[0] > 0.8) & (point[0] < 1.0) & (point[1] > 0.4) & (point[1] < 0.6)  # Right
    conditions = [region_tl, region_tr, region_bl, region_br]
    choices = [np.array([0.0, 1.0]), np.array([0.0, -1.0]), np.array([1.0, 0.0]), np.array([-1.0, 0.0])]

    # Apply the forcing function with a time decay and force in the defined regions
    return time_decay * np.select(conditions, choices)

def circular_inflow(time, point):
    """ Radial inflow at a specific region, creating a circular source. """
    A = 1.0  # constant amplitude
    x0, y0 = 0.5, 0.5  # center of the circular inflow
    time_decay = np.maximum(2.0 - 0.5 * time, 0.0)

    # Flatten the grid of points into a 2D array for vectorized computation
    x, y = point[..., 0], point[..., 1]

    # Calculate force direction based on point position relative to the center (x0, y0)
    theta = np.arctan2(y - y0, x - x0)
    fx = A * np.cos(theta)
    fy = A * np.sin(theta)

    return time_decay * np.stack([fx, fy], axis=-1)

def point_vortex(time, point):
    """ Point vortex at a specific location. """
    A = 1.0  # constant amplitude
    x0, y0 = 0.5, 0.5  # location of vortex center
    time_decay = np.maximum(1.0 - 0.5 * time, 0.0)

    # Flatten the grid of points into a 2D array for vectorized computation
    x, y = point[..., 0], point[..., 1]

    # Calculate distance vectors relative to the vortex center
    r_x = y - y0
    r_y = -(x - x0)
    fx = A * r_x
    fy = A * r_y

    return time_decay * np.stack([fx, fy], axis=-1)

def sine_wave_oscillation(time, point):
    """ Sine wave oscillation creating wave-like patterns. """
    A = 1.0  # constant amplitude
    k = 2 * np.pi  # wave number
    omega = 1.0  # frequency
    time_decay = np.maximum(2.0 - 0.5 * time, 0.0)

    # Flatten the grid of points into a 2D array for vectorized computation
    x, y = point[..., 0], point[..., 1]

    # Apply sine function to x and y components
    fx = A * np.sin(k * x + omega * time)
    fy = A * np.sin(k * y + omega * time)

    return time_decay * np.stack([fx, fy], axis=-1)

def gaussian_jet_stream(time, point):
    """ Gaussian jet stream (high-velocity jet). """
    A = 1.0  # constant amplitude
    y0 = 0.5  # y-coordinate of the jet stream
    sigma = 0.1  # standard deviation of the Gaussian profile

    # Flatten the grid of points into a 2D array for vectorized computation
    x, y = point[..., 0], point[..., 1]

    fx = A * np.exp(-((y - y0) ** 2) / (2 * sigma ** 2))
    fy = np.zeros_like(fx)  # No flow in x-direction for simplicity

    return np.stack([fx, fy], axis=-1)

def spiral_forcing(time, point):
    """ Spiral forcing function creating vortex-like spiral patterns. """
    A = 1.0  # constant amplitude
    x0, y0 = 0.5, 0.5  # center of the spiral
    omega = 1.0  # angular velocity
    time_decay = np.maximum(2.0 - 0.5 * time, 0.0)

    # Flatten the grid of points into a 2D array for vectorized computation
    x, y = point[..., 0], point[..., 1]

    theta = np.arctan2(y - y0, x - x0)
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    fx = A * np.cos(omega + theta) * r
    fy = A * np.sin(omega + theta) * r

    return time_decay * np.stack([fx, fy], axis=-1)

def checkerboard_forces(time, point):
    """ Checkerboard forcing with alternating inflows and outflows. """
    A = 1.0  # constant amplitude
    w = 0.1  # width of each checkerboard square

    # Flatten the grid of points into a 2D array for vectorized computation
    x, y = point[..., 0], point[..., 1]

    fx = A * ((np.floor(x / w) + np.floor(y / w)) % 2 == 0)
    fy = fx  # Same for y-direction

    return np.stack([fx, fy], axis=-1)

def random_noise(time, point):
    """ Random noise for turbulent-like behavior. """
    A = 1.0  # constant amplitude

    # Flatten the grid of points into a 2D array for vectorized computation
    fx = A * np.random.uniform(-1, 1, size=point.shape[:-1])
    fy = A * np.random.uniform(-1, 1, size=point.shape[:-1])

    return np.stack([fx, fy], axis=-1)

#decent
def elliptical_sink_source(time, point):
    """ Elliptical sink/source pattern. """
    A = 1.0  # constant amplitude
    x0, y0 = 0.5, 0.5  # center of the ellipse
    a, b = 0.2, 0.3  # semi-major and semi-minor axes

    # Flatten the grid of points into a 2D array for vectorized computation
    x, y = point[..., 0], point[..., 1]

    fx = A * np.exp(-((x - x0) ** 2 / a ** 2) - ((y - y0) ** 2 / b ** 2))
    fy = fx  # Same for y-direction

    return np.stack([fx, fy], axis=-1)

def time_varying_pulses(time, point):
    """ Time-varying pulse forcing with a Gaussian profile. """
    A = 1.0  # constant amplitude
    x0, y0 = 0.5, 0.5  # center of the pulse
    sigma = 0.1  # standard deviation of the Gaussian profile
    omega = 1.0  # frequency of the time-varying pulse
    time_decay = np.maximum(4.0 - 0.5 * time, 0.0)

    # Flatten the grid of points into a 2D array for vectorized computation
    x, y = point[..., 0], point[..., 1]

    fx = A * np.cos(omega * time) * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    fy = fx  # Same for y-direction

    return time_decay * np.stack([fx, fy], axis=-1)

def double_helix(time, point):
    """ Double helix vortex pattern (two point vortices). """
    offset = 0.1  # offset for second vortex

    # Apply point vortex to both centers
    fx1, fy1 = point_vortex(time, point)
    fx2, fy2 = point_vortex(time, point + offset)
    fx = fx1 + fx2
    fy = fy1 + fy2

    return np.stack([fx, fy], axis=-1)


def get_force_function(func_name):
    # Retrieve the function from globals
    func = globals().get(func_name)
    if not callable(func):
        raise ValueError(f"No callable function named '{func_name}' found.")
    # Return the vectorized version of the function
    return np.vectorize(
        pyfunc=func,
        signature="(),(d)->(d)"
    )