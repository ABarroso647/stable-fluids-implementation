import numpy as np
import scipy.sparse.linalg as splinalg
from scipy import interpolate
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.graph_objects as go
import json
from applied_forces import get_force_function
import imageio
import os

def gradient(field, element_length):
    grad_list = []
    for i in range(field.ndim):
        # Calculate partial derivatives for each dimension
        grad = partial_derivative(field, element_length, i)[..., np.newaxis]
        grad_list.append(grad)

    # Concatenate the partial derivatives along the last axis
    return np.concatenate(grad_list, axis=-1)

def partial_derivative(field, element_length, axis):
    # Compute partial derivative with center method
    diff = ((np.roll(field, -1, axis=axis) - np.roll(field, 1, axis=axis)) / (2 * element_length))
    #Handle boundary conditions
    diff = FluidSimu.set_edges_to_zero(diff)
    return diff



def laplace(field, element_length):
    diff = np.zeros_like(field)
    # discrete laplace calculations
    for i in range(field.ndim):
       diff += np.roll(field,1,i)
       diff += np.roll(field,-1,i)
    diff -= 4*field
    diff /= element_length**2
    return FluidSimu.set_edges_to_zero(diff)


class FluidSimu:
    def __init__(self, domain_size, n_points, n_time_steps, time_step_length, kinematic_viscosity):
        #Define Grid and Fluid Parmaeters
        self.domain_size = domain_size
        self.n_points = n_points
        self.n_time_steps = n_time_steps
        self.time_step_length = time_step_length
        self.kinematic_viscosity = kinematic_viscosity
        self.element_length = self.domain_size / (self.n_points - 1)
        self.scalar_shape = (self.n_points, self.n_points)
        self.scalar_dof = self.n_points ** 2
        self.vector_shape = (self.n_points, self.n_points, 2)
        self.vector_dof = self.n_points ** 2 * 2

        #Define grid and coordinate sytem
        self.x = np.linspace(0.0, self.domain_size, self.n_points)
        self.y = np.linspace(0.0, self.domain_size, self.n_points)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")
        self.coordinates = np.stack([self.X, self.Y], axis=-1)

        #Define Velocity Storage
        self.velocities = np.zeros(self.vector_shape)
        self.pressure = np.zeros(self.scalar_shape)
    @staticmethod
    def set_edges_to_zero(array):
        """
        Sets the edges of an array to zero for arrays of any shape.

        Parameters:
        - array (numpy.ndarray): The input array of any shape.

        Returns:
        - numpy.ndarray: The array with edges set to zero.
        """
        # Create a view of the array
        array_edges_zeroed = array.copy()

        # Iterate over all dimensions
        for dim in range(array.ndim):
            # Set edges along the current dimension to zero
            slicer_first = [slice(None)] * array.ndim
            slicer_last = [slice(None)] * array.ndim
            slicer_first[dim] = 0  # First slice along this dimension
            slicer_last[dim] = -1  # Last slice along this dimension
            array_edges_zeroed[tuple(slicer_first)] = 0
            array_edges_zeroed[tuple(slicer_last)] = 0

        return array_edges_zeroed

    def divergence(self, vector_field):
        return (
            partial_derivative(vector_field[..., 0], self.element_length,0) +
            partial_derivative(vector_field[..., 1], self.element_length,1)
        )
    def apply_forces(self, forces):
        self.velocities += self.time_step_length*forces

    def self_advect(self):

        #We use the updated advect formula from
        #http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf
        #This can technically introduce some instabilities but for our cases replicates the paper close enough
        #without needing to use rk2
        backtraced_positions = np.clip(
            self.coordinates - self.time_step_length * self.velocities,
            0.0,
            self.domain_size
        )
        #interpolate the backgtrade
        self.velocities =  interpolate.interpn(
            points=(self.x, self.y),
            values=self.velocities,
            xi=backtraced_positions
        )

    def diffusion_operator(self, vector_field_flattened):
        vector_field = vector_field_flattened.reshape(self.vector_shape)
        return (
            vector_field - self.kinematic_viscosity * self.time_step_length *
            laplace(vector_field, self.element_length)
        ).flatten()

    def poisson_operator(self, field_flattened ):
        field = field_flattened.reshape(self.scalar_shape)
        return laplace(field, self.element_length).flatten()


    def curl_2d(self, vector_field):
        return (
                partial_derivative(vector_field[..., 1], self.element_length,0) -
                partial_derivative(vector_field[..., 0], self.element_length,1)
        )

def main():

    # Load data from the JSON file
    with open('hyperparams.json', 'r') as f:
        params = json.load(f)

    # Assign the loaded values to the variables
    domain_size = params["domain_size"]
    n_points = params["n_points"]
    n_time_steps = params["n_time_steps"]
    time_steps_length = params["time_steps_length"]
    kinematic_viscosity = params["kinematic_viscosity"]
    force_func_name = params["force_func_name"]
    max_iter = params["max_iter"]
    render_fps = params["render_fps"]
    arrow_scale = params["arrow_scale"]
    output_quiver = params["output_quiver"]
    force_function = get_force_function(force_func_name)

    simu = FluidSimu(domain_size, n_points, n_time_steps, time_steps_length,kinematic_viscosity)


    time_current = 0.0

    plt.style.use("dark_background")
    plt.figure()
    curl = None
    frames = []
    frames_gif = []

    for _ in tqdm(range(n_time_steps)):
        time_current += time_steps_length

        #Apply forces to Sstep
        cur_forces = force_function(time_current, simu.coordinates)

        #Apply forces to velocity
        simu.apply_forces(cur_forces)

        #Do transportation
        #Looking at the advection the fluid performs on itself
        simu.self_advect()
        #Velocity Diffusal
        simu.velocities = splinalg.cg(
            A=splinalg.LinearOperator(
                shape=(simu.vector_dof, simu.vector_dof),
                matvec=lambda vf: simu.diffusion_operator(vf)),
            b=simu.velocities.flatten(),
            maxiter=max_iter
        )[0].reshape(simu.vector_shape)
        #Solving for pressure in projection step
        simu.pressure = splinalg.cg(
            A=splinalg.LinearOperator(
                shape=(simu.scalar_dof, simu.scalar_dof),
                matvec=lambda f: simu.poisson_operator(f)),
            b=simu.divergence(simu.velocities).flatten(),
            maxiter=max_iter
        )[0].reshape(simu.scalar_shape)
        #Projection and swap step
        simu.velocities -= gradient(simu.pressure, simu.element_length)
        #drawing
        curl = simu.curl_2d(simu.velocities)

        #Drawing for immediate visualization on plotly
        frames.append(go.Frame(data=[       go.Heatmap(
                                z=curl.T,
                                x=simu.x,  # Assuming x coordinates of the simulation grid
                                y=simu.y,  # Assuming y coordinates of the simulation grid
                                colorscale="RdBu",
                                colorbar=dict(title="Curl (Vorticity)"),
                                zmin=np.min(curl),  # Optionally set min/max for consistent coloring
                                zmax=np.max(curl),
                            )]))
        #Saving visualization to gif for viewing later
        fig, ax = plt.subplots()
        ax.contourf(simu.X, simu.Y, curl, levels=100, cmap="RdBu")
        if output_quiver:
            ax.quiver(simu.X, simu.Y, simu.velocities[..., 0], simu.velocities[..., 1], color="dimgray",
                                scale=arrow_scale, width=0.002)
        force_func_spaces = force_func_name.replace("_", " ")
        ax.set_title(f"{force_func_spaces}:Curl (Vorticity)")
        plt.axis('off')
        plt.savefig("outputs/frame.png", bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        frames_gif.append(imageio.v2.imread("outputs/frame.png"))

    if os.path.exists("outputs/frame.png"):
        os.remove("outputs/frame.png")
    fig = go.Figure(
        data=[
            go.Heatmap(z=curl.T, x=simu.x, y=simu.y, colorscale="RdBu",  colorbar=dict(title="Curl (Vorticity)"),
                                zmin=np.min(curl),  # Optionally set min/max for consistent coloring
                                zmax=np.max(curl))],
        layout=go.Layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                        ),
                    ],
                )
            ],
            xaxis=dict(title="X", showgrid=False),
            yaxis=dict(title="Y", scaleanchor="x", showgrid=False),

        ),
        frames=frames,
    )


    durations = [1/render_fps] * len(frames)  # Default duration for each frame
    durations[-1] = 30 / render_fps # 2 times the normal frame duration for the pause
    output_path = f"outputs/{force_func_name}.gif"
    imageio.mimwrite(output_path, frames_gif, duration=durations, loop = 0)
    fig.show()

if __name__ == "__main__":
    main()
