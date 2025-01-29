import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def viewTomo(path):
    with mrcfile.open(f'{path}', permissive=True) as mrc:
        data = mrc.data

    # Set up the figure with 3 subplots (one for each axis)
    fig, (ax_z, ax_y, ax_x) = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=0.3)

    # Initial slice indices for each axis
    slice_z = 0
    slice_y = 0
    slice_x = 0

    # Initial plots for each axis
    img_z = ax_z.imshow(data[slice_z, :, :], cmap='gray', origin='lower')
    img_y = ax_y.imshow(data[:, slice_y, :], cmap='gray', origin='lower')
    img_x = ax_x.imshow(data[:, :, slice_x], cmap='gray', origin='lower')

    # Set titles
    ax_z.set_title("Z-axis slice")
    ax_y.set_title("Y-axis slice")
    ax_x.set_title("X-axis slice")

    # Add sliders for each axis
    ax_slider_z = plt.axes([0.25, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_slider_y = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    ax_slider_x = plt.axes([0.25, 0.11, 0.65, 0.03], facecolor='lightgoldenrodyellow')

    slider_z = Slider(ax_slider_z, 'Z Slice', 0, data.shape[0]-1, valinit=slice_z, valstep=1)
    slider_y = Slider(ax_slider_y, 'Y Slice', 0, data.shape[1]-1, valinit=slice_y, valstep=1)
    slider_x = Slider(ax_slider_x, 'X Slice', 0, data.shape[2]-1, valinit=slice_x, valstep=1)

    # Update function for the sliders
    def update(val):
        # Get the current slider values
        slice_z = int(slider_z.val)
        slice_y = int(slider_y.val)
        slice_x = int(slider_x.val)

        # Update the image for each axis
        img_z.set_data(data[slice_z, :, :])
        img_y.set_data(data[:, slice_y, :])
        img_x.set_data(data[:, :, slice_x])

        # Update titles to show the current slice number
        ax_z.set_title(f"XY Slice: {slice_z + 1}")
        ax_y.set_title(f"XZ Slice: {slice_y + 1}")
        ax_x.set_title(f"YZ Slice: {slice_x + 1}")

        # Redraw the figure
        fig.canvas.draw_idle()

    # Link the sliders to the update function
    slider_z.on_changed(update)
    slider_y.on_changed(update)
    slider_x.on_changed(update)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    viewTomo('/home/frasunkiewicz/Projects/isonet/artificial_data_tests/repetitive_sphere/tomos/marked_volume.mrc')
