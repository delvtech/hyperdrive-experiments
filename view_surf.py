# %%
import matplotlib.patches as mpatches
from scipy.interpolate import griddata

# %%
# surf plot

# Extracting data from the dataframe
x = df2.loc[idx, vars[0]].values
y = df2.loc[idx, vars[0]].values
z = df2.loc[idx, "apr"].values

# Creating a meshgrid
xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolating z values
zi = griddata((x, y), z, (xi, yi), method="linear")

# Check if zi is a tuple
if isinstance(zi, tuple):
    raise ValueError("zi should not be a tuple.")

# Creating the plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(221, projection="3d")
surface = ax.plot_surface(xi, yi, zi, cmap="viridis")
# Rotate 90 degrees and re-plot
for i in range(1, 4):
    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    surface = ax.plot_surface(xi, yi, zi, cmap="viridis")
    ax.set_title(f"Rotation {i} (90 degrees)")
    ax.view_init(30, 30 + i * 90)  # Rotate by 90 degrees each time

plt.tight_layout()
plt.show()

# %%
# Best angle
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")
surface = ax.plot_surface(xi, yi, zi, cmap="viridis", label="apr")
ax.view_init(30, 60 + 180)

# Adding labels and title
ax.set_xlabel(vars[0])
ax.set_ylabel(vars[1])
ax.set_zlabel("apr")

ax.set_title(f"n={df2.loc[idx,:].shape[0]} rsq={model.rsquared:,.2f}\n{formula}")

breakeven_value = df2.loc[df2.username == "share price", "apr"].values.mean()
# Ensuring breakeven_value is a single value, not an array
if isinstance(breakeven_value, np.ndarray):
    breakeven_value = breakeven_value.item()
# draw a transparent horizontal plane at the breakeven value
ax.plot_surface(
    xi,
    yi,
    np.full_like(zi, breakeven_value),
    alpha=0.5,
    label="Breakeven",
    color="green",
)

# Custom legend
apr_patch = mpatches.Patch(color="blue", label="apr")
breakeven_patch = mpatches.Patch(color="green", label="Breakeven")
ax.legend(handles=[apr_patch, breakeven_patch])

# Display the plot
# plt.tight_layout()
plt.show()