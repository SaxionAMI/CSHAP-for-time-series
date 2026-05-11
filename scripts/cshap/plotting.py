# Import packages
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import patches

# Helper functions used for plotting results

def plot_shap_decomposition(t, x, y_pred, components, component_names, 
                            base_value, shap_values, class_selected=None, title=None, 
                            figsize=None, dpi=100, ylims=None, xlabel=None,
                            shap_lims=None):
    """
    Code used for visualizing local C-SHAP explanations.
    t: time steps to use along x axis as numpy.array of shape (# time steps)
    x: original univariate signal as numpy.array of shape (# time steps)
    y_pred: output of predictive model for this sample as float
    components: components for plotting as numpy.array of shape (# components, # time steps)
    component_names: names to use for components for labelling as list of strings [# components]
    base_value: SHAP base value for this sample (and for classification the selected class) as float
    shap_values: SHAP values for this sample (and for classification the selected class) as numpy.array of shape (# components)
    class_selected: if classification, specify class to autogenerate title for figure
    title: custom title for figure as string
    figsize: matplotlib size of figure as tuple (horizontal, vertical)
    dpi: matplotlib dpi
    ylims: vertical limits for plots of original signal and components as list of length # components + 1 of lists [ymin, ymax]
    xlabel: label to use for x axis as string 
    shap_lims: horizontal limit to use for SHAP values as [xmin, xmax]
    returns: matplotlib figure
    """
    # Number of components
    n = len(component_names)        

    # Get into easier shape
    components = components.T

    # If figsize is not specified set automatically
    if not figsize:                
        figsize = (7, 2 * (n + 1))

    # Initialize figure by specfying number of rows and columns and user supplied arguments
    fig, ax = plt.subplots(n + 1, 2, figsize=figsize, dpi=dpi, width_ratios=[1.2, 2])

    # Set title if information has been given
    if title:                       # If user specified title, use that
        fig.suptitle(title)
    elif class_selected is not None:    # If user specified class selected put that into the title
        fig.suptitle('Explanation for ' + class_selected, fontsize=10)

    # Use tight layout for the figure
    fig.tight_layout()

    # Remove space between Axes
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)

    # Choose the vertical limits for the original signal (plotted at the top)
    if ylims is None:           # If not specified by user determine based on signal and components
        y_min = np.min([np.min(components), np.min(x)]) - .5
        y_max = np.max([np.max(components), np.max(x)]) + .5
    else:                       # Extract user specified limits
        y_min, y_max = ylims[0]

    # Remove axes for top left plot
    ax[0, 0].axis('off')   

    # Plot original signal using helper function
    plot_signal(ax, t, x, y_min, y_max)

    # Plot components either using user specified vertical limits or previously determined limits
    if ylims is None:
        plot_components(ax, t, components, [[y_min, y_max] for _ in range(len(components))], xlabel=xlabel)
    else:
        plot_components(ax, t, components, ylims[1:], xlabel=xlabel)

    # Use helper functions to plot SHAP values
    plot_shap_values(fig, ax, shap_values, base_value, component_names)
    set_shap_xlims(ax, shap_values, base_value, shap_lims)

    # Add base value and prediction annotations if they have been specified by the user
    if base_value is not None and y_pred is not None:
        set_labels(ax, base_value, y_pred, n)

    # Remove unnecessary xticks to clear up clutter
    xticks = ax[n, 1].xaxis.get_major_ticks()
    xticks[0].set_visible(False)

    # Resize title for visual aesthetics
    fig.subplots_adjust(top=0.93)
    return fig


# -------------------- Helper functions for function above --------------------------------
def plot_signal(ax, t, x, y_min, y_max):
    """
    Plot the original signal
    ax: axes to plot signal at
    t: time steps to use along x axis as numpy.array of shape (# time steps)
    x: original univariate signal as numpy.array of shape (# time steps)
    y_min: vertical lower limit for plot
    y_max: vertical upper limit for plot
    """
    ax[0, 1].set_title('signal decomposition', fontsize=10)
    ax[0, 1].plot(t, x) 
    ax[0, 1].set_xlim(t[0], t[-1])
    ax[0, 1].set_ylim(y_min, y_max)
    ax[0, 1].yaxis.tick_right()
    ax[0, 1].set_xticks([])

def plot_components(ax, t, components, ylims, xlabel=None):
    """
    Plot the original signal
    ax: axes to plot signal at
    t: time steps to use along x axis as numpy.array of shape (# time steps)
    components: components for plotting as numpy.array of shape (# components, # time steps)
    ylims: vertical limits for plots of components as list of length # components of lists [ymin, ymax]
    xlabel: label to use below the very bottom component to annotate the time steps
    """
    n = len(components)     # Number of components
    for i in range(n):      # Plot components one by one
        ax[i + 1, 1].plot(t, components[i])         # Plot component
        ax[i + 1, 1].set_xlim(t[0], t[-1])          # Set tight horizontal limits
        ax[i + 1, 1].set_ylim(ylims[i][0], ylims[i][1]) # Set custom vertical limits
        ax[i + 1, 1].yaxis.tick_right()                 # Move axis ticks to the right side of the plot

        if i < n - 1:                                   # For all but the last plot remove the horizontal ticks
            ax[i + 1, 1].set_xticks([])
    ax[i + 1, 1].set_xlabel(xlabel)                     # Set label for time steps


def get_head_length(shap_value):
    """
    Set length of arrow based on SHAP value, removes head for very small SHAP values or makes it smaller
    shap_value: SHAP value of the arrow to set the head length for
    returns: head length
    """
    head_length = 0.04
    if np.abs(shap_value) < 0.02:
        head_length = 0
    elif np.abs(shap_value) < 0.04:
        head_length = 0.02
    return head_length


def plot_shap_values(fig, ax, shap_values, base_value, component_names):
    """
    Plot SHAP values as SHAP arrows (or bars for small values) in the main figure
    fig: the figure to add SHAP values to
    ax: axes of the figure
    shap_values: SHAP values for this sample (and for classification the selected class) as numpy.array of shape (# components)
    base_value: SHAP base value for this sample (and for classification the selected class) as float
    component_names: names to use for components for labelling as list of strings [# components]
    """
    startx = base_value                         # Where to start SHAP arrows
    renderer = fig.canvas.get_renderer()        # Used for determining scaling of figure to determine whether text fits in arrows

    for i, val in enumerate(shap_values):       # Plot SHAP values one by one
        if val >= 0:                            # Colour based on positive or negative SHAP values
            color = '#f12476'
        else:
            color = '#248ef1'

        # Plot arrow
        head_length = get_head_length(shap_values[i])   # Determine length of arrow head (or no head)
        arrow = patches.FancyArrow(startx, 0, shap_values[i], 0, width=0.3, 
                length_includes_head=True, head_length=head_length, head_width=0.3, color=color)
        ax[i + 1, 0].add_patch(arrow)        

        # Add text to arrow as the SHAP value
        if shap_values[i] >= 0:
            txt_string = "+{val:.2f}".format(val=shap_values[i])
        else:
            txt_string = "{val:.2f}".format(val=shap_values[i])
        txt = ax[i + 1, 0].text(startx + 0.5 * shap_values[i],
            0, txt_string, horizontalalignment="center",
            verticalalignment="center", color='w')

        # Move text next to arrow if arrow is too small
        # The threshold 15 may need to be manually changed depending on the scale of the SHAP values
        text_bbox = txt.get_window_extent(renderer=renderer)
        arrow_bbox = arrow.get_window_extent(renderer=renderer)

        if 15 * text_bbox.width > arrow_bbox.width:
            txt.remove()

            if shap_values[i] >= 0:
                loc = startx + shap_values[i] + 0.01
                align = "left"
            else:
                loc = startx + 0.01
                align = "left"

            txt = ax[i + 1, 0].text(loc, 0, txt_string, horizontalalignment=align,
                verticalalignment="center", color='k')

        # Keep track of where the current arrow head is placed so the next SHAP value is plotted from there
        startx += shap_values[i]
        ax[i + 1, 0].set_ylabel(component_names[i]) # Annotate value with component name

        # For all except the last plot remove x ticks
        if i + 1 < len(shap_values) - 1:
            ax[i + 1, 0].set_xticks([])

def set_shap_xlims(ax, shap_values, base_value, shap_lims=None):
    """
    Based on the way the SHAP arrows are plotted determine horizontal limits or use
    custom specified limits
    ax: axes of the figure
    shap_values: SHAP values for this sample (and for classification the selected class) as numpy.array of shape (# components)
    base_value: SHAP base value for this sample (and for classification the selected class) as float
    shap_lims: horizontal limit to use for SHAP values as [xmin, xmax]
    """
    if shap_lims is None:
        min_reach, max_reach = 0, 0
        cur_reach = 0

        for s in shap_values:
            cur_reach += s
            if cur_reach < min_reach:
                min_reach = cur_reach
            if cur_reach > max_reach:
                max_reach = cur_reach

        min_x = base_value + min_reach - 0.25
        max_x = base_value + max_reach + 0.25

        min_x -= 0.25 * (max_reach - min_reach)
        max_x += 0.25 * (max_reach - min_reach) 
    
    else:
        min_x, max_x = shap_lims

    for i in range(shap_values.shape[0]):
        ax[i + 1, 0].set_xlim(min_x, max_x)
        ax[i + 1, 0].set_ylim(-.5, .5)
        ax[i + 1, 0].set_yticks([])

def set_labels(ax, base_value, y_pred, n_components):
    """
    Helper function used to annotate plot with the base value and value predicted by the model.
    ax: axes of the figure
    base_value: SHAP base value for this sample (and for classification the selected class) as float
    y_pred: output of predictive model for this sample as float
    n_components: number of components
    """
    ax[1, 0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax[1, 0].set_xticks([base_value])
    ax[1, 0].set_xticklabels(["base value\n= {base:.2f}".format(base=base_value)])


    ax[n_components, 0].set_xticks([y_pred])
    ax[n_components, 0].set_xticklabels(["prediction\n= {y_pred:.2f}".format(y_pred=y_pred)])


