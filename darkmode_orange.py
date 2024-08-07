"""Dark mode for matplotlib - orange version."""

import matplotlib as mpl

# Plotting defaults
BLACK = "black"
WHITE = "white"
GREY = "#303030"
BLUE = "#1f77b4"
LIGHTGREY = "#737373"
ORANGE = "#ff4500"  # "#ff7f0e"
CYCLE = "colorblind"
CMAPCYC = "twilight"
CMAPDIV = "BuRd"
CMAPSEQ = "viridis"
CMAPCAT = "colorblind10"
DIVERGING = "div"
FRAMEALPHA = 0.0  # legend and colorbar
FONTNAME = "sans-serif"
FONTSIZE = 9.0
GRIDALPHA = 0.4  # 0.1
GRIDBELOW = "line"
GRIDPAD = 3.0
GRIDRATIO = 0.5  # differentiated from major by half size reduction
GRIDSTYLE = "-"
LABELPAD = 4.0  # default is 4.0, previously was 3.0
LARGESIZE = "large"
LEGENDLOC = "best"
LINEWIDTH = 0.6  # 0.6
MARGIN = 0.05
SMALLSIZE = "medium"
TITLEWEIGHT = "bold"
TICKDIR = "out"
TICKLEN = 4.0
TICKLENRATIO = 0.5  # very noticeable length reduction
TICKMINOR = True
TICKPAD = 2.0
TICKWIDTHRATIO = 0.8  # very slight width reduction
TITLEPAD = 5.0  # default is 6.0, previously was 3.0
ZLINES = 2  # default zorder for lines
ZPATCHES = 1
DPI = 200

# Overrides of matplotlib default style
rc_params = {
    "axes.axisbelow": GRIDBELOW,
    "axes.grid": True,  # enable lightweight transparent grid by default
    "axes.grid.which": "major",
    "axes.labelpad": LABELPAD,  # more compact
    "axes.labelsize": SMALLSIZE,
    "axes.labelweight": "normal",
    "axes.linewidth": LINEWIDTH,
    "axes.titlepad": TITLEPAD,  # more compact
    "axes.titlesize": LARGESIZE,
    "axes.titleweight": TITLEWEIGHT,
    "axes.xmargin": MARGIN,
    "axes.ymargin": MARGIN,
    "errorbar.capsize": 3.0,
    "figure.autolayout": False,
    "figure.figsize": (4.0, 4.0),  # for interactife backends
    "figure.dpi": DPI,
    "figure.titlesize": LARGESIZE,
    "figure.titleweight": TITLEWEIGHT,
    "font.family": FONTNAME,
    "font.size": FONTSIZE,
    "grid.alpha": GRIDALPHA,  # lightweight unobtrusive gridlines
    "grid.color": ORANGE,  # lightweight unobtrusive gridlines
    "grid.linestyle": GRIDSTYLE,
    "grid.linewidth": LINEWIDTH,
    "hatch.color": BLACK,
    "hatch.linewidth": LINEWIDTH,
    "image.cmap": CMAPSEQ,
    "lines.linestyle": "-",
    "lines.linewidth": 1.5,
    "lines.markersize": 6.0,
    "legend.loc": LEGENDLOC,
    "legend.borderaxespad": 0,  # i.e. flush against edge
    "legend.borderpad": 0.5,  # a bit more roomy
    "legend.columnspacing": 1.5,  # a bit more compact (see handletextpad)
    "legend.edgecolor": BLACK,
    "legend.facecolor": WHITE,
    "legend.fancybox": True,  # i.e. BboxStyle 'square' not 'round'
    "legend.fontsize": SMALLSIZE,
    "legend.framealpha": FRAMEALPHA,
    "legend.handleheight": 1.0,  # default is 0.7
    "legend.handlelength": 1.0,  # default is 2.0
    "legend.handletextpad": 0.5,  # a bit more compact (see columnspacing)
    "mathtext.default": "it",
    "mathtext.fontset": "custom",
    "mathtext.bf": "regular:bold",  # custom settings implemented above
    "mathtext.cal": "cursive",
    "mathtext.it": "regular:italic",
    "mathtext.rm": "regular",
    "mathtext.sf": "regular",
    "mathtext.tt": "monospace",
    "patch.linewidth": LINEWIDTH,
    "savefig.directory": "",  # use the working directory
    "savefig.dpi": DPI,
    "savefig.format": "png",  # use vector graphics
    "savefig.transparent": False,
    "savefig.bbox": "tight",
    "savefig.edgecolor": "none",
    "xtick.direction": TICKDIR,
    "xtick.labelsize": SMALLSIZE,
    "xtick.major.pad": TICKPAD,
    "xtick.major.size": TICKLEN,
    "xtick.major.width": LINEWIDTH,
    "xtick.minor.pad": TICKPAD,
    "xtick.minor.size": TICKLEN * TICKLENRATIO,
    "xtick.minor.width": LINEWIDTH * TICKWIDTHRATIO,
    "xtick.minor.visible": TICKMINOR,
    "ytick.direction": TICKDIR,
    "ytick.labelsize": SMALLSIZE,
    "ytick.major.pad": TICKPAD,
    "ytick.major.size": TICKLEN,
    "ytick.major.width": LINEWIDTH,
    "ytick.minor.pad": TICKPAD,
    "ytick.minor.size": TICKLEN * TICKLENRATIO,
    "ytick.minor.width": LINEWIDTH * TICKWIDTHRATIO,
    "ytick.minor.visible": TICKMINOR,
    "lines.solid_capstyle": "round",
    "axes.edgecolor": ORANGE,
    "axes.facecolor": BLACK,
    "axes.labelcolor": WHITE,
    "figure.facecolor": GREY,
    "patch.edgecolor": GREY,
    "patch.force_edgecolor": True,
    "text.color": WHITE,
    "xtick.color": BLUE,
    "ytick.color": BLUE,
    "savefig.facecolor": GREY,
}

# Dark mode


# Set the params
mpl.rcParams.update(rc_params)
