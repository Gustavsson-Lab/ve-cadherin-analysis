# %%
"""
This script is formatted to be run in Jupyter Notebook or (like I did) in VSCode with Jupyter Notebook support.
This script is intended to allow the user to load single-particle trajectories from TrackMate and results from 
TraJectory Classifier, along with a  diffraction-limited image of the field of view being analyzed, and 
manually choose a region of interest to analyze. This ROI will be analyzed along with the inverse of the selection.

Be sure to use at least python 3.10, which introduced the match-case statement !!!

Data should be formatted as .csv, and images should be .bmp. No error handling and limited comments, sorry.
Get in contact if you have questions B^)
"""

# %%
from dataclasses import dataclass
import matplotlib.style
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
import seaborn as sns
import tkinter as tk
from tkinter import filedialog as fd
import glob
import os
import re
from tqdm import tqdm
import subprocess

matplotlib.use("ipympl")
plt.style.use("default")
pd.options.mode.chained_assignment = None  # default='warn'


# %%

# specify absolute location of TrackMate data (.csv)
directory = '/data'

# specify transformation matrices to register trajectories to diffraction-limited image
# use identity if not needed
TFORM = [
    [1.01071603879964, 0.00537354230753136, -8.10166375660023], 
    [-0.00428346587523311, 1.00236259677335, 12.5958716335600], 
    [0, 0, 1]
]

# specify additional offset to properly translate trajectories to image [pixels]
x_offset = -543
y_offset = 3

# specify pixel scaling factor [pixels/μm]
scale_factor = 7.6494


# %%

@dataclass
class Importer:

    exposure_time = 0.025
    tform = TFORM
    sf = scale_factor
    x_offset = x_offset
    y_offset = y_offset

    def import_all(self, name: str, source_directory):
        """Create new objects for each dataset acquired from Trajectory
        Classifier & Trackmate"""

        root = tk.Tk()
        root.withdraw()

        self.name = name
        self.directory = source_directory

        # get all filenames containing "name"
        filenames = glob.glob(os.path.join(self.directory, "*.csv"))
        imagenames = sorted(glob.glob(os.path.join(self.directory, "*.bmp")))

        self.import_parents(filenames)
        self.import_settings(filenames)
        self.import_normal(filenames)
        self.import_directed(filenames)
        self.import_confined(filenames)
        self.import_sub(filenames)

        self.import_tracks(filenames)
        self.import_spots(filenames)
        self.import_edges(filenames)

        self.import_image(imagenames)

    def import_parents(self, filenames):
        regex = re.compile("(?=.*Parents).*$")
        path = list(filter(regex.match, filenames))
        df = pd.read_csv(path[0], encoding="unicode_escape", low_memory=False)

        d = dict([(y, x) for x, y in enumerate(sorted(set(df["ID"])))])
        df["ID"] = df["ID"].map(d)

        self.parents = df
        self.id_map = d

    def import_settings(self, filenames):
        regex = re.compile("(?=.*Settings).*$")
        path = list(filter(regex.match, filenames))
        if not path:
            self.settings = pd.DataFrame()
        else:
            self.settings = pd.read_csv(
                path[0], encoding="unicode_escape", low_memory=False
            )

    def import_normal(self, filenames):
        regex = re.compile("(?=.*NORM).*$")
        path = list(filter(regex.match, filenames))
        df = pd.read_csv(path[0], encoding="unicode_escape", low_memory=False)

        df["PARENT-ID"] = df["PARENT-ID"].map(self.id_map)

        self.normal = df

    def import_directed(self, filenames):
        regex = re.compile("(?=.*DIRECTED).*$")
        path = list(filter(regex.match, filenames))
        df = pd.read_csv(path[0], encoding="unicode_escape", low_memory=False)

        df["PARENT-ID"] = df["PARENT-ID"].map(self.id_map)

        self.directed = df

    def import_confined(self, filenames):
        regex = re.compile("(?=.*CONFINED).*$")
        path = list(filter(regex.match, filenames))
        df = pd.read_csv(path[0], encoding="unicode_escape", low_memory=False)

        df["PARENT-ID"] = df["PARENT-ID"].map(self.id_map)

        self.confined = df

    def import_sub(self, filenames):
        regex = re.compile("(?=.*SUBDIFFUSION).*$")
        path = list(filter(regex.match, filenames))
        df = pd.read_csv(path[0], encoding="unicode_escape", low_memory=False)

        df["PARENT-ID"] = df["PARENT-ID"].map(self.id_map)

        self.sub = df

    def import_tracks(self, filenames):
        regex = re.compile("(?=.*tracks).*$")
        path = list(filter(regex.match, filenames))
        df = pd.read_csv(path[0], encoding="unicode_escape", low_memory=False)

        df = df.drop(
            columns=[
                "LABEL",
                "TRACK_INDEX",
                "NUMBER_SPLITS",
                "NUMBER_MERGES",
                "NUMBER_COMPLEX",
            ]
        )
        df = df.drop([0, 1, 2])
        df = df.astype("float")
        df = df.sort_values(by=["TRACK_ID"]).reset_index(drop=True)

        # optional: reject trajectories shorter than a given length
        rejects = df.TRACK_ID[df.NUMBER_SPOTS < 50]
        df = df[df["NUMBER_SPOTS"] >= 50].reset_index(drop=True)

        d = dict([(y, x) for x, y in enumerate(sorted(set(df["TRACK_ID"])))])
        df["TRACK_ID"] = df["TRACK_ID"].map(d)

        self.tracks = df
        self.rejects = np.asarray(rejects)

    def import_spots(self, filenames):
        regex = re.compile("(?=.*spots).*$")
        path = list(filter(regex.match, filenames))
        df = pd.read_csv(path[0], encoding="unicode_escape", low_memory=False)

        df = df.drop(columns=["LABEL", "MANUAL_SPOT_COLOR"])
        df = df.drop([0, 1, 2])
        df = df.astype("float")
        df["POSITION_Z"] = df["POSITION_Z"] + 1
        df["POSITION_T"] = df["POSITION_T"] / 1000 + self.exposure_time
        df = df.sort_values(by=["TRACK_ID", "FRAME"])

        # optional: reject trajectories shorter than a given length
        mask = df["TRACK_ID"].isin(self.rejects)
        df = df[~mask]

        d = dict([(y, x) for x, y in enumerate(sorted(set(df["TRACK_ID"])))])
        df["TRACK_ID"] = df["TRACK_ID"].map(d)

        coords = np.asarray(
            df[["POSITION_X", "POSITION_Y", "POSITION_Z"]].copy()
        )
        
        coords[:, 0] = coords[:, 0] * self.sf + self.x_offset
        coords[:, 1] = coords[:, 1] * self.sf + self.y_offset

        for i in range(len(coords)):
            coords[i] = np.matmul(self.tform, coords[i, :])

        df["POSITION_X_TF"] = coords[:, 0]
        df["POSITION_Y_TF"] = coords[:, 1]

        df = df.sort_values(by=["TRACK_ID", "FRAME"]).reset_index(drop=True)

        self.spots = df

    def import_edges(self, filenames):
        regex = re.compile("(?=.*edges).*$")
        path = list(filter(regex.match, filenames))
        df = pd.read_csv(path[0], encoding="unicode_escape", low_memory=False)

        df = df.drop(columns=["LABEL", "MANUAL_EDGE_COLOR"])
        df = df.drop([0, 1, 2])
        df = df.astype("float")
        df["EDGE_Z_LOCATION"] = df["EDGE_Z_LOCATION"] + 1
        df["EDGE_TIME"] = df["EDGE_TIME"] / 1000 + self.exposure_time
        df = df.sort_values(by=["TRACK_ID", "EDGE_TIME"]).reset_index(
            drop=True
        )

        # optional: reject trajectories shorter than a given length
        mask = df["TRACK_ID"].isin(self.rejects)
        df = df[~mask]

        d = dict([(y, x) for x, y in enumerate(sorted(set(df["TRACK_ID"])))])
        df["TRACK_ID"] = df["TRACK_ID"].map(d)

        self.edges = df

    def import_image(self, imagenames):
        # regex = re.compile(f"^(?=.*{name}).*$")
        # path = list(filter(regex.match, imagenames))

        self.image_path = imagenames
        self.image = plt.imread(imagenames[0])

    def plot_and_select(
        self, alpha=1, size=0.25, figsize=(12,12), show_tracks=True
        ):
        num_tracks = self.spots["TRACK_ID"].nunique()

        fig1, ax1 = plt.subplots(figsize=figsize, layout="constrained")
        ax1.imshow(
            self.image,
            interpolation="nearest",
            rasterized=False,
            cmap="Greys_r",
        )

        if show_tracks == True:
            pts = ax1.scatter(
                self.spots["POSITION_X_TF"],
                self.spots["POSITION_Y_TF"],
                s=size,
                edgecolors=None,
                c="r",
                alpha=alpha
            )
        else:
            pts = ax1.scatter(
                self.spots["POSITION_X_TF"],
                self.spots["POSITION_Y_TF"],
                s=size,
                edgecolors=None,
                c="r",
                alpha=0
            )

        selector = SelectFromCollection(ax1, pts)

        def accept(event):
            if event.key == "enter":
                selector.disconnect()
                ax1.set_title("Points selected!")
                fig1.canvas.draw()
            return points

        points = fig1.canvas.mpl_connect("key_press_event", accept)
        ax1.set_title("Press enter to accept selected points")

        plt.show()

        self.selector = selector

    def save_selection(
        self, export_path, name="default", figsize=(12,12), invert=False, other_fov=None, export=False
    ):
        plt.close()

        # change mask depending on if we want to invert selection or not
        match invert:
            case False:
                data_temp = self.spots

                path = Path(self.selector.verts, closed=True)
                included = path.contains_points(
                    data_temp[["POSITION_X_TF", "POSITION_Y_TF"]]
                )
                data_temp["SELECTION"] = included

                self.data_temp = data_temp

                dmask = data_temp[data_temp["SELECTION"] == True]
            case True:
                data_temp = other_fov.data_temp
                dmask = data_temp[data_temp["SELECTION"] == False]
                selected_data = data_temp[data_temp["SELECTION"] == True]
                self.selector = other_fov.selector

        print(
            f"{dmask["TRACK_ID"].nunique()} of {data_temp["TRACK_ID"].nunique()} tracks selected:"
        )
        print(str(dmask["TRACK_ID"].unique()))
        print("\n")

        comparisons = dmask["TRACK_ID"].unique()
        num_tracks = dmask["TRACK_ID"].nunique()
        comparisons = np.vstack((comparisons, np.zeros(num_tracks))).T

        for index, track in enumerate(comparisons):
            comparisons[index, 1] = len(
                dmask[dmask["TRACK_ID"] == track[0]]
            ) == len(data_temp[data_temp["TRACK_ID"] == track[0]])

        for index, track in enumerate(comparisons):
            if comparisons[index, 1] == False:
                dmask = dmask.drop(dmask[dmask.TRACK_ID == track[0]].index)
                dmask = pd.concat(
                    [dmask, data_temp[data_temp["TRACK_ID"] == track[0]]]
                )
                # set SELECTION = True for repaired parts of track
                data_temp.loc[
                    data_temp["TRACK_ID"] == track[0], "SELECTION"
                ] = True
                print("Track " + str(int(index)) + " incomplete, replaced")
        print("Track repair finished.")

        dmask = dmask.sort_values(by=["TRACK_ID", "FRAME"])

        ### ~~~ MASK TRAJECTORY CLASSIFIER DATA ~~~ ###
        included_tracks = dmask["TRACK_ID"].unique()


        self.selected_parents = self.parents.loc[
            self.parents["ID"].isin(included_tracks)
        ]
        self.selected_normal = self.normal.loc[
            self.normal["PARENT-ID"].isin(included_tracks)
        ]
        self.selected_directed = self.directed.loc[
            self.directed["PARENT-ID"].isin(included_tracks)
        ]
        self.selected_confined = self.confined.loc[
            self.confined["PARENT-ID"].isin(included_tracks)
        ]
        self.selected_sub = self.sub.loc[
            self.sub["PARENT-ID"].isin(included_tracks)
        ]

        self.selected_edges = self.edges.loc[
            self.edges["TRACK_ID"].isin(included_tracks)
        ]
        self.selected_tracks = self.tracks.loc[
            self.tracks["TRACK_ID"].isin(included_tracks)
        ]


        ### ~~~ PLOT TRACKS ~~~ ###

        fig2, ax2 = plt.subplots(figsize=figsize)
        ax2.imshow(
            self.image,
            interpolation="nearest",
            rasterized=False,
            cmap="Greys_r",
        )
        plt.plot(
            np.asarray(self.selector.verts)[:, 0],
            np.asarray(self.selector.verts)[:, 1],
            linewidth=1,
            color="w",
        )

        match invert:
            case True:
                color_in = "r"
                color_out = "b"
            case False:
                color_in = "b"
                color_out = "r"

        ax2 = sns.scatterplot(
            data_temp,
            x=data_temp[data_temp["SELECTION"] == invert]["POSITION_X_TF"],
            y=data_temp[data_temp["SELECTION"] == invert]["POSITION_Y_TF"],
            markers="o",
            s=1,
            legend=False,
            linewidth=0,
            alpha=1,
            color=color_in,
        )
        ax2 = sns.scatterplot(
            data_temp,
            x=dmask["POSITION_X_TF"],
            y=dmask["POSITION_Y_TF"],
            markers="+",
            s=2,
            legend=False,
            linewidth=0,
            alpha=1,
            color=color_out,
        )

        plt.axes = "equal"

        if export == True:
            plt.savefig(
                f"{export_path}/{name}_selection.pdf", bbox_inches="tight", pad_inches=0.1
            )

        plt.show()

        self.selection = dmask
        
    
    def save_selection_data(self, export_path, export=True, name="default"):
        """Save diffusion type data associated with selection"""
        if export == True:
            
            sel_conf = self.selected_confined
            sel_conf["NAME"] = name
            sel_conf.to_csv(f"{export_path}/{name}_selected_confined.csv")    
            
            sel_direc = self.selected_directed
            sel_direc["NAME"] = name
            sel_direc.to_csv(f"{export_path}/{name}_selected_directed.csv")
            
            sel_norm = self.selected_normal
            sel_norm["NAME"] = name
            sel_norm.to_csv(f"{export_path}/{name}_selected_normal.csv")
            
            sel_sub = self.selected_sub
            sel_sub["NAME"] = name
            sel_sub.to_csv(f"{export_path}/{name}_selected_sub.csv")
            
            sel_edges = self.selected_edges
            sel_edges["NAME"] = name
            sel_edges.to_csv(f"{export_path}/{name}_selected_edges.csv")
            
            sel_tracks = self.selected_tracks
            sel_tracks["NAME"] = name
            sel_tracks.to_csv(f"{export_path}/{name}_selected_tracks.csv")
            
            sel_parents = self.selected_parents
            sel_parents["NAME"] = name
            sel_parents.to_csv(f"{export_path}/{name}_selected_parents.csv")
            
            sel_spots = self.selection
            sel_spots["NAME"] = name
            sel_spots.to_csv(f"{export_path}/{name}_selected_spots.csv")
            
            

    def plot_selection(self, export_path, annotate=True, export=False, name="default"):
        # render bounds
        x_min = min(self.selection["POSITION_X_TF"])
        x_max = max(self.selection["POSITION_X_TF"])
        y_min = min(self.selection["POSITION_Y_TF"])
        y_max = max(self.selection["POSITION_Y_TF"])
        x_pad = np.sqrt((x_min - x_max) ** 2) * 0.05
        y_pad = np.sqrt((y_min - y_max) ** 2) * 0.05

        fig = plt.figure()
        track_numbers = self.selection["TRACK_ID"].unique()

        ax = fig.add_subplot(111)
        ax.imshow(
            self.image,
            interpolation="nearest",
            rasterized=False,
            cmap="Greys_r",
        )

        # draw tracks
        for i in track_numbers:
            lines = colored_line(
                np.asarray(
                    self.selection.loc[self.selection["TRACK_ID"] == i][
                        "POSITION_X_TF"
                    ]
                ),
                np.asarray(
                    self.selection.loc[self.selection["TRACK_ID"] == i][
                        "POSITION_Y_TF"
                    ]
                ),
                np.asarray(
                    self.selection.loc[self.selection["TRACK_ID"] == i][
                        "POSITION_T"
                    ]
                ),
                ax,
                linewidth=1,
                cmap="plasma",
            )

            ax.add_collection(lines)

        if annotate == True:
            tracks = self.selection["TRACK_ID"].unique()
            for track in tracks:
                midpoint = int(
                    (
                        self.selection.loc[
                            self.selection["TRACK_ID"] == track, "FRAME"
                        ].idxmax()
                        + self.selection.loc[
                            self.selection["TRACK_ID"] == track, "FRAME"
                        ].idxmin()
                    )
                    / 2
                )
                # get x and y positions at middle point of track
                x_midpoint = self.selection.loc[
                    self.selection["TRACK_ID"] == track, "POSITION_X_TF"
                ][midpoint]
                y_midpoint = self.selection.loc[
                    self.selection["TRACK_ID"] == track, "POSITION_Y_TF"
                ][midpoint]
                ax.text(
                    x_midpoint,
                    y_midpoint,
                    str(track),
                    color="w",
                    fontweight="semibold",
                )

        ax.set_xlim([x_min - x_pad, x_max + x_pad])
        ax.set_ylim([y_max + y_pad, y_min - y_pad])
        # ^^^ y_lim has to be reversed because the image indexes y at the top

        if export == True:
            plt.savefig(
                f"{export_path}/{name}_selection_tracks.pdf",
                bbox_inches="tight",
                pad_inches=0.1,
            )
        plt.show()

    def calculate_msds(
        self,
        export_path,
        export=True,
        name="default",
        lagtime_cutoff=25,
    ):
        # create empty array to hold MSDs (up to lagtime cutoff)
        tracks = self.selection["TRACK_ID"].unique()
        num_tracks = len(tracks)
        msd_all = np.zeros([num_tracks * lagtime_cutoff])

        # loop over each track
        for index, track in tqdm(enumerate(tracks)):
            # call msd_fft() and delete first datapoint for each track (always 0)
            msd_track = np.delete(msd_fft(self.selection, track), (0), axis=0)
            # amend individual track MSD (up to lagtime_cutoff) to list of MSDs
            msd_all[
                index * lagtime_cutoff : index * lagtime_cutoff
                + lagtime_cutoff
            ] = msd_track[0:lagtime_cutoff]

        # reshape msd_all to column vector
        msd_all = np.reshape(msd_all, (len(msd_all), 1))

        # create and append track labels column (if not a subregion selection)
        # if region == False:
        # label_elems = np.arange(0, num_tracks, 1)
        label_reps = np.repeat(tracks, lagtime_cutoff)
        label_reps = np.reshape(label_reps, (len(label_reps), 1))
        # msd_all = np.hstack([msd_all, label_reps])

        # create and append lagtime labels column
        lagtime_elems = (
            np.linspace(1, lagtime_cutoff, lagtime_cutoff) * self.exposure_time
        )
        lagtime_tile = np.tile(lagtime_elems, num_tracks)
        lagtime_tile = np.reshape(lagtime_tile, (len(lagtime_tile), 1))

        msd_all = np.hstack([msd_all, label_reps, lagtime_tile])

        # convert np array to dataframe, add name column
        msd_all = pd.DataFrame(msd_all)
        msd_all.columns = ["MSD", "Track", "Lag Time (sec)"]
        msd_all["Name"] = self.name

        print("MSD Calculation done\n")
        
        if export == True:
            msd_all.to_csv(f"{export_path}/{name}_MSDs.csv")

        self.msds = msd_all

    def fit_msds(self, msds, export_path, export=True, name="default", 
                 fit_function="Backlund"):

        # enforce fit_function arg to allow only valid options
        valid = {"Backlund", "Simple"}
        if fit_function not in valid:
            raise ValueError("fit_function must be one of %r." % valid)

        # get number of tracks
        num_tracks = msds["Track"].nunique()
        # get list of tracks
        tracks = msds["Track"].unique()

        # create empty fit parameter array
        params = np.zeros([num_tracks, 4])

        # create empty lists for track numbers and fov names
        track_list = list()
        name_list = list()

        # remove tqdm() if it's not installed -- it isn't necessary
        # fit track_i using defined function
        # lag times are modified to give natural numbers. Fit functions include
        # exposure time, so the parameters are still dimensionally correct :^)
        match fit_function:
            case "Backlund":
                for index, track in tqdm(enumerate(tracks)):
                    popt, pcov = curve_fit(
                        backlund_fit,
                        np.array(
                            msds.loc[msds["Track"] == track, "Lag Time (sec)"]
                            * 40
                        ),
                        np.array(msds.loc[msds["Track"] == track, "MSD"]),
                        bounds=(
                            (
                                0.0001,  # D* lower bound
                                0,  # alpha lower bound
                                np.min(
                                    msds.loc[msds["Track"] == track, "MSD"]
                                ),  # sigma lower bound
                            ),
                            (
                                5,  # D* upper bound
                                2,  # alpha upper bound
                                1,  # sigma upper bound
                            ),
                        ),
                        maxfev=10000,
                    )

                    # populate ith row of fit parameters
                    params[index, 0:3] = popt

                    # calculate r^2 value for this track fit
                    params[index, 3] = r2_score(
                        np.array(msds.loc[msds["Track"] == track, "MSD"]),
                        backlund_fit(
                            np.array(
                                msds.loc[
                                    msds["Track"] == track,
                                    "Lag Time (sec)",
                                ]
                                * 40
                            ),
                            *popt,
                        ),
                    )

                    # append track number label to track list
                    track_list.append(
                        msds.loc[msds["Track"] == track, "Track"].iloc[0]
                    )
                    # append name to name list
                    name_list.append(
                        msds.loc[msds["Track"] == track, "Name"].iloc[0]
                    )

                params = pd.DataFrame(params)
                params.columns = ["D*", "Alpha", "Sigma", "R^2"]
                params["Track"] = track_list
                params["Name"] = name_list

                self.msd_fits = params
                print("MSD fitting done\n")
                
                if export == True:
                    params.to_csv(f"{export_path}/{name}_MSD_fit_parameters.csv")

            case "Simple":
                print("not finished")

    def filter_fits(
        self,
        export_path,
        export=True,
        name="default",
        D_star_bound=[0, 1.75],
        alpha_bound=[0.0025, 1.9975],
        sigma_bound=[0, 2],
        R2_bound=[0.98, 1],
    ):
        """tbd"""
        filtered_fits = self.msd_fits[
            self.msd_fits["D*"].between(
                left=D_star_bound[0], right=D_star_bound[1]
            )
            & self.msd_fits["Alpha"].between(
                left=alpha_bound[0], right=alpha_bound[1]
            )
            & self.msd_fits["Sigma"].between(
                left=sigma_bound[0], right=sigma_bound[1]
            )
            & self.msd_fits["R^2"].between(left=R2_bound[0], right=R2_bound[1])
        ].reset_index(drop=True)

        # print filtered tracks
        tracks = list(self.msd_fits["Track"])
        filtered_tracks = list(filtered_fits["Track"])
        diff = sorted(list(set(tracks).difference(filtered_tracks)))
        print(f"{len(diff)} filtered out: {diff}")

        # self.msds_filtered = self.msds.loc[
        #     self.msds["Track"].isin(filtered_fits["Track"])
        # ]

        # self.msds_discarded = self.msds.loc[
        #     ~self.msds["Track"].isin(filtered_fits["Track"])
        # ]

        self.msd_fits_filtered = filtered_fits
        print(filtered_fits)
        
        if export == True:
            self.msd_fits_filtered.to_csv(f"{export_path}/{name}_Filtered_MSD_fit_parameters.csv")

    def filter_stationary_msds(self, export_path, name="default", export=False, filter_ratio=1):
        """tbd"""
        tracks = self.msds["Track"].unique()
        kept_tracks = np.zeros(1)
        for index, track in enumerate(tracks):
            x = self.msds.loc[self.msds["Track"] == track, "MSD"]
            x_range = (x.max() - x.min()) / x.median()
            if x_range > filter_ratio:
                kept_tracks = np.hstack((kept_tracks, track))

        kept_tracks = np.delete(kept_tracks, 0, axis=0)

        self.msds_filtered = self.msds.loc[
            self.msds["Track"].isin(kept_tracks)
        ]
        
        if export == True:
            self.msds_filtered.to_csv(f"{export_path}/{name}_Non-stationary_MSDs.csv")

        self.msds_discarded = self.msds.loc[
            ~self.msds["Track"].isin(kept_tracks)
        ]
        
        if export == True:
            self.msds_discarded.to_csv(f"{export_path}/{name}_Stationary_MSDs.csv")

    def plot_msds(self, msds, title=None, log_axes=True, export=False):
        """tbd"""
        fig_msds = plt.subplots()
        ax = sns.lineplot(
            msds,
            x=msds["Lag Time (sec)"],
            y=msds["MSD"],
            hue=msds["Track"],
            legend=False,
        )
        if log_axes == True:
            ax.set_yscale("log")
            ax.set_xscale("log")
        namelist = list(msds["Name"].unique())

        ax.set_title(title)

        if export == True:
            plt.savefig(f"{title}.pdf", bbox_inches="tight", pad_inches=0.1)

        plt.show(fig_msds)

    def compare_msds(
        self,
        msds_1,
        msds_2,
        export_path,
        title=None,
        log_axes=True,
        normalize=False,
        export=False,
    ):
        """tbd"""

        if normalize == True:
            # get median/mean of both datasets' medians (named median for consistency)
            med_1 = msds_1.loc[
                msds_1["Lag Time (sec)"] == 0.025, "MSD"
            ].median()
            med_2 = msds_2.loc[
                msds_2["Lag Time (sec)"] == 0.025, "MSD"
            ].median()
            median = (med_1 + med_2) / 2
            # normalize each track's MSDs to start at 1
            norm_msds_1 = normalize_msds(msds_1, median)
            norm_msds_2 = normalize_msds(msds_2, median)
            combined_data = pd.concat([norm_msds_1, norm_msds_2])

        else:
            combined_data = pd.concat([msds_1, msds_2])

        fig_msds = plt.subplots()
        ax = sns.lineplot(
            combined_data,
            x=combined_data["Lag Time (sec)"],
            y=combined_data["MSD"],
            hue=combined_data["Name"],
            legend=True,
        )
        if log_axes == True:
            ax.set_xscale("log")
            ax.set_yscale("log")
        namelist = list(combined_data["Name"].unique())

        if normalize == True:
            norm_id = " Normalized"
        else:
            norm_id = ""

        ax.set_title(f"{title}{norm_id}")

        if export == True:
            plt.savefig(
                f"{export_path}/{title}{norm_id}.pdf", bbox_inches="tight", pad_inches=0.1
            )

        plt.show(fig_msds)


####################################################### End of Importer() class


def normalize_msds(msds, median):
    """tbd"""
    data = msds.copy()
    tracks = data["Track"].unique()

    for track in tracks:
        # get data for current track
        current = data.loc[data["Track"] == track]
        # get first MSD for current track
        current_first = current.loc[
            current["Lag Time (sec)"] == 0.025, "MSD"
        ].values
        # get difference between median and the first MSD for current track
        diff = median - current_first
        # add difference to all MSDs of current track (in main msds dataframe)
        data.loc[data["Track"] == track, "MSD"] = current["MSD"] + diff

    return data


def backlund_fit(x, D_star, alpha, sigma, exposure_time=0.025):
    """Fitting function for the "Backlund" equation in 2D, which accounts for
    static and dynamic error. DOI:https://doi.org/10.1103/PhysRevE.91.062716

    :param x: Lag number (NOT lagtime, must be series of integers)
    :type x:  float | int
    :param D_star: Effective diffusion coefficient D* (um^2/sec^alpha)
    :type D_star:  float
    :param alpha: Type of diffusion (alpha in (0,2]). Alpha<1 is subdiffusive,
    alpha=1 is freely diffusive, and alpha>1 is superdiffusive
    :type alpha:  float
    :param sigma: Localization error (um^2)
    :type sigma:  float
    :param exposure_time: Camera exposure time (sec)
    :type exposure_time:  float
    """
    func = (
        (4 * D_star * exposure_time**alpha)
        / ((alpha + 2) * (alpha + 1))
        * (
            (x + 1) ** (alpha + 2)
            + (x - 1) ** (alpha + 2)
            - 2 * x ** (alpha + 2)
        )
        - (8 * D_star * exposure_time**alpha) / ((alpha + 2) * (alpha + 1))
        + 4 * sigma**2
    )
    return func


def simple_fit(x, D_star, alpha, sigma, exposure_time=0.025):
    """Fitting function for the "Simple" equation in 2D, which accounts for
    static, but not dynamic, error.

    :param x: Lag number (NOT lagtime, must be series of integers)
    :type x:  float | int
    :param D_star: Effective diffusion coefficient D* (um^2/sec)
    :type D_star:  float
    :param alpha: Type of diffusion (alpha in (0,2]). Alpha<1 is subdiffusive,
    alpha=1 is freely diffusive, and alpha>1 is superdiffusive
    :type alpha:  float
    :param sigma: Localization error (um^2)
    :type sigma:  float
    :param exposure_time: Camera exposure time (sec)
    :type exposure_time:  float
    """
    func = 4 * D_star * (x * exposure_time) ** alpha + 4 * sigma**2
    return func


def autocorrFFT(x):
    """
    calculate autocorrelation of trajectory positions, using the technique for calculating
    MSDs as described in Section 4.1 of http://dx.doi.org/10.1051/sfn/201112010 (Fast Correlation
    Algorithm)
    """
    N = len(x)                  # length of trajectory
    F = np.fft.fft(x, n=N*2)    # zero-padded Fourier transform of trajectory
    PSD = F * F.conjugate()     # power spectral density
    res = np.fft.ifft(PSD)      # inverse Fourier transform of PSD
    res = (res[:N]).real        # resulting autocorrelation; positive semi-definite
    n = N * np.ones(N) - np.arange(0, N)  
    return res / n              # unbiased autocorrelation


### TODO: create proper docstring
def msd_fft(data, track):
    """
    use the result from autocorrFFT() to calculate the MSD of a trajectory, as described in 
    Section 4.2 of http://dx.doi.org/10.1051/sfn/201112010 (Efficient computation of 
    Mean-square displacements)
    """
    R = np.asarray(
        data.loc[(data["TRACK_ID"] == track), ["POSITION_X", "POSITION_Y"]]
    )                               # array of X and Y positions
    N = len(R)                      # length of trajectory
    D = np.square(R).sum(axis=1)    # sum of squares of particle position per frame
    D = np.append(D, 0)
    S_AB = sum([autocorrFFT(R[:, i]) for i in range(R.shape[1])])
                                    # sum of X and Y autocorrelations
    Q = 2 * D.sum()                 # sum of squared particle positions
    S_AABB = np.zeros(N)
    for m in range(N):
        Q = Q - D[m - 1] - D[N - m]
        S_AABB[m] = Q / (N - m) 
    return S_AABB - 2 * S_AB


class SelectFromCollection:
    """
    Select indices from a matplotlib collection using `LassoSelector`.

    Selected indices are saved in the `ind` attribute. This tool fades out the
    points that are not part of the selection (i.e., reduces their alpha
    values). If your collection has alpha < 1, this tool will permanently
    alter the alpha values.

    Note that this tool selects collection objects based on their *origins*
    (i.e., `offsets`).

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        Axes to interact with.
    collection : `matplotlib.collections.Collection` subclass
        Collection you want to select from.
    alpha_other : 0 <= float <= 1
        To highlight a selection, this tool sets all selected points to an
        alpha value of 1 and non-selected points to *alpha_other*.
    """

    def __init__(self, ax, collection, alpha_other=0.3):
        self.canvas = ax.figure.canvas
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError("Collection must have a facecolor")
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(ax, onselect=self.onselect, useblit=True)
        self.ind = []

    def onselect(self, verts):
        path = Path(verts)
        self.verts = verts
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()
        self.selected = np.nonzero(path.contains_points(self.xys))
        return verts

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()


def colored_line(x, y, c, ax, **lc_kwargs):
    """Create line for track plotting"""
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[
        :, np.newaxis, :
    ]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def plot_single_track(data, track: int):
    """
    tbd
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # draw tracks
    lines = colored_line(
        np.asarray(data.loc[(data["TRACK_ID"] == track)]["POSITION_X"]),
        np.asarray(data.loc[(data["TRACK_ID"] == track)]["POSITION_Y"]),
        np.asarray(data.loc[(data["TRACK_ID"] == track)]["POSITION_T"]),
        ax,
        linewidth=1,
        cmap="plasma",
    )

    ax.add_collection(lines)
    ax.axis("equal")
    plt.gca().invert_yaxis()  # flip y-axis to match tracks plotted on image
    plt.autoscale()

    plt.show()


# define new plot selection function to plot data not included in class
def plot_selection_func(
    selection, image, name, linewidth, export_path, annotate=True, export=False
):
    # render bounds
    x_min = min(selection["POSITION_X_TF"])
    x_max = max(selection["POSITION_X_TF"])
    y_min = min(selection["POSITION_Y_TF"])
    y_max = max(selection["POSITION_Y_TF"])
    x_pad = np.sqrt((x_min - x_max) ** 2) * 0.05
    y_pad = np.sqrt((y_min - y_max) ** 2) * 0.05

    fig = plt.figure()
    track_numbers = selection["TRACK_ID"].unique()

    ax = fig.add_subplot(111)
    ax.imshow(
        image,
        interpolation="nearest",
        rasterized=False,
        cmap="Greys_r",
    )

    # draw tracks
    for i in track_numbers:
        lines = colored_line(
            np.asarray(
                selection.loc[selection["TRACK_ID"] == i]["POSITION_X_TF"]
            ),
            np.asarray(
                selection.loc[selection["TRACK_ID"] == i]["POSITION_Y_TF"]
            ),
            np.asarray(
                selection.loc[selection["TRACK_ID"] == i]["POSITION_T"]
            ),
            ax,
            linewidth=linewidth,
            cmap="plasma",
        )

        ax.add_collection(lines)

    if annotate == True:
        tracks = selection["TRACK_ID"].unique()
        for track in tracks:
            midpoint = int(
                (
                    selection.loc[
                        selection["TRACK_ID"] == track, "FRAME"
                    ].idxmax()
                    + selection.loc[
                        selection["TRACK_ID"] == track, "FRAME"
                    ].idxmin()
                )
                / 2
            )
            # get x and y positions at middle point of track
            x_midpoint = selection.loc[
                selection["TRACK_ID"] == track, "POSITION_X_TF"
            ][midpoint]
            y_midpoint = selection.loc[
                selection["TRACK_ID"] == track, "POSITION_Y_TF"
            ][midpoint]
            ax.text(
                x_midpoint,
                y_midpoint,
                str(track),
                color="w",
                fontweight="semibold",
            )

    ax.set_xlim([x_min - x_pad, x_max + x_pad])
    ax.set_ylim([y_max + y_pad, y_min - y_pad])
    ax.set_title(name)
    # ^^^ y_lim has to be reversed because the image indexes y at the top

    if export == True:
        plt.savefig(
            f"{export_path}/{name}_selection.pdf", bbox_inches="tight", pad_inches=0.1
        )
    plt.show()


# %%


#
#                    #######
#                  /       ###
#                 /         ##   #                             #
#                 ##        #   ##                            ##
#                  ###          ##                            ##
#                 ## ###      ######## /###   ###  /###     ########
#                  ### ###   ######## / ###  / ###/ #### / ########
#                    ### ###    ##   /   ###/   ##   ###/     ##
#                      ### /##  ##  ##    ##    ##            ##
#                        #/ /## ##  ##    ##    ##            ##
#                         #/ ## ##  ##    ##    ##            ##
#                          # /  ##  ##    ##    ##            ##
#                /##        /   ##  ##    /#    ##            ##
#               /  ########/    ##   ####/ ##   ###           ##
#              /     #####       ##   ###   ##   ###           ##
#              |
#               \)
#
#
#
#                   #####    ##                               ###
#                ######  /  #### /                             ###
#               /#   /  /   ####/                               ##
#              /    /  /    # #                                 ##
#                  /  /     #                                   ##
#                 ## ##     #      /##  ###  /###     /##       ##
#                 ## ##     #     / ###  ###/ #### / / ###      ##
#                 ## ########    /   ###  ##   ###/ /   ###     ##
#                 ## ##     #   ##    ### ##       ##    ###    ##
#                 ## ##     ##  ########  ##       ########     ### /
#                 #  ##     ##  #######   ##       #######       ##/
#                    /       ## ##        ##       ##
#                /##/        ## ####    / ##       ####    /     #
#               /  #####      ## ######/  ###       ######/     ###
#              /     ##           #####    ###       #####       #
#              #
#               ##
#
#        << After you've already run the code up to this point once :^) >>


# %%
### Name FOV and create subdirectories for data and figures export

FOV_source = "FOV1_post"            ###### <<<--- CHANGE THIS FOR EACH ANALYSIS
treatment_name = "ROCK_REP2"
FOV = f"{treatment_name}_{FOV_source}" 
# PREPOST = "PRE"                   # comment according to which label you want to include
PREPOST = "POST"

### vvv CHECK THIS WHEN SWITCHING DATASETS vvv
source_directory = f'/data/dataset_ROCK_1/{PREPOST}/{FOV_source}'

analysis_root = "ANALYSIS"

if not os.path.exists(f"{directory}/{analysis_root}"):
    os.mkdir(f"{directory}/{analysis_root}")

FOV_root = f"{directory}/{analysis_root}/{FOV}"
if not os.path.exists(f"{FOV_root}"):
    os.mkdir(f"{FOV_root}")

FOV_figure_dir = f"{FOV_root}/FIGURES"
if not os.path.exists(f"{FOV_root}/FIGURES"):
    os.mkdir(f"{FOV_root}/FIGURES")

FOV_data_dir = f"{FOV_root}/DATA"
if not os.path.exists(f"{FOV_root}/DATA"):
    os.mkdir(f"{FOV_root}/DATA")


fov1 = Importer()
fov2 = Importer()

# %%

export_choice = True  # export figures?         ##### CHECK THIS BEFORE CONTINUING
general_fov_name_j = f"{FOV}_Junction"  
general_fov_name_nj = f"{FOV}_Non-Junction"  
dataset_name = f"xxx_J-NJ_{FOV}"                ##### CHECK THIS BEFORE CONTINUING


# %%

fov1.import_all(name=f"{general_fov_name_j}", source_directory=source_directory)

fov2.import_all(name=f"{general_fov_name_nj}", source_directory=source_directory)


# %%

fov1.plot_and_select(show_tracks=False, alpha=0.5, figsize=(20,10))

# %%

fov1.save_selection(export_path=FOV_figure_dir, name=f"{general_fov_name_j}", export=export_choice)

fov1.save_selection_data(export_path=FOV_data_dir, name=general_fov_name_j, export=export_choice)

# %%
if False:
    fov1.plot_selection(
        export_path=FOV_figure_dir, annotate=False, name=f"{general_fov_name_j}", export=export_choice
    )


# --

# fov2.plot_and_select()



fov2.save_selection(
    export_path=FOV_figure_dir,
    name=f"{general_fov_name_nj}",
    other_fov=fov1,
    invert=True,
    export=False,
)

fov2.save_selection_data(export_path=FOV_data_dir, name=general_fov_name_nj, export=export_choice)

if False:
    fov2.plot_selection(
        export_path=FOV_figure_dir, 
        annotate=False, 
        name=f"{general_fov_name_nj}", 
        export=export_choice
    )


fov1.calculate_msds(export_path=FOV_data_dir, export=export_choice, name=general_fov_name_j, lagtime_cutoff=10)
# fov1.msds
fov2.calculate_msds(export_path=FOV_data_dir, export=export_choice, name=general_fov_name_nj, lagtime_cutoff=10)

# --
################# EXPORT MSD DATA TESTING ####################


def export_msds(msds_1, msds_2, data_name, export_path):

    df = pd.concat(
        [
            msds_1,
            msds_2,
        ]
    )
    df["Dataset"] = data_name

    df.to_csv(f"{export_path}/{data_name}_MSDs.csv")


export_msds(fov1.msds, fov2.msds, dataset_name, export_path=FOV_data_dir,)


# --

fov1.filter_stationary_msds(export_path=FOV_data_dir, name=general_fov_name_j, export=True, filter_ratio=0.9)
# fov1.msds_filtered AND fov1.msds_discarded
fov2.filter_stationary_msds(export_path=FOV_data_dir, name=general_fov_name_nj, export=True, filter_ratio=0.9)


### STILL NEED TO CHANGE SAVE DIRECTORY AND NAMING ### TODO

fov1.compare_msds(
    msds_1=fov1.msds_discarded,
    msds_2=fov2.msds_discarded,
    export_path=FOV_figure_dir,
    title=f"{general_fov_name_j} vs {general_fov_name_nj};\nStationary Tracks Only",
    normalize=False,
    export=export_choice,
)

fov1.compare_msds(
    msds_1=fov1.msds_discarded,
    msds_2=fov2.msds_discarded,
    export_path=FOV_figure_dir,
    title=f"{general_fov_name_j} vs {general_fov_name_nj};\nStationary Tracks Only,",
    normalize=True,
    export=export_choice,
)



fov1.compare_msds(
    msds_1=fov1.msds_filtered,
    msds_2=fov2.msds_filtered,
    export_path=FOV_figure_dir,
    title=f"{general_fov_name_j} vs {general_fov_name_nj};\nStationary Tracks Filtered Out",
    normalize=False,
    export=export_choice,
)

fov1.compare_msds(
    msds_1=fov1.msds_filtered,
    msds_2=fov2.msds_filtered,
    export_path=FOV_figure_dir,
    title=f"{general_fov_name_j} vs {general_fov_name_nj};\nStationary Tracks Filtered Out,",
    normalize=True,
    export=export_choice,
)



fov1.compare_msds(
    msds_1=fov1.msds,
    msds_2=fov2.msds,
    export_path=FOV_figure_dir,
    title=f"{general_fov_name_j} vs {general_fov_name_nj};\nStationary Tracks Included",
    normalize=False,
    export=export_choice,
)

fov1.compare_msds(
    msds_1=fov1.msds,
    msds_2=fov2.msds,
    export_path=FOV_figure_dir,
    title=f"{general_fov_name_j} vs {general_fov_name_nj};\nStationary Tracks Included,",
    normalize=True,
    export=export_choice,
)

# --

fov1.fit_msds(fov1.msds, 
              export_path=FOV_data_dir, 
              export=True,
              name=general_fov_name_j, 
              fit_function="Backlund")
# fov1.fit_msds
fov2.fit_msds(fov2.msds, 
              export_path=FOV_data_dir, 
              export=True, 
              name=general_fov_name_nj,
              fit_function="Backlund")
# re-run this method with filtered/discarded msds to overwrite fov1.fit_msds

# --

fov1.filter_fits(export_path=FOV_data_dir, name=general_fov_name_j, export=True)
# fov1.msd_fits_filtered AND
fov2.filter_fits(export_path=FOV_data_dir, name=general_fov_name_nj, export=True)

# --

# %%

#               ___
#              (   )
#      .--.     | |_       .--.      .-..
#    /  _  \   (   __)    /    \    /    \
#   . .' `. ;   | |      |  .-. ;  ' .-,  ;
#   | '   | |   | | ___  | |  | |  | |  . |
#   _\_`.(___)  | |(   ) | |  | |  | |  | |
#  (   ). '.    | | | |  | |  | |  | |  | |
#   | |  `\ |   | ' | |  | '  | |  | |  ' |
#   ; '._,' '   ' `-' ;  '  `-' /  | `-'  '
#    '.___.'     `.__.    `.__.'   | \__.'
#                                  | |
#                                 (___)
#
#               ___
#              (   )
#      .--.     | |_       .--.      .-..
#    /  _  \   (   __)    /    \    /    \
#   . .' `. ;   | |      |  .-. ;  ' .-,  ;
#   | '   | |   | | ___  | |  | |  | |  . |
#   _\_`.(___)  | |(   ) | |  | |  | |  | |
#  (   ). '.    | | | |  | |  | |  | |  | |
#   | |  `\ |   | ' | |  | '  | |  | |  ' |
#   ; '._,' '   ' `-' ;  '  `-' /  | `-'  '
#    '.___.'     `.__.    `.__.'   | \__.'
#                                  | |
#                                 (___)
#
#               ___
#              (   )
#      .--.     | |_       .--.      .-..
#    /  _  \   (   __)    /    \    /    \
#   . .' `. ;   | |      |  .-. ;  ' .-,  ;
#   | '   | |   | | ___  | |  | |  | |  . |
#   _\_`.(___)  | |(   ) | |  | |  | |  | |
#  (   ). '.    | | | |  | |  | |  | |  | |
#   | |  `\ |   | ' | |  | '  | |  | |  ' |
#   ; '._,' '   ' `-' ;  '  `-' /  | `-'  '
#    '.___.'     `.__.    `.__.'   | \__.'
#                                  | |
#                                 (___)
#
