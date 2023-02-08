import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.gridspec as gridspec


def calculate_and_plot_errors(
    df: pd.DataFrame,
    TRIAL_TAG: str,
    first_sample: int = 0,
    max_var:int = 200,
    max_plots: int = 5,
    screen_res: tuple = (1920, 1080),
    verbose:bool=True,
):
    """_summary_

    Args:
        df (Pandas DataFrame): Data
        TRIAL_TAG (str): Type of trial (`validation-stimulus` or `fixation-stimulus`)
        first_sample (int, optional): First sample fo evaluate. Useful for filtering. Defaults to 0.
        max_plots (int, optional): Max number of plots. Defaults to 5.
        max_var (int, optional): Max variance accepted in trial. Default to 200.
        screen_res (tuple, optional): Screen Resolution of experiment. Defaults to (1920, 1080).

    Returns:
        pd.DataFrame: Analyzed data with columns:
            "trials": trials
            "first_sample": first_sample
            "presented_point": presented_points
            "time_between_samples_mean": time_between_samples_mean
            "time_between_samples_std": time_between_samples_std
            "sampling_rate_mean": sampling_rate_mean
            "sampling_rate_std": sampling_rate_std
            "last_time_sample": last_time_sample
            "horizontal_errors_pxs_mean": horizontal_errors_pxs_mean,
            "horizontal_errors_pxs_std": horizontal_errors_pxs_std,
            "vertical_errors_pxs_mean": vertical_errors_pxs_mean,
            "vertical_errors_pxs_std": vertical_errors_pxs_std,
            "total_errors_pxs_mean": total_errors_pxs_mean,
            "total_errors_pxs_std": total_errors_pxs_std
            "metadata": "name-id","webcam-id","computer-id","web-browser","operating-system"
    """
    trials = []
    presented_points = []
    time_between_samples_mean = []
    time_between_samples_std = []
    sampling_rate_min = []
    sampling_rate_max = []
    sampling_rate_mean = []
    sampling_rate_std = []
    last_time_sample = []
    horizontal_errors_pxs_mean = []
    horizontal_errors_pxs_std = []
    vertical_errors_pxs_mean = []
    vertical_errors_pxs_std = []
    total_errors_pxs_mean = []
    total_errors_pxs_std = []
    metadata = df["response"].iloc[0]
    xs_all = []
    ys_all = []
    ts_all = []

    center_x = df[df["trial-tag"] == TRIAL_TAG]["center_x"].iloc[0]
    center_y = df[df["trial-tag"] == TRIAL_TAG]["center_y"].iloc[0]

    nv = df[df["trial-tag"] == TRIAL_TAG]["validation-id"].values
    xv = df[df["trial-tag"] == TRIAL_TAG]["start-x"].values + center_x
    yv = df[df["trial-tag"] == TRIAL_TAG]["start-y"].values + center_y

    k = 0
    for d in df[df["trial-tag"] == TRIAL_TAG]["webgazer_data"].map(eval):

        xs = []
        ys = []
        ts = []
        for i in d:
            xs.append(i["x"])
            ys.append(i["y"])
            ts.append(i["t"])

        trials.append(nv[k])
        presented_points.append((xv[k], yv[k]))
        time_between_samples_mean.append(np.mean(np.diff(ts)))
        time_between_samples_std.append(np.std(np.diff(ts)))
        sampling_rate_max.append(np.max(1000 / np.diff(ts)))
        sampling_rate_min.append(np.min(1000 / np.diff(ts)))
        sampling_rate_mean.append(np.mean(1000 / np.diff(ts)))
        sampling_rate_std.append(np.std(1000 / np.diff(ts)))
        last_time_sample.append(max(ts))

        xs = np.array(xs)
        ys = np.array(ys)
        ts = np.array(ts)

        # Add before fitering
        xs_all.append(xs)
        ys_all.append(ys)
        ts_all.append(ts)
        
        xs = xs[ts > first_sample]
        ys = ys[ts > first_sample]
        ts = ts[ts > first_sample]

        if np.std(xs) > max_var or np.std(ys) > max_var:
            ex = np.nan
            ey = np.nan
            ee = np.nan
        else:
            ex = abs(xs - xv[k])
            ey = abs(ys - yv[k])
            ee = np.sqrt(ex**2 + ey**2)

        horizontal_errors_pxs_mean.append(np.mean(ex))
        horizontal_errors_pxs_std.append(np.std(ex))
        vertical_errors_pxs_mean.append(np.mean(ey))
        vertical_errors_pxs_std.append(np.std(ey))
        total_errors_pxs_mean.append(np.mean(ee))
        total_errors_pxs_std.append(np.std(ee))


        gs = gridspec.GridSpec(2, 4)
        gs.update(wspace=2)
        ax1 = plt.subplot(
            gs[0, :2],
        )
        ax1.set_title("Gaze estimation")
        ax1.scatter(xv[k], yv[k], c="c")
        ax1.vlines(screen_res[0] / 2, 0, screen_res[1], "k")
        ax1.hlines(screen_res[1] / 2, 0, screen_res[0], "k")
        colored_plot = ax1.scatter(xs, ys, c=ts)
        plt.colorbar(colored_plot, ax=ax1)
        ax1.scatter(xv[k], yv[k], c="b")
        ax1.set_xlim(0, screen_res[0])
        ax1.set_ylim(0, screen_res[1])

        ax2 = plt.subplot(gs[0, 2:])
        ax2.set_title("X coordinate over time")
        ax2.plot(ts, xs, "k.")
        ax2.set_ylim(0, center_x * 2)
        ax2.hlines(center_x, 0, max(ts), "k")

        ax3 = plt.subplot(gs[1, 1:3])
        ax3.set_title("y coordinate over time")
        ax3.plot(ts, ys, "k.")
        ax3.set_ylim(0, center_y * 2)
        ax3.hlines(center_y, 0, max(ts), "k")

        if verbose:
            plt.show()
            print("Horizontal error (pxs) = %.0f +- %.0f " % (np.mean(ex), np.std(ex)))
            print("Vertical error (pxs) = %.0f +- %.0f " % (np.mean(ey), np.std(ey)))
            print("Total error (pxs) = %.0f +- %.0f " % (np.mean(ee), np.std(ee)))
            print("Validation point (%d): %d, %d" % (nv[k], xv[k], yv[k]))
            print(
                "Time between samples (ms) = %.0f +- %.0f "
                % (np.mean(np.diff(ts)), np.std(np.diff(ts)))
            )
            print(
                "Sampling rate (Hz) = %.0f +- %.0f "
                % (np.mean(1000 / np.diff(ts)), np.std(1000 / np.diff(ts)))
            )
            print(
                "Range Sampling rate (Hz) = [%.0f %.0f]"
                % (np.min(1000 / np.diff(ts)), np.max(1000 / np.diff(ts)))
            )
            print("Last time sample = %d" % max(ts))

        print("k:", k)
        k += 1
        if k == max_plots:
            break

    return pd.DataFrame(  # TODO: Agregar data del virtual chin rest
        {
            "trials": trials,
            "first_sample": first_sample,
            "max_var": max_var,
            "presented_point": presented_points,
            "time_between_samples_mean": time_between_samples_mean,
            "time_between_samples_std": time_between_samples_std,
            "sampling_rate_mean": sampling_rate_mean,
            "sampling_rate_std": sampling_rate_std,
            "sampling_rate_min": sampling_rate_min,
            "sampling_rate_max": sampling_rate_max,
            "last_time_sample": last_time_sample,
            "horizontal_errors_pxs_mean": horizontal_errors_pxs_mean,
            "horizontal_errors_pxs_std": horizontal_errors_pxs_std,
            "vertical_errors_pxs_mean": vertical_errors_pxs_mean,
            "vertical_errors_pxs_std": vertical_errors_pxs_std,
            "total_errors_pxs_mean": total_errors_pxs_mean,
            "total_errors_pxs_std": total_errors_pxs_std,
            "webgazer_x": xs_all,
            "webgazer_y": ys_all,
            "webgazer_t": ts_all,
            "metadata": metadata,
        }
    )


def evaluate_experiment_instances(files):
    for file in files:
        df_res = pd.read_csv(file)
        print(file)
        print(
            f"Error: {df_res['total_errors_pxs_mean'].mean():.2f} +- {df_res['total_errors_pxs_mean'].std():.2f}"
        )
        print(
            f"Sampling rate: {df_res['sampling_rate_mean'].mean():.2f} +- {df_res['sampling_rate_std'].mean():.2f}"
        )
        print("---")


def get_rastoc_events(df, event="rastoc:stillness-position-lost"):
    return [i for i in eval(df["events"].iloc[-2]) if i["event_name"] == event]
