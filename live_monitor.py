from collections import deque

import matplotlib.pyplot as plt

from engine.detection_engine import DetectionEngine, simulate_realtime_sample


CONDITIONS = ("apnea", "hypoxia", "arrhythmia", "sepsis")
COLORS = {
    "apnea": "tab:blue",
    "hypoxia": "tab:red",
    "arrhythmia": "tab:green",
    "sepsis": "tab:orange",
}


def run_live_monitor(window_size=120, max_steps=None):
    engine = DetectionEngine()
    time_points = deque(maxlen=window_size)
    history = {name: deque(maxlen=window_size) for name in CONDITIONS}

    plt.ion()
    fig, ax = plt.subplots(figsize=(11, 6))
    lines = {}
    for name in CONDITIONS:
        (line,) = ax.plot([], [], label=name, color=COLORS[name], linewidth=2)
        lines[name] = line

    ax.set_title("Live Detection Probabilities")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    t = 0
    try:
        while max_steps is None or t < max_steps:
            sample = simulate_realtime_sample(t)
            output = engine.process_sample(sample)

            time_points.append(t)
            for name in CONDITIONS:
                prob = output[name]["probability"]
                history[name].append(float(prob) if prob is not None else 0.0)
                lines[name].set_data(list(time_points), list(history[name]))

            if time_points:
                xmin = time_points[0]
                xmax = time_points[-1] if time_points[-1] > xmin else xmin + 1
                ax.set_xlim(xmin, xmax)

            dominant = output["dominant"]["state"]
            ax.set_title(f"Live Detection Probabilities | t={t} | dominant={dominant}")
            fig.canvas.draw_idle()
            plt.pause(0.01)

            t += 1
    except KeyboardInterrupt:
        print("Stopping live monitor...")
    finally:
        engine.shutdown()
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    run_live_monitor()
