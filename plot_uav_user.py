import os
import json
import numpy as np
import matplotlib.pyplot as plt


def load_uav_trajectories(traj_json_file):
    """读取 UAV 轨迹 {'trajectories': {'uav_0': [...], ...}}"""
    with open(traj_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    trajs = {}
    keys = sorted(data["trajectories"].keys(), key=lambda k: int(k.split("_")[1]))

    for k in keys:
        uid = int(k.split("_")[1])
        arr = np.asarray(data["trajectories"][k], dtype=float).reshape(-1, 2)
        trajs[uid] = arr

    return trajs


def load_user_trajectories(user_json_file):
    """读取用户轨迹 {'trajectories': {'0': [[x,y,*], ...], ...}}"""
    with open(user_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    out = {}
    for uid, seq in data["trajectories"].items():
        arr = np.asarray(seq, dtype=float)
        if arr.shape[1] >= 2:     # 只取前两列
            arr = arr[:, :2]
        out[int(uid)] = arr

    return out


def plot_one_episode(uav_traj_file, user_json, output_dir="trajectory_plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Episode index
    fname = os.path.basename(uav_traj_file)
    ep = None
    if "episode_" in fname:
        try:
            ep = int(fname.split("episode_")[1].split("_")[0])
        except:
            pass

    uav_trajs = load_uav_trajectories(uav_traj_file)
    user_trajs = load_user_trajectories(user_json)

    fig, ax = plt.subplots(figsize=(8, 8))

    # ---- 用户轨迹 ----
    for uid, traj in user_trajs.items():
        ax.plot(traj[:, 0], traj[:, 1], "--", alpha=0.6, label=f"User {uid}")

    # ---- UAV 轨迹 ----
    for uid, traj in uav_trajs.items():
        ax.plot(traj[:, 0], traj[:, 1], "-", linewidth=2.5, label=f"UAV {uid}")
        ax.plot(traj[0, 0], traj[0, 1], "o", markersize=10, color="green")   # start
        ax.plot(traj[-1, 0], traj[-1, 1], "s", markersize=10, color="red")  # end

    ax.set_title(f"Episode {ep}" if ep is not None else "Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend()

    out_file = os.path.join(output_dir, f"ep_{ep}.png")
    plt.savefig(out_file, dpi=200)
    plt.close()

    print(f"✓ Saved {out_file}")


def plot_all(uav_traj_dir, user_json, output_dir="trajectory_plots"):
    """批量处理 episode_*_trajectories.json"""
    files = [f for f in os.listdir(uav_traj_dir) if f.endswith("_trajectories.json")]
    files.sort()

    print(f"发现 {len(files)} 个 episode 轨迹文件")

    for f in files:
        plot_one_episode(
            os.path.join(uav_traj_dir, f),
            user_json,
            output_dir,
        )

    print("✅ 全部绘制完成")


if __name__ == "__main__":
    # 示例
    # 修改为你的路径
    uav_traj_dir = "uav_trajectories/run_20260124_143401"
    user_json = "user_trajectories_hot.json"

    plot_all(uav_traj_dir, user_json)
