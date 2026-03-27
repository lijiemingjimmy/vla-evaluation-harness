from __future__ import annotations

"""SyncEpisodeRunner: waits for inference before stepping."""

from pathlib import Path
import re
import itertools
from typing import Any

import imageio.v2 as imageio
import numpy as np

from vla_eval.benchmarks.base import Benchmark
from vla_eval.runners.base import EpisodeRunner
from vla_eval.types import EpisodeResult, Task


def _extract_frame_from_obs(obs: Any) -> np.ndarray | None:
    """
    尽量从 observation 里提取一张 RGB 图像用于录像。

    优先顺序：
    1. obs["images"] 中第一张图
    2. obs["image"] / obs["rgb"] / obs["agentview_image"] / obs["agentview_rgb"]

    返回:
        HWC, uint8, RGB 的 numpy 数组；如果拿不到则返回 None
    """
    frame = None

    if isinstance(obs, dict):
        # 常见情况：obs["images"] 是一个 dict
        images = obs.get("images")
        if isinstance(images, dict) and len(images) > 0:
            # 优先拿常见的 agentview key
            for preferred_key in ["agentview_rgb", "agentview_image", "rgb", "image"]:
                if preferred_key in images:
                    frame = images[preferred_key]
                    break

            # 没找到就拿第一张
            if frame is None:
                for _, img in images.items():
                    frame = img
                    break

        # 退而求其次，直接看顶层字段
        if frame is None:
            for key in ["image", "rgb", "agentview_image", "agentview_rgb"]:
                if key in obs:
                    frame = obs[key]
                    break

    if frame is None:
        return None

    frame = np.asarray(frame)

    # 如果是 CHW，转成 HWC
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    # 灰度图扩成 3 通道
    if frame.ndim == 2:
        frame = np.stack([frame, frame, frame], axis=-1)

    # RGBA -> RGB
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]

    # 非 uint8，转成 uint8
    if frame.dtype != np.uint8:
        frame = np.asarray(frame, dtype=np.float32)

        # 常见情况是 [0, 1]
        if frame.max() <= 1.0:
            frame = np.clip(frame, 0.0, 1.0) * 255.0
        else:
            frame = np.clip(frame, 0.0, 255.0)

        frame = frame.astype(np.uint8)

    return frame


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(name)).strip("_")[:100]


def _save_episode_video(
    frames: list[np.ndarray],
    *,
    task_name: str,
    episode_result: dict[str, Any],
    out_dir: str | Path = "results/videos",
    fps: int = 10,
) -> str | None:
    if not frames:
        return None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_id = episode_result.get("episode_id")
    if episode_id:
        file_stem = _safe_name(str(episode_id))
    else:
        file_stem = _safe_name(task_name)

    out_path = out_dir / f"{file_stem}.mp4"
    imageio.mimsave(out_path, frames, fps=fps)
    return str(out_path)


class SyncEpisodeRunner(EpisodeRunner):
    """Synchronous episode runner: one observation → one action per step.

    Episode flow:
        1. ``benchmark.start_episode(task)``
        2. ``benchmark.get_observation()`` → initial observation.
        3. ``conn.start_episode(task_info)``
        4. Step loop (up to ``max_steps``):
           a. ``conn.act(obs)`` → action from model server
           b. ``benchmark.apply_action(action)``
           c. If ``benchmark.is_done()``: break
           d. ``benchmark.get_observation()`` → next observation
        5. ``conn.end_episode()``
    """

    async def run_episode(
        self,
        benchmark: Benchmark,
        task: Task,
        conn: Any,  # Connection
        *,
        max_steps: int | None = None,
    ) -> EpisodeResult:
        """Run a synchronous episode."""
        await benchmark.start_episode(task)
        obs_dict = await benchmark.get_observation()

        # 视频帧缓存
        frames: list[np.ndarray] = []

        first_frame = _extract_frame_from_obs(obs_dict)
        if first_frame is not None:
            frames.append(first_frame)

        # Send only serializable task info to the model server
        task_info = {k: v for k, v in task.items() if isinstance(v, (str, int, float, bool, list))}
        await conn.start_episode({"task": task_info})

        # 给视频命名用
        task_name = task_info.get("task", task_info.get("instruction", str(task_info)))
        if not task_name:
            task_name = str(task)

        steps = range(max_steps) if max_steps is not None else itertools.count()

        step = -1
        for step in steps:
            action = await conn.act(obs_dict)
            await benchmark.apply_action(action)

            if await benchmark.is_done():
                break

            obs_dict = await benchmark.get_observation()

            frame = _extract_frame_from_obs(obs_dict)
            if frame is not None:
                frames.append(frame)

        elapsed = await benchmark.get_time()
        episode_result = await benchmark.get_result()
        episode_result["steps"] = step + 1
        episode_result["elapsed_sec"] = elapsed

        # 保存视频
        video_path = _save_episode_video(
            frames,
            task_name=task_name,
            episode_result=episode_result,
            out_dir="results/videos",
            fps=10,
        )
        if video_path is not None:
            episode_result["video_path"] = video_path
            print(f"[video] saved to {video_path}")

        await conn.end_episode(episode_result)
        return episode_result