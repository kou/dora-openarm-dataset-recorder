# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Node to record data from OpenArm and cameras as OpenArm dataset."""

import argparse
from dataclasses import dataclass, field
import copy
import dora
import os
import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.typing import ArrayLike
import shutil
import yaml
import time


@dataclass
class Episode:
    """Episode related data."""

    number: int = 0
    success: bool = False
    task_index: int = 0

    right_action_timestamps: ArrayLike = field(default_factory=list)
    right_actions: ArrayLike = field(default_factory=list)
    right_observation_timestamps: ArrayLike = field(default_factory=list)
    right_observations: ArrayLike = field(default_factory=list)
    left_action_timestamps: ArrayLike = field(default_factory=list)
    left_actions: ArrayLike = field(default_factory=list)
    left_observation_timestamps: ArrayLike = field(default_factory=list)
    left_observations: ArrayLike = field(default_factory=list)


class EpisodeWriter:
    """Writer an episode."""

    def __init__(self, directory, episode):
        """Initialize variables."""
        self._directory = directory
        self._episode = episode
        self._base_directory = self._directory / "episodes" / str(self._episode.number)

    def write_camera_image(self, name, image, timestamp, format):
        """Write an image from a camera."""
        output_path = self._base_directory / "cameras" / name / f"{timestamp}.{format}"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as output:
            # Using pa.PythonFile here is for zero-copy. We can't
            # write pa.Buffer data to Python's IO directory. If we use
            # Python's IO, we need to copy data in pa.Buffer as
            # Python's bytes. We want to avoid it.
            with pa.PythonFile(output) as pa_output:
                pa_output.write(image.buffers()[1])

    def finish(self):
        """Write all pending data."""
        if self._episode.right_actions:
            self._write_positions(
                self._base_directory / "action" / "arms" / "right" / "qpos.parquet",
                self._episode.right_action_timestamps,
                self._episode.right_actions,
            )
        if self._episode.right_observations:
            self._write_positions(
                self._base_directory / "obs" / "arms" / "right" / "qpos.parquet",
                self._episode.right_observation_timestamps,
                self._episode.right_observations,
            )
        if self._episode.left_actions:
            self._write_positions(
                self._base_directory / "action" / "arms" / "left" / "qpos.parquet",
                self._episode.left_action_timestamps,
                self._episode.left_actions,
            )
        if self._episode.left_observations:
            self._write_positions(
                self._base_directory / "obs" / "arms" / "left" / "qpos.parquet",
                self._episode.left_observation_timestamps,
                self._episode.left_observations,
            )

    def cancel(self):
        """Cancel this episode."""
        shutil.rmtree(self._base_directory, ignore_errors=True)

    def _write_positions(self, output_path, timestamps, positions):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        list_type = pa.list_(pa.float32())
        table = pa.table(
            {
                "timestamp": pa.array(timestamps, type=pa.timestamp("ns")),
                "positions": pa.array(positions, type=list_type),
            }
        )
        pq.write_table(table, output_path)


class DatasetWriter:
    """Writer a dataset."""

    _VERSION = "0.1.0"

    def __init__(self, directory, name, metadata_template):
        """Initialize variables."""
        self._directory = directory
        self._name = name
        self._metadata_template = metadata_template or {}
        self._base_directory = self._directory / name
        self._episode_results = []
        shutil.rmtree(self._base_directory, ignore_errors=True)
        self._base_directory.mkdir(parents=True)

    def create_episode_writer(self, episode):
        """Create a writer for the given episode."""
        return EpisodeWriter(self._base_directory, episode)

    def finish_episode(self, episode):
        """Add a finished episode to the writer."""
        self._episode_results.append(
            dict(
                id=str(episode.number),
                success=episode.success,
                task_index=episode.task_index,
            )
        )
        self._write_metadata_file()

    def _write_metadata_file(self):
        metadata = copy.deepcopy(self._metadata_template)
        metadata["version"] = self._VERSION
        metadata["episodes"] = self._episode_results
        docker_image = os.getenv("IMAGE")
        if docker_image:
            metadata["docker_image"] = docker_image
        output_path = self._base_directory / "metadata.yaml"
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)


def main():
    """Collect data and record them."""
    parser = argparse.ArgumentParser(description="Record data as OpenArm dataset")
    parser.add_argument(
        "--directory",
        default=os.getenv("DIRECTORY", os.getcwd()),
        help="The output directory",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--name",
        default=os.getenv("NAME", "dataset"),
        help="The dataset name",
        type=str,
    )
    parser.add_argument(
        "--metadata-file",
        default=os.getenv("METADATA_FILE"),
        help="The metadata file",
        type=pathlib.Path,
    )
    args = parser.parse_args()

    node = dora.Node()
    if args.metadata_file is None:
        metadata_template = None
    else:
        with open(args.metadata_file, encoding="utf-8") as f:
            metadata_template = yaml.safe_load(f)
    dataset_writer = DatasetWriter(args.directory, args.name, metadata_template)
    episode = None
    episode_writer = None

    for event in node:
        if event["type"] != "INPUT":
            continue

        event_id = event["id"]
        if event_id == "command":
            command = event["value"][0].as_py()
            if command == "start":
                episode = Episode()
                episode.number = event["metadata"].get("episode_number", 0)
                episode.task_index = event["metadata"].get("task_index", 0)
                episode_writer = dataset_writer.create_episode_writer(episode)
            elif command in ("success", "fail"):
                if command == "success":
                    episode.success = True
                episode_writer.finish()
                dataset_writer.finish_episode(episode)
                episode = None
                episode_writer = None
            elif command == "cancel":
                episode_writer.cancel()
                episode = None
                episode_writer = None
            elif command == "quit":
                if episode is not None:
                    episode_writer.finish()
                    dataset_writer.finish_episode(episode)
                    episode = None
                    episode_writer = None
                break
            continue

        # Main process
        if episode is None:
            continue
        if event_id.startswith("arm_"):
            value = event["value"]
            if isinstance(value, pa.StructArray):
                position = value.field("new_position")
                # TODO: Use timestamp in the given event
                timestamp = time.time_ns()
            else:
                position = value
                timestamp = event["metadata"]["timestamp"]

            # arm_right_action ->
            # right_action
            key_prefix = event_id.removeprefix("arm_")
            # right_action ->
            # right_actions
            positions_key = f"{key_prefix}s"
            getattr(episode, positions_key).append(position)
            # right_action ->
            # right_action_timestamps
            timestamps_key = f"{key_prefix}_timestamps"
            getattr(episode, timestamps_key).append(timestamp)
        elif event_id.startswith("camera_"):
            name = event_id.removeprefix("camera_")
            image = event["value"]
            format = event["metadata"]["encoding"]
            # TODO: Use timestamp in the given event
            timestamp = time.time_ns()
            episode_writer.write_camera_image(name, image, timestamp, format)


if __name__ == "__main__":
    main()
