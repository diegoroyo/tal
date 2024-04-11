import argparse
import datetime
import multiprocessing
import os
import shutil
import sys
import platform
from functools import partial
from multiprocessing.queues import Queue
from typing import Optional, Any

import numpy as np
import yaml
from tqdm import tqdm

import tal
from tal.config import local_file_path
from tal.enums import FileFormat, GridFormat, HFormat
from tal.io.capture_data import NLOSCaptureData
from tal.log import LogLevel, TQDMLogRedirect, log
from tal.render.util import import_mitsuba_backend, Point2D, expand


# Entry point (kept as a free function to avoid breaking API)
def render_nlos_scene(config_path, args, num_retries=0):
    renderer = Renderer(config_path, args)
    renderer.render_nlos_scene(num_retries)


class StdoutQueue(Queue):
    def __init__(self, *args, **kwargs):
        ctx = multiprocessing.get_context()
        super(StdoutQueue, self).__init__(*args, **kwargs, ctx=ctx)

    def write(self, msg):
        self.put(msg)


class SimulationParams:
    def __init__(
        self,
        defines: dict[str, Optional[Any]],
        nlos_scene_xml_path: str,
        hdr_path: str,
        logfile: Any,
        render_name: str,
        args: argparse.Namespace,
    ):
        self.queue = StdoutQueue()
        self.defines = defines
        self.nlos_scene_xml_path = nlos_scene_xml_path
        self.hdr_path = hdr_path
        self.logfile = logfile
        self.render_name = render_name
        self.args = args

    def make_mitsuba_call(self, backend):
        # NOTE: something here has a memory leak (probably Mitsuba-related)
        # We run Mitsuba in a separate process to ensure that the leaks do not add up
        # as they can fill your RAM in exhaustive scans
        run_mitsuba_f = partial(
            backend.run_mitsuba,
            self.nlos_scene_xml_path,
            self.hdr_path,
            self.defines,
            self.render_name,
            sys.stdout,
            self.args,
            queue=self.queue,
        )

        return run_mitsuba_f


class Renderer:
    def __init__(self, config_path: str, args: argparse.Namespace) -> None:
        self.config_path = os.path.abspath(config_path)
        self.args = args
        assert os.path.exists(self.config_path), f"{self.config_path} does not exist"

        try:
            self._load_scene_config()
        except yaml.YAMLError as e:
            raise AssertionError(f"Invalid YAML format in TAL config file: {e}") from e

        self.steady_xml, self.nlos_xml = self.mitsuba_backend.get_scene_xml(
            self.scene_config, random_seed=args.seed, quiet=args.quiet
        )

        try:
            self._setup_directories()
        except OSError as e:
            raise AssertionError(f"Invalid permissions: {e}") from e

    def render_nlos_scene(self, num_retries: int = 0):
        self._render_nlos_scene_impl(num_retries)

    def _render_nlos_scene_impl(self, num_retries: int = 0):
        self.steady_scene_xml_path = os.path.join(
            self._get_directory("root_dir"), "steady_scene.xml"
        )
        with open(self.steady_scene_xml_path, "w") as f:
            f.write(self.steady_xml)

        self.nlos_scene_xml_path = os.path.join(
            self._get_directory("root_dir"), "nlos_scene.xml"
        )
        with open(self.nlos_scene_xml_path, "w") as f:
            f.write(self.nlos_xml)

        relay_wall = next(
            filter(
                lambda g: g["name"] == self.scene_config["relay_wall"],
                self.scene_config["geometry"],
            )
        )
        assert (
            "rot_degrees_x" not in relay_wall
            and "rot_degrees_y" not in relay_wall
            and "rot_degrees_z" not in relay_wall
        ), "Relay wall rotation is NYI"

        self.laser_lookats = self._setup_laser_params(self.scan_type)

        displacement, sensor_grid_coords = self._compute_sensor_grid_coords(relay_wall)

        laser_grid_coords = self._compute_laser_grid_coords(
            displacement, relay_wall, self.scan_type
        )

        # TODO(diego): rotate [0, 0, 1] by rot_degrees_x (assumes RW is a plane)
        # or use a more generalist approach
        sensor_grid_normals = expand(
            np.array([0, 0, 1]),
            self.scene_config["sensor_width"],
            self.scene_config["sensor_height"],
        )
        laser_grid_normals = expand(
            np.array([0, 0, 1]),
            self.scene_config["laser_width"],
            self.scene_config["laser_height"],
        )

        if self.args.do_steady_renders:
            self._render_steady("back_view", 0)
            self._render_steady("side_view", 1)

        pbar = tqdm(
            enumerate(self.laser_lookats),
            desc=f"Rendering {self.scene_config['name']} ({self.scan_type})...",
            file=TQDMLogRedirect(),
            ascii=True,
            total=len(self.laser_lookats),
        )
        for i, laser_lookat in enumerate(self.laser_lookats):
            self._run_single_simulation(pbar, laser_lookat, i)
        pbar.close()

        if self.args.dry_run:
            return

        if not self.args.quiet:
            log(LogLevel.INFO, "Merging partial results...")

        capture_data = self._fill_capture_metadata(
            laser_grid_normals,
            laser_grid_coords,
            sensor_grid_normals,
            sensor_grid_coords,
        )

        self._fill_capture_data(capture_data, num_retries)

        hdf5_path = os.path.join(
            self._get_directory("root_dir"), f"{self.scene_config['name']}.hdf5"
        )
        tal.io.write_capture(capture_data, hdf5_path, file_format=FileFormat.HDF5_TAL)

        if not self.args.quiet:
            log(LogLevel.INFO, f"Stored result in {hdf5_path}")

        self._cleanup()

    def _fill_capture_metadata(
        self,
        laser_grid_normals,
        laser_grid_coords,
        sensor_grid_normals,
        sensor_grid_coords,
    ):
        capture_data = NLOSCaptureData()
        capture_data.sensor_xyz = np.array(
            [
                self.scene_config["sensor_x"],
                self.scene_config["sensor_y"],
                self.scene_config["sensor_z"],
            ],
            dtype=np.float32,
        )
        capture_data.sensor_grid_xyz = sensor_grid_coords
        capture_data.sensor_grid_normals = sensor_grid_normals
        capture_data.sensor_grid_format = GridFormat.X_Y_3
        capture_data.laser_xyz = np.array(
            [
                self.scene_config["laser_x"],
                self.scene_config["laser_y"],
                self.scene_config["laser_z"],
            ],
            dtype=np.float32,
        )
        capture_data.laser_grid_xyz = laser_grid_coords
        capture_data.laser_grid_normals = laser_grid_normals
        capture_data.laser_grid_format = GridFormat.X_Y_3
        # NOTE(diego): we do not store volume information for now
        # capture_data.volume_format = VolumeFormat.X_Y_Z_3
        capture_data.delta_t = self.scene_config["bin_width_opl"]
        capture_data.t_start = self.scene_config["start_opl"]
        capture_data.t_accounts_first_and_last_bounces = self.scene_config[
            "account_first_and_last_bounces"
        ]
        capture_data.scene_info = {
            "tal_version": tal.__version__,
            "config": self.scene_config,
            "args": vars(self.args),
        }
        return capture_data

    def _compute_laser_grid_coords(self, displacement, relay_wall, scan_type):
        if scan_type == "single":
            laser_lookat = self.laser_lookats[0]

            px = relay_wall["scale_x"] * (
                (laser_lookat.x / self.scene_config["sensor_width"]) * 2 - 1
            )
            py = relay_wall["scale_y"] * (
                (laser_lookat.y / self.scene_config["sensor_height"]) * 2 - 1
            )
            laser_grid_coords = np.array(
                [
                    [
                        [px, py, 0],
                    ]
                ],
                dtype=np.float32,
            )
        else:
            laser_grid_coords = Renderer._get_grid_xyz(
                self.scene_config["laser_width"],
                self.scene_config["laser_height"],
                relay_wall["scale_x"],
                relay_wall["scale_y"],
            )
            laser_grid_coords += displacement

        return laser_grid_coords

    def _compute_sensor_grid_coords(self, relay_wall):
        # TODO(diego): rotate
        displacement = np.array(
            [
                relay_wall["displacement_x"],
                relay_wall["displacement_y"],
                relay_wall["displacement_z"],
            ]
        )
        sensor_grid_coords = Renderer._get_grid_xyz(
            self.scene_config["sensor_width"],
            self.scene_config["sensor_height"],
            relay_wall["scale_x"],
            relay_wall["scale_y"],
        )
        sensor_grid_coords += displacement
        return displacement, sensor_grid_coords

    def _load_scene_config(self) -> None:
        scene_config = yaml.safe_load(open(self.config_path, "r")) or dict()

        if scene_config.get("mitsuba_variant", "").startswith("cuda"):
            assert len(self.args.gpus) > 0, (
                "You must specify at least one GPU to use CUDA. "
                "Use tal --gpu <id1> <id2> ..."
            )
            gpu_ids = ",".join(map(str, self.args.gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.mitsuba_backend = import_mitsuba_backend()

        scene_defaults = yaml.safe_load(
            open(local_file_path("render/scene_defaults.yaml"), "r")
        )
        scene_defaults["mitsuba_variant"] = self.mitsuba_backend.get_default_variant()

        self.scene_config = {**scene_defaults, **scene_config}

        if not self.args.dry_run:
            self.mitsuba_backend.set_variant(self.scene_config["mitsuba_variant"])

    def _setup_directories(self) -> None:
        config_dir, config_filename = os.path.split(self.config_path)

        in_progress = False
        self.progress_file_path = os.path.join(config_dir, "IN_PROGRESS")

        if os.path.exists(self.progress_file_path):
            with open(self.progress_file_path, "r") as f:
                progress_folder = f.read()
            if (
                os.path.exists(os.path.join(config_dir, progress_folder))
                and not self.args.quiet
            ):
                in_progress = True
            else:
                log(LogLevel.INFO, "The IN_PROGRESS file is stale, removing it...")
                os.remove(self.progress_file_path)

        if in_progress and not self.args.quiet:
            log(
                LogLevel.INFO,
                f"Found a render in progress ({progress_folder}), continuing...",
            )

        if not in_progress:
            progress_folder = datetime.datetime.now().strftime(r"%Y%m%d-%H%M%S")

        root_dir = os.path.join(config_dir, progress_folder)
        self.directories = {
            "root_dir": root_dir,
            "partial_results_dir": os.path.join(root_dir, "partial"),
            "steady_dir": os.path.join(root_dir, "steady"),
            "log_dir": os.path.join(root_dir, "logs"),
        }

        if not in_progress:
            for dir_name in self.directories.values():
                os.mkdir(dir_name)

            shutil.copy(
                self.config_path, os.path.join(root_dir, f"{config_filename}.old")
            )

        with open(self.progress_file_path, "w") as f:
            f.write(progress_folder)

    def _setup_laser_params(self, scan_type: str) -> list[Point2D]:
        laser_lookats = []

        if scan_type == "single":
            self.laser_width = 1
            self.laser_height = 1

            laser_lookat = Point2D(
                self.scene_config["laser_lookat_x"]
                or self.scene_config["sensor_width"] / 2,
                self.scene_config["laser_lookat_y"]
                or self.scene_config["sensor_height"] / 2,
            )
            laser_lookats.append(laser_lookat)

        elif scan_type == "exhaustive" or scan_type == "confocal":
            assert not (
                scan_type == "confocal"
                and (
                    self.laser_width != self.scene_config["sensor_width"]
                    or self.laser_height != self.scene_config["sensor_height"]
                )
            ), "If using scan_type=confocal, sensor_{width|height} must match laser_{width|height}"
            for y in range(self.laser_height):
                for x in range(self.laser_width):
                    laser_lookat = Point2D(
                        (x + 0.5)
                        * self.scene_config["sensor_width"]
                        / self.laser_width,
                        (y + 0.5)
                        * self.scene_config["sensor_height"]
                        / self.laser_height,
                    )

                    laser_lookats.append(laser_lookat)
        else:
            raise AssertionError(
                "Invalid scan_type, must be one of {single|exhaustive|confocal}"
            )

        return laser_lookats

    def _render_steady(self, render_name, sensor_index):
        if not self.args.quiet:
            log(
                LogLevel.INFO,
                f"{render_name} for {self.experiment_name} steady render...",
            )

        hdr_ext = self.mitsuba_backend.get_hdr_extension()
        hdr_path = os.path.join(
            self._get_directory("partial_results_dir"),
            f"{self.experiment_name}_{render_name}.{hdr_ext}",
        )
        ldr_path = os.path.join(
            self._get_directory("steady_dir"),
            f"{self.experiment_name}_{render_name}.png",
        )

        if os.path.exists(ldr_path) and not self.args.quiet:
            pass  # skip
        else:
            logfile = None
            if self.args.do_logging and not self.args.dry_run:
                logfile = open(
                    os.path.join(
                        self._get_directory("log_dir"),
                        f"{self.experiment_name}_{render_name}.log",
                    ),
                    "w",
                )
            # NOTE: something here has a memory leak (probably Mitsuba-related)
            # We run Mitsuba in a separate process to ensure that the leaks do not add up
            # as they can fill your RAM in exhaustive scans
            queue = StdoutQueue()
            run_mitsuba_f = partial(
                self.mitsuba_backend.run_mitsuba,
                self.steady_scene_xml_path,
                hdr_path,
                dict(),
                render_name,
                logfile,
                self.args,
                sensor_index,
                queue,
            )
            self._run_mitsuba(hdr_path, ldr_path, logfile, queue, run_mitsuba_f)

    def _run_mitsuba(self, hdr_path, ldr_path, logfile, queue, run_mitsuba_f):
        if platform.system() == "Windows":
            # NOTE: Windows does not support multiprocessing
            run_mitsuba_f()
        else:
            # HACK: on macOS "spawn" method, which is the default since 3.8,
            # is considered more safe than "fork", but requires serialization methods available
            # to send the objects to the spawned process. So a proper fix would be to add them
            # (see e.g. https://stackoverflow.com/a/65513291 and
            # https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods
            # for more details)
            if platform.system() == "Darwin":
                multiprocessing.set_start_method("fork")

            process = multiprocessing.Process(target=run_mitsuba_f)
            try:
                process.start()
                process.join()
            except KeyboardInterrupt:
                process.terminate()
                raise KeyboardInterrupt
        if self.args.do_logging and not self.args.dry_run:
            while not queue.empty():
                e = queue.get()
                if isinstance(e, Exception):
                    raise e
                else:
                    logfile.write(e)
            queue.close()
            logfile.close()
        if not self.args.dry_run:
            self.mitsuba_backend.convert_hdr_to_ldr(hdr_path, ldr_path)

    def _run_single_simulation(self, pbar: Any, laser_lookat: Point2D, i: int):
        simulation_params = self._setup_single_simulation(
            self.experiment_name, i, laser_lookat
        )

        if simulation_params is None:
            return  # Skip

        run_mitsuba_f = simulation_params.make_mitsuba_call(self.mitsuba_backend)

        if platform.system() == "Windows":
            # Windows does not support multiprocessing
            run_mitsuba_f()
        else:
            process = multiprocessing.Process(target=run_mitsuba_f)
            try:
                process.start()
                process.join()
            except KeyboardInterrupt:
                process.terminate()
                raise KeyboardInterrupt
        self._teardown_single_simulation(simulation_params, i)

        if self.scan_type == "exhaustive" and i == 0:
            size_bytes = os.path.getsize(simulation_params.hdr_path)
            final_size_gb = size_bytes * len(self.laser_lookats) / 2**30
            pbar.set_description(
                f"Rendering {self.experiment_name} ({self.scan_type}, estimated size: {final_size_gb:.2f} GB)..."
            )

    def _setup_single_simulation(
        self, experiment_name, i, laser_lookat: Point2D
    ) -> Optional[SimulationParams]:
        try:
            hdr_path, is_dir = self.mitsuba_backend.partial_laser_path(
                self._get_directory("partial_results_dir"),
                experiment_name,
                laser_lookat.x,
                laser_lookat.y,
            )
            # ToDo: why do we have `self.args.quiet` here?
            if os.path.exists(hdr_path) and not self.args.quiet:
                return None  # Skip this render
            if is_dir:
                os.mkdir(hdr_path)
        except OSError as e:
            raise AssertionError(f"Invalid permissions: {e}") from e
        defines = {
            "laser_lookat_x": laser_lookat.x,
            "laser_lookat_y": laser_lookat.y,
        }
        logfile = None
        if self.args.do_logging and not self.args.dry_run:
            logfile = open(
                os.path.join(
                    self._get_directory("log_dir"),
                    f"{experiment_name}_L[{laser_lookat.x}][{laser_lookat.y}].log",
                ),
                "w",
            )

        render_name = f"Laser {i + 1} of {len(self.laser_lookats)}"

        params = SimulationParams(
            defines, self.nlos_scene_xml_path, hdr_path, logfile, render_name, self.args
        )
        return params

    def _teardown_single_simulation(self, params: SimulationParams, i: int):
        if self.args.do_logging and not self.args.dry_run:
            past_elems = []
            while not params.queue.empty():
                e = params.queue.get()
                past_elems.append(e)
                if isinstance(e, Exception):
                    log(LogLevel.ERROR, "")
                    log(LogLevel.ERROR, "/!\ Mitsuba thread got an exception!")
                    log(LogLevel.ERROR, "")
                    for pe in past_elems[:-1]:
                        log(LogLevel.ERROR, pe)
                    log(LogLevel.ERROR, "")
                    raise e
                else:
                    params.logfile.write(e)
                params.logfile.flush()
            params.queue.close()
            params.logfile.close()

    def _fill_capture_data(self, capture_data, num_retries=0):
        if self.scan_type == "single":
            hdr_path, _ = self.mitsuba_backend.partial_laser_path(
                self._get_directory("partial_results_dir"),
                self.experiment_name,
                *self.laser_lookats[0],
            )
            capture_data.H = self.mitsuba_backend.read_transient_image(hdr_path)
            capture_data.H_format = HFormat.T_Sx_Sy
        elif self.scan_type == "exhaustive" or self.scan_type == "confocal":
            if self.scan_type == "exhaustive":
                capture_data.H = np.empty(
                    (
                        self.num_bins,
                        self.laser_width,
                        self.laser_height,
                        self.scene_config["sensor_width"],
                        self.scene_config["sensor_height"],
                    ),
                    dtype=np.float32,
                )
                capture_data.H_format = HFormat.T_Lx_Ly_Sx_Sy
            elif self.scan_type == "confocal":
                capture_data.H = np.empty(
                    (self.num_bins, self.laser_width, self.laser_height),
                    dtype=np.float32,
                )
                capture_data.H_format = HFormat.T_Sx_Sy
            else:
                raise AssertionError

            e_laser_lookats = enumerate(self.laser_lookats)
            if not self.args.quiet and len(self.laser_lookats) > 1:
                e_laser_lookats = tqdm(
                    e_laser_lookats,
                    desc="Merging partial results...",
                    file=TQDMLogRedirect(),
                    ascii=True,
                    total=len(self.laser_lookats),
                )
            try:
                for i, (laser_lookat_x, laser_lookat_y) in e_laser_lookats:
                    hdr_path = self._fill_capture_frame(
                        capture_data, i, laser_lookat_x, laser_lookat_y
                    )
            except Exception as e:
                if num_retries >= 10:
                    raise AssertionError(
                        f"Failed to read partial results after {num_retries} retries"
                    )
                self.mitsuba_backend.remove_transient_image(hdr_path)
                # TODO Mitsuba sometimes fails to write some images,
                # it seems like some sort of race condition
                # If there is a partial result missing, just re-launch for now
                log(
                    LogLevel.INFO,
                    f"We missed some partial results (iteration {i} failed because: {e}), re-launching...",
                )
                return self._render_nlos_scene_impl(num_retries=num_retries + 1)
        else:
            raise AssertionError(
                "Invalid scan_type, must be one of {single|exhaustive|confocal}"
            )

    def _fill_capture_frame(self, capture_data, i, laser_lookat_x, laser_lookat_y):
        x = i % self.laser_width
        y = i // self.laser_width
        hdr_path, _ = self.mitsuba_backend.partial_laser_path(
            self._get_directory("partial_results_dir"),
            self.experiment_name,
            laser_lookat_x,
            laser_lookat_y,
        )
        if self.scan_type == "confocal":
            capture_data.H[:, x : x + 1, y : y + 1, ...] = (
                self.mitsuba_backend.read_transient_image(hdr_path)
            )
        elif self.scan_type == "exhaustive":
            capture_data.H[:, x, y, ...] = self.mitsuba_backend.read_transient_image(
                hdr_path
            )
        else:
            raise AssertionError
        return hdr_path

    def _cleanup(self):
        # remove IN_PROGRESS file
        os.remove(self.progress_file_path)

        if self.args.keep_partial_results:
            return

        if not self.args.quiet:
            log(
                LogLevel.INFO,
                f"Cleaning partial results in {self._get_directory('partial_results_dir')}...",
            )

        shutil.rmtree(self._get_directory("partial_results_dir"))

        if not self.args.quiet:
            log(LogLevel.INFO, "All clean.")

    def _get_directory(self, name: str) -> str:
        return self.directories[name]

    @property
    def experiment_name(self):
        return self.scene_config["name"]

    @property
    def scan_type(self):
        return self.scene_config["scan_type"]

    @property
    def num_bins(self):
        return self.scene_config["num_bins"]

    @property
    def sensor_width(self):
        return self.scene_config["sensor_width"]

    @property
    def sensor_height(self):
        return self.scene_config["sensor_height"]

    # We have setters for these next two properties since we set them to (1, 1) if `scan_type` is 'single'
    @property
    def laser_width(self):
        return self.scene_config["laser_width"]

    @laser_width.setter
    def laser_width(self, value):
        self.scene_config["laser_width"] = value

    @property
    def laser_height(self):
        return self.scene_config["laser_height"]

    @laser_height.setter
    def laser_height(self, value):
        self.scene_config["laser_height"] = value

    @staticmethod
    def _get_grid_xyz(nx, ny, rw_scale_x, rw_scale_y) -> np.array:
        px = rw_scale_x
        py = rw_scale_y
        xg = np.stack((np.linspace(-px, px, num=2 * nx + 1)[1::2],) * ny, axis=1)
        yg = np.stack((np.linspace(-py, py, num=2 * ny + 1)[1::2],) * nx, axis=0)
        assert (
            xg.shape[0] == yg.shape[0] == nx and xg.shape[1] == yg.shape[1] == ny
        ), "Incorrect shapes"
        return np.stack([xg, yg, np.zeros((nx, ny))], axis=-1).astype(np.float32)
