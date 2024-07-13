import os
from typing import Optional
import ppcgrader.config


class Config(ppcgrader.config.Config):
    def __init__(self, openmp: bool = False, gpu: bool = False):
        super().__init__(binary='is',
                         cfg_file=__file__,
                         gpu=gpu,
                         openmp=openmp)
        self.default_demo = [os.path.join(self.base_dir, 'demo.png')]
        self.demo_flags = self._demo_flags_png
        self.demo_post = self._demo_post_png

    def parse_output(self, output):
        time = None
        errors = None
        input_data = {"nx": None, "ny": None}
        output_data = {
            "y0": None,
            "y1": None,
            "x0": None,
            "x1": None,
            "outer": None,
            "inner": None
        }
        output_errors = {
            "expected": {
                "y0": None,
                "y1": None,
                "x0": None,
                "x1": None,
                "outer": None,
                "inner": None
            }
        }
        statistics = {}
        size = None
        target = None
        triples = []
        for line in output.splitlines():
            what, arg = line.split('\t')
            if what == 'result':
                errors = {'fail': True, 'pass': False, 'done': False}[arg]
            elif what == 'time':
                time = float(arg)
            elif what == 'perf_wall_clock_ns':
                time = int(arg) / 1e9
                statistics[what] = int(arg)
            elif what.startswith('perf_'):
                statistics[what] = int(arg)
            elif what in ['ny', 'nx']:
                input_data[what] = int(arg)
            elif what in ['error_magnitude', 'threshold']:
                output_errors[what] = float(arg)
            elif what == 'what':
                target = {
                    'expected': output_errors['expected'],
                    'got': output_data,
                }[arg]
            elif what in ['y0', 'y1', 'x0', 'x1']:
                target[what] = int(arg)
            elif what in ['inner', 'outer']:
                parsed = [float(c) for c in arg.split(',')]
                target[what] = parsed
            elif what == 'size':
                size = arg
            elif what == 'triple':
                parsed = [float(c) for c in arg.split(',')]
                triples.append(parsed)
        if size == "small":
            nx = input_data["nx"]
            ny = input_data["ny"]
            assert len(triples) == nx * ny
            input_data["data"] = [
                triples[i * nx:(i + 1) * nx] for i in range(ny)
            ]
        if errors:
            output_errors["wrong_output"] = True

        return time, errors, input_data, output_data, output_errors, statistics

    def explain_terminal(self, output, color=False) -> Optional[str]:
        from .info import explain_terminal
        return explain_terminal(output, color)
