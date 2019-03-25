from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--outline_style', type=int, default=0, help='which edge style')
        self.parser.add_argument('--shading_style', type=int, default=0, help='which shading style')
        self.parser.add_argument('--Sigma', type=float, default=2.5, help='sigma for XDoG')
        self.parser.add_argument('--pad', type=int, default=10)
        self.parser.add_argument('--r', type=int, default=11)
        self.parser.add_argument('--eps', type=float, default=0.1)
        self.isTrain = False
