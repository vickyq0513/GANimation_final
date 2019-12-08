from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self._parser.add_argument('--test_id',type=str,help='test id .jpg')
        self._parser.add_argument('--test_id_desired',type=str, default='190017',help='test id.jpg desired exp')
        self._parser.add_argument('--input_path', type=str, help='path to image')
        self._parser.add_argument('--output_dir', type=str, default='./output', help='output path')
        self.is_train = False
