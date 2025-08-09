
import unittest
from unittest import mock
from code import skeleton_visualizer

class TestSkeletonVisualizer(unittest.TestCase):
    @mock.patch("builtins.open", new_callable=mock.mock_open, read_data='{"frames": [{"pose": [{"x": 0.5, "y": 0.5, "z": 0.0}]}]}')
    @mock.patch("cv2.VideoWriter")
    def test_main(self, mock_videowriter, mock_open):
        # Use a dummy json path and output paths
        json_path = 'data/output/skeleton.json'
        out_2d = 'tests/tmp_2d.mp4'
        out_3d = 'tests/tmp_3d.mp4'
        # Mock VideoWriter instance
        mock_writer = mock.Mock()
        mock_videowriter.return_value = mock_writer
        # Run main
        skeleton_visualizer.main(json_path, out_2d, out_3d)
        # Check that open was called to read JSON
        mock_open.assert_called_with(json_path)
        # Check that VideoWriter was created for both outputs
        self.assertTrue(mock_videowriter.called)
        self.assertGreaterEqual(mock_videowriter.call_count, 2)
        # Check that write was called on the VideoWriter
        self.assertTrue(mock_writer.write.called)


if __name__ == '__main__':
    unittest.main()
