
import unittest
from unittest import mock
from code import video_to_skeleton

class TestVideoToSkeleton(unittest.TestCase):
    @mock.patch("builtins.open", new_callable=mock.mock_open)
    @mock.patch("cv2.VideoWriter")
    @mock.patch("cv2.VideoCapture")
    @mock.patch("pathlib.Path.mkdir")
    def test_video_to_skeleton(self, mock_mkdir, mock_videocap, mock_videowriter, mock_open):
        # Mock video capture to simulate a short video
        mock_cap = mock.Mock()
        mock_cap.isOpened.return_value = True
        # Return reasonable defaults for any property
        mock_cap.get.side_effect = lambda x: {4: 30, 3: 640, 2: 480, 7: 2}.get(x, 0)
        import numpy as np
        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 127
        mock_cap.read.side_effect = [
            (True, dummy_frame),
            (True, dummy_frame),
            (False, None)
        ]
        mock_videocap.return_value = mock_cap

        # Patch mediapipe pose
        with mock.patch("code.video_to_skeleton.mp_pose.Pose") as mock_pose_class:
            mock_pose = mock.Mock()
            mock_pose.process.return_value = mock.Mock(pose_landmarks=None)
            mock_pose_class.return_value.__enter__.return_value = mock_pose

            video_to_skeleton.video_to_skeleton("fake.mp4", "fake.json", "fake.mp4")

        # Check that open was called to write JSON
        mock_open.assert_called()
        # Check that VideoWriter was created
        mock_videowriter.assert_called()
