import unittest
import os
from code import video_to_skeleton

class TestVideoToSkeleton(unittest.TestCase):
    def test_video_to_skeleton(self):
        # This test will only check if the function runs without error on a short video
        video_path = 'data/input/inoue.mp4'
        out_json = 'tests/tmp_skeleton.json'
        out_vid = 'tests/tmp_skeleton.mp4'
        try:
            video_to_skeleton.video_to_skeleton(video_path, out_json, out_vid)
            self.assertTrue(os.path.exists(out_json))
            self.assertTrue(os.path.exists(out_vid))
        finally:
            if os.path.exists(out_json):
                os.remove(out_json)
            if os.path.exists(out_vid):
                os.remove(out_vid)

if __name__ == '__main__':
    unittest.main()
