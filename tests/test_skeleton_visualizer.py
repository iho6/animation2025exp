import unittest
import os
from code import skeleton_visualizer

class TestSkeletonVisualizer(unittest.TestCase):
    def test_main(self):
        # Use a small sample skeleton.json for testing
        json_path = 'data/output/skeleton.json'
        out_2d = 'tests/tmp_2d.mp4'
        out_3d = 'tests/tmp_3d.mp4'
        try:
            skeleton_visualizer.main(json_path, out_2d, out_3d)
            self.assertTrue(os.path.exists(out_2d))
            self.assertTrue(os.path.exists(out_3d))
        finally:
            if os.path.exists(out_2d):
                os.remove(out_2d)
            if os.path.exists(out_3d):
                os.remove(out_3d)

if __name__ == '__main__':
    unittest.main()
