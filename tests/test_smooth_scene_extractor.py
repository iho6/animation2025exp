import unittest
import os
import tempfile
from code import smooth_scene_extractor

class TestSmoothSceneExtractor(unittest.TestCase):
    def test_detect_scenes(self):
        # Use a very short video or mock
        video_path = 'data/input/inoue.mp4'
        scenes = smooth_scene_extractor.detect_scenes(video_path)
        self.assertIsInstance(scenes, list)
        self.assertTrue(len(scenes) > 0)

    def test_extract_features_from_frame(self):
        import cv2
        import torch
        import torchvision.transforms as T
        import torchvision.models as models
        frame = cv2.imread('data/input/inoue.mp4')
        device = torch.device('cpu')
        resnet18 = models.resnet18(weights=None)
        resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1])
        resnet18.eval().to(device)
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        # Use a dummy image if needed
        import numpy as np
        dummy = np.ones((224,224,3), dtype=np.uint8)*127
        features = smooth_scene_extractor.extract_features_from_frame(dummy, resnet18, transform, device)
        self.assertEqual(features.shape[0], 512)

if __name__ == '__main__':
    unittest.main()
