
import subprocess
import os
import pytest

pytest.importorskip('human_body_prior', reason="human_body_prior not installed")
pytest.importorskip('body_visualizer', reason="body_visualizer not installed")

def test_amass_setup_runs(tmp_path):
    # Use a small sample file and output dir
    data_file = os.path.abspath('third_party/amass/support_data/github_data/amass_sample.npz')
    model_dir = os.path.abspath('third_party/amass/body_models')
    output_dir = tmp_path / "amass_output"
    output_dir.mkdir()
    result = subprocess.run([
        'python', 'code/amass_setup.py',
        '--data', data_file,
        '--body_models_dir', model_dir,
        '--output_dir', str(output_dir)
    ], capture_output=True, text=True)
    assert result.returncode == 0 or 'Saved:' in result.stdout
    # Check that at least one output image was created
    assert any(f.suffix == '.png' for f in output_dir.iterdir())
