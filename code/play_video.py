import sys
import os
import subprocess


def play_video(filepath):
    if not os.path.isfile(filepath):
        print(f"File not found: {filepath}")
        return

    # Always generate and open HTML preview
    html_path = filepath + '_preview.html'
    rel_path = os.path.relpath(filepath, os.path.dirname(html_path))
    with open(html_path, 'w') as f:
        f.write(f'''<!DOCTYPE html>\n<html><body style="background:#222;color:#eee;font-family:sans-serif;">\n<h2>Video Preview: {os.path.basename(filepath)}</h2>\n<video width="800" controls autoplay loop>\n  <source src="{rel_path}" type="video/mp4">\n  Your browser does not support the video tag.\n</video>\n</body></html>''')
    print(f"Opening HTML preview: {html_path}")
    # Try to open in VS Code Simple Browser if available
    try:
        import webbrowser
        webbrowser.open(f'file://{os.path.abspath(html_path)}')
    except Exception:
        print(f"Please open {html_path} in your browser or VS Code Simple Browser.")


def main():
    if len(sys.argv) < 2:
        print("Usage: python play_video.py <video_file> [--open-html]")
        return
    filepath = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] == '--open-html':
        html_path = filepath + '_preview.html'
        if not os.path.isfile(html_path):
            print(f"HTML preview not found: {html_path} (run without --open-html first)")
            return
        import webbrowser
        print(f"Opening {html_path} in browser...")
        webbrowser.open(f'file://{os.path.abspath(html_path)}')
    else:
        play_video(filepath)


if __name__ == "__main__":
    main()
