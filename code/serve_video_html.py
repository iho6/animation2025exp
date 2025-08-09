import sys
import os
import http.server
import socketserver
import webbrowser

PORT = 8080

def serve_and_open(filepath):
    html_path = filepath + '_preview.html'
    if not os.path.isfile(html_path):
        print(f"HTML preview not found: {html_path}\nRun play_video.py first.")
        return
    dir_path = os.path.dirname(os.path.abspath(html_path))
    os.chdir(dir_path)
    url = f"http://localhost:{PORT}/{os.path.basename(html_path)}"
    print(f"Serving {dir_path} at {url}")
    try:
        webbrowser.open(url)
    except Exception:
        print(f"Please open {url} in your browser.")
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Press Ctrl+C to stop serving.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python serve_video_html.py <video_file>")
    else:
        serve_and_open(sys.argv[1])
