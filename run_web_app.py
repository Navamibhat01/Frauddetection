import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000
DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def run():
    # Change current working directory to script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Check if web directory exists
    if not os.path.exists(DIRECTORY):
        print(f"Error: Directory '{DIRECTORY}' not found.")
        sys.exit(1)
        
    handler = Handler
    
    # Enable socket re-use to avoid port bind conflicts
    socketserver.TCPServer.allow_reuse_address = True
    
    try:
        with socketserver.TCPServer(("", PORT), handler) as httpd:
            url = f"http://localhost:{PORT}/index.html"
            print(f"Serving AeroPay GNN Shield app at: {url}")
            print("Press Ctrl+C to stop the server.")
            
            # Automatically open the browser
            webbrowser.open(url)
            
            # Keep serving
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    run()
