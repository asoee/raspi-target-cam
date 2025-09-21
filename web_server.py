#!/usr/bin/env python3
"""
Web Server for Camera Control Interface
Serves HTML interface on port 8081
"""

import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn


class CameraWebHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)

    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.path = '/camera_interface.html'
        super().do_GET()

    def log_message(self, format, *args):
        # Custom logging or suppress
        print(f"Web Server: {format % args}")


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    server = ThreadingHTTPServer(('0.0.0.0', 8081), CameraWebHandler)
    print("Web interface server starting on port 8081...")
    print("Access the interface at: http://localhost:8081")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down web server...")
        server.shutdown()


if __name__ == '__main__':
    main()