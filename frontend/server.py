#!/usr/bin/env python3
"""
Simple HTTP server to serve the HTML frontend
"""
import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

def start_frontend_server(port=8501):
    """Start a simple HTTP server for the HTML frontend"""
    
    # Change to frontend directory
    frontend_dir = Path(__file__).parent
    os.chdir(frontend_dir)
    
    Handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            print(f"🖥️ Frontend server starting...")
            print(f"📍 Frontend available at: http://localhost:{port}")
            print(f"📍 Open in browser: http://localhost:{port}/index.html")
            print("Press Ctrl+C to stop the server")
            
            # Try to open in browser
            try:
                webbrowser.open(f"http://localhost:{port}/index.html")
            except:
                pass
            
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Frontend server stopped")
    except Exception as e:
        print(f"❌ Error starting frontend server: {e}")

if __name__ == "__main__":
    start_frontend_server()
