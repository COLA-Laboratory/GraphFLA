"""Serve independent preview sites; --variants efg serves the three new skins."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

ROOT = Path(__file__).resolve().parent / '.build'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variants', default='g', help='Candidate letters from abcdefg; G is selected')
args = parser.parse_args()
if not args.variants or any(c not in 'abcdefg' for c in args.variants) or len(set(args.variants)) != len(args.variants):
    parser.error('Provide unique candidate letters from abcdefg.')
servers = []
for candidate in args.variants:
    i = 'abcdefg'.index(candidate)
    handler = partial(SimpleHTTPRequestHandler, directory=str(ROOT / candidate))
    server = ThreadingHTTPServer(('127.0.0.1', 8811+i), handler)
    Thread(target=server.serve_forever, daemon=True).start()
    servers.append(server)
    print(f'{candidate.upper()}: http://127.0.0.1:{8811+i}/', flush=True)
try:
    import signal
    signal.pause()
except KeyboardInterrupt:
    for server in servers:
        server.shutdown()
