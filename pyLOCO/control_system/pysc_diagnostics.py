"""Opt-in, read-only diagnostics of the actual demo SC, never a machine driver."""
import hashlib
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer


def start_diagnostics(sc, profile, control_port, port):
    identity = dict(instance=str(uuid.uuid4()), profile=profile.key,
                    machine=profile.label, scenario=profile.scenario,
                    control_port=control_port, seed=profile.configuration.get("random_seed"),
                    lattice_sha256=hashlib.sha256(profile.resolve("lattice_file").read_bytes()).hexdigest())

    def snapshot():
        rows = []
        for name, control in sc.magnet_settings.controls.items():
            links = control._links
            if not name.endswith('/B2') or len(links) != 1:
                continue
            link = links[0]
            magnet = sc.magnet_settings.magnets[link.magnet_name]
            if link.component != 'B' or link.order != 2 or link.is_integrated or magnet.imperfections is not None:
                continue
            element = sc.lattice.ring[magnet.sim_index]
            rows.append(dict(control=name, component='B2', unit='m^-2',
                             ordinal=magnet.sim_index, common_name=str(getattr(element, 'CommonName', '')),
                             family=str(element.FamName), factor=float(link.error.factor),
                             offset=float(link.error.offset), current=float(control.setpoint),
                             physical=float(element.PolynomB[1])))
        return dict(identity=identity, quadrupoles=rows)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != '/snapshot':
                self.send_error(404); return
            data = json.dumps(snapshot(), allow_nan=False).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
