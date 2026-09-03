"""Validate four synthetic B2 changes and injected post-SET failure on local pySC."""
import argparse
import json
from pathlib import Path
from pyLOCO.correct.quadrupole_transaction import SimulationConnection
from pyLOCO.correct.multi_quadrupole_transaction import MultiQuadrupoleTransaction


class FailThirdSet(SimulationConnection):
    """Test-only fault injection; the real third SET succeeds before raising."""
    def __init__(self, port): super().__init__(port); self.sets=0
    def interface(self, identity):
        interface=super().interface(identity); parent=self
        class Proxy:
            def get(self,name): return interface.get(name)
            def set(self,name,value):
                interface.set(name,value); parent.sets+=1
                if parent.sets==3: raise RuntimeError('Deliberate test failure after third real SET')
        return Proxy()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--diagnostics-port',type=int,default=13332)
    parser.add_argument('--output',type=Path,default=Path('/private/tmp/pyloco-multi-b2-validation'))
    args=parser.parse_args(); request=json.loads(Path(__file__).with_name('petra_four_b2.json').read_text())
    for failure in (False,True):
        connection=(FailThirdSet if failure else SimulationConnection)(args.diagnostics_port)
        t=MultiQuadrupoleTransaction(connection,args.output); t.preview(request)
        try:
            try: t.apply(confirmed=True)
            except RuntimeError:
                if not failure or t.record['status']!='restored': raise
            else:
                if failure: raise AssertionError('Injection did not fire')
        finally:
            if t.record['status'] in ('applied','write_pending','restore_failed'):t.restore()
        assert t.record['summary']['all_restored']
        print('FAILURE CASE' if failure else 'SUCCESS CASE', json.dumps(t.record,indent=2)); print('JOURNAL',t.path)


if __name__=='__main__':main()
