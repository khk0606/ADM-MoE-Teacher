"""Dependency-free checks for the public repository layout."""
import ast
import hashlib
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / 'docs/provenance/small_room30'

def entries(path):
    return {line.split(None, 1)[1]: line.split(None, 1)[0]
            for line in path.read_text().splitlines() if line.strip()}

class LayoutTests(unittest.TestCase):
    def test_root_contains_only_project_metadata(self):
        files = {p.name for p in ROOT.iterdir() if p.is_file() and not p.name.startswith('.')}
        self.assertEqual(files, {'README.md', 'LICENSE', 'requirements.txt'})

    def test_current_manifest_and_original_model_sources(self):
        old = entries(PROVENANCE / 'original-package.sha256')
        new = entries(PROVENANCE / 'layout.sha256')
        self.assertEqual(len(new), 28)
        for name, digest in new.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest, name)
        for name, digest in old.items():
            if name.startswith('prepare/'):
                self.assertEqual(new[name], digest, name)

    def test_adm_entry_points(self):
        for name in ('train.py', 'train_ddp.py', 'test.py'):
            text = (ROOT / 'scripts/adm' / name).read_text()
            ast.parse(text)
            self.assertIn('config_path="../../configs"', text)
            self.assertIn('parents[2]', text)
        for p in (ROOT / 'scripts').rglob('*.sh'):
            self.assertIsNone(re.search(r'python (?:train|test)\\.py', p.read_text()), str(p))

    def test_current_markdown_links(self):
        paths = [ROOT / 'README.md'] + list((ROOT / 'docs/guides').glob('*.md')) + list((ROOT / 'docs/results/small_room30_anywhere').glob('*.md'))
        for p in paths:
            for link in re.findall(r'\\]\\(([^)]+)\\)', p.read_text()):
                if '://' not in link and not link.startswith('#'):
                    self.assertTrue((p.parent / link.split('#')[0]).exists(), (p, link))

if __name__ == '__main__':
    unittest.main()
