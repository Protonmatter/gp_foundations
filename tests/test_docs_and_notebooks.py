import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / 'docs'
NOTEBOOKS = ROOT / 'notebooks'
REFERENCES = ROOT / 'references' / 'pdfs'


class DocumentationTests(unittest.TestCase):
    def test_index_links_exist(self) -> None:
        index_text = (DOCS / 'index.md').read_text()
        links = re.findall(r'\(([^)]+)\)', index_text)
        for link in links:
            if link.startswith('http'):
                continue
            path = (DOCS / link).resolve() if not link.startswith('../') else (DOCS / link).resolve()
            self.assertTrue(path.exists(), msg=f'missing linked file: {link}')

    def test_pdf_corpus_is_present(self) -> None:
        pdfs = sorted(REFERENCES.glob('*.pdf'))
        self.assertEqual(len(pdfs), 10)

    def test_notebooks_are_valid_json(self) -> None:
        for notebook_path in NOTEBOOKS.glob('*.ipynb'):
            with self.subTest(notebook=notebook_path.name):
                notebook = json.loads(notebook_path.read_text())
                self.assertIn('cells', notebook)
                self.assertGreaterEqual(len(notebook['cells']), 2)
                self.assertEqual(notebook['nbformat'], 4)


if __name__ == '__main__':
    unittest.main()
