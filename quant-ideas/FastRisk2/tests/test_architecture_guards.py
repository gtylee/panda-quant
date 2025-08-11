import os
import re
import unittest


class TestArchitectureGuards(unittest.TestCase):
    def test_no_imports_from_archive(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        disallowed_pattern = re.compile(r"^(from|import)\s+Archive_20[0-9]{6}")

        offenders = []
        for entry in os.listdir(repo_root):
            if not entry.endswith('.py'):
                continue
            file_path = os.path.join(repo_root, entry)
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, start=1):
                    if disallowed_pattern.search(line.strip()):
                        offenders.append((entry, i, line.strip()))

        if offenders:
            details = "\n".join(f"{fname}:{lineno}: {text}" for fname, lineno, text in offenders)
            self.fail(f"Disallowed imports from Archive_ directories detected:\n{details}")


if __name__ == "__main__":
    unittest.main()


