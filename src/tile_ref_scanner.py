"""
Tile Reference Scanner — find, validate, and insert ref: comments in code.

Every line of code should reference a wiki page + line.
This scanner validates existing refs and finds gaps.

Usage:
    python tile_ref_scanner.py scan /path/to/repo
    python tile_ref_scanner.py gaps /path/to/repo
    python tile_ref_scanner.py graph /path/to/repo
"""

import os
import re
import json
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ref: wiki/page.md#L42 or ref: wiki/page.md or ref: docs/architecture.md#L15
REF_PATTERN = re.compile(r'ref:\s*(\S+?)(?:#L(\d+))?\s*(?:—|-)?\s*(.*?)$', re.MULTILINE)
WIKI_REF_PATTERN = re.compile(r'ref:\s*wiki/(\S+?)(?:#L(\d+))?')


class TileRefScanner:
    """Scan codebases for tile references and build navigation graphs."""
    
    def __init__(self, repo_path: str, wiki_path: str = "wiki"):
        self.repo_path = Path(repo_path)
        self.wiki_path = self.repo_path / wiki_path
        self.refs: List[Dict] = []
        self.gaps: List[Dict] = []
        self.graph: Dict[str, List[str]] = defaultdict(list)  # file → [wiki pages]
        self.wiki_pages: Dict[str, List[str]] = {}  # page → [lines]
    
    def scan(self) -> Dict:
        """Scan all code files for ref: comments."""
        self.refs = []
        code_extensions = {'.py', '.rs', '.c', '.h', '.ts', '.js', '.go', '.zig', '.toml', '.yaml'}
        
        for filepath in self._walk_files(code_extensions):
            with open(filepath) as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                # Check for ref: in comments
                if 'ref:' in line and ('//' in line or '#' in line or '/*' in line or '--' in line):
                    matches = REF_PATTERN.findall(line)
                    for match in matches:
                        ref_target, ref_line, ref_desc = match
                        ref = {
                            "source_file": str(filepath.relative_to(self.repo_path)),
                            "source_line": line_num,
                            "target": ref_target,
                            "target_line": int(ref_line) if ref_line else None,
                            "description": ref_desc.strip(),
                            "valid": self._validate_ref(ref_target, ref_line),
                        }
                        self.refs.append(ref)
                        self.graph[ref["source_file"]].append(ref_target)
        
        return {
            "total_refs": len(self.refs),
            "valid_refs": sum(1 for r in self.refs if r["valid"]),
            "invalid_refs": sum(1 for r in self.refs if not r["valid"]),
            "files_with_refs": len(set(r["source_file"] for r in self.refs)),
        }
    
    def find_gaps(self) -> List[Dict]:
        """Find functions/methods that lack ref: comments."""
        self.gaps = []
        code_extensions = {'.py', '.rs', '.c', '.h', '.ts', '.js', '.go'}
        
        for filepath in self._walk_files(code_extensions):
            with open(filepath) as f:
                content = f.read()
                lines = content.split('\n')
            
            # Find function definitions (Python, Rust, C, TypeScript)
            for line_num, line in enumerate(lines, 1):
                is_function = False
                func_name = ""
                
                # Python: def func_name(
                if re.match(r'\s*def\s+(\w+)', line):
                    is_function = True
                    func_name = re.match(r'\s*def\s+(\w+)', line).group(1)
                # Rust/C: fn func_name( or type func_name(
                elif re.match(r'\s*(pub\s+)?fn\s+(\w+)', line):
                    is_function = True
                    func_name = re.match(r'\s*(pub\s+)?fn\s+(\w+)', line).group(2)
                # C: type func_name(
                elif re.match(r'\s*\w+\s+(\w+)\s*\(', line) and not line.strip().startswith('//'):
                    match = re.match(r'\s*\w+\s+\*?(\w+)\s*\(', line)
                    if match and match.group(1) not in ('if', 'while', 'for', 'switch', 'return'):
                        is_function = True
                        func_name = match.group(1)
                
                if is_function:
                    # Check if any nearby line has a ref:
                    nearby = '\n'.join(lines[max(0, line_num-3):min(len(lines), line_num+2)])
                    if 'ref:' not in nearby:
                        self.gaps.append({
                            "file": str(filepath.relative_to(self.repo_path)),
                            "line": line_num,
                            "function": func_name,
                            "priority": self._estimate_complexity(func_name, lines, line_num),
                        })
        
        # Sort by priority (most complex first)
        self.gaps.sort(key=lambda g: g["priority"], reverse=True)
        return self.gaps
    
    def build_navigation_tiles(self, output_path: str = "nav_tiles.json") -> Dict:
        """Build navigation tiles from the reference graph."""
        tiles = []
        
        for ref in self.refs:
            tile = {
                "tile_type": "code_reference",
                "source": f"{ref['source_file']}#L{ref['source_line']}",
                "target": ref["target"],
                "target_line": ref.get("target_line"),
                "description": ref.get("description", ""),
                "valid": ref["valid"],
            }
            tiles.append(tile)
        
        # Add gap tiles (undocumented functions)
        for gap in self.gaps[:100]:  # top 100 undocumented
            tiles.append({
                "tile_type": "documentation_gap",
                "source": f"{gap['file']}#L{gap['line']}",
                "function": gap["function"],
                "priority": gap["priority"],
                "needs_wiki_entry": True,
            })
        
        output = Path(output_path)
        with open(output, 'w') as f:
            json.dump(tiles, f, indent=2)
        
        return {
            "navigation_tiles": len([t for t in tiles if t["tile_type"] == "code_reference"]),
            "gap_tiles": len([t for t in tiles if t["tile_type"] == "documentation_gap"]),
            "output": str(output),
        }
    
    def _walk_files(self, extensions: set) -> List[Path]:
        """Walk repo and return files with given extensions."""
        files = []
        skip_dirs = {'.git', 'node_modules', '__pycache__', 'target', 'build', '.cargo', 'dist'}
        for root, dirs, filenames in os.walk(self.repo_path):
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            for fn in filenames:
                if Path(fn).suffix in extensions:
                    files.append(Path(root) / fn)
        return files
    
    def _validate_ref(self, target: str, line: Optional[str]) -> bool:
        """Check if a reference target exists."""
        # Check wiki path
        full_path = self.repo_path / target
        if full_path.exists():
            return True
        # Check with wiki/ prefix
        wiki_path = self.repo_path / "wiki" / target
        if wiki_path.exists():
            return True
        # Check docs/ prefix
        docs_path = self.repo_path / "docs" / target
        if docs_path.exists():
            return True
        return False
    
    def _estimate_complexity(self, func_name: str, lines: List[str], def_line: int) -> int:
        """Estimate function complexity for prioritization."""
        # Simple heuristic: count lines until next def/fn
        func_lines = 0
        for i in range(def_line, min(def_line + 50, len(lines))):
            if i > def_line and (lines[i].strip().startswith('def ') or 
                                  lines[i].strip().startswith('fn ') or
                                  lines[i].strip().startswith('pub fn ')):
                break
            func_lines += 1
        
        # Longer functions = higher priority for documentation
        # Common names (main, test, init) = lower priority
        score = func_lines
        if func_name.startswith('test_'):
            score -= 5
        if func_name in ('main', 'init', '__init__'):
            score -= 3
        if len(func_name) > 15:  # Descriptive names slightly lower priority
            score -= 1
        
        return max(score, 1)
    
    def stats(self) -> Dict:
        return {
            "refs_found": len(self.refs),
            "valid": sum(1 for r in self.refs if r["valid"]),
            "invalid": sum(1 for r in self.refs if not r["valid"]),
            "gaps_found": len(self.gaps),
            "files_with_refs": len(set(r["source_file"] for r in self.refs)),
            "wiki_pages_referenced": len(set(r["target"] for r in self.refs)),
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python tile_ref_scanner.py [scan|gaps|graph|tiles] /path/to/repo")
        sys.exit(1)
    
    command = sys.argv[1]
    repo_path = sys.argv[2]
    
    scanner = TileRefScanner(repo_path)
    
    if command == "scan":
        result = scanner.scan()
        print(f"References: {result['total_refs']} ({result['valid_refs']} valid, {result['invalid_refs']} invalid)")
        print(f"Files with refs: {result['files_with_refs']}")
        for ref in scanner.refs[:10]:
            status = "✅" if ref["valid"] else "❌"
            print(f"  {status} {ref['source_file']}#L{ref['source_line']} → {ref['target']}")
    
    elif command == "gaps":
        scanner.scan()
        gaps = scanner.find_gaps()
        print(f"Undocumented functions: {len(gaps)}")
        for gap in gaps[:20]:
            print(f"  {gap['file']}#L{gap['line']} — {gap['function']} (complexity: {gap['priority']})")
    
    elif command == "graph":
        scanner.scan()
        print(f"Reference graph ({len(scanner.graph)} files):")
        for file, targets in sorted(scanner.graph.items()):
            print(f"  {file} → {', '.join(set(targets))}")
    
    elif command == "tiles":
        scanner.scan()
        scanner.find_gaps()
        result = scanner.build_navigation_tiles(f"{repo_path}/nav_tiles.json")
        print(f"Navigation tiles: {result['navigation_tiles']} refs + {result['gap_tiles']} gaps")
        print(f"Output: {result['output']}")
    
    else:
        print(f"Unknown command: {command}")
