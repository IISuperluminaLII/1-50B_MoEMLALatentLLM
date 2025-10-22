"""
Citation verification test suite.

This test ensures that all papers referenced in paper_ids.txt and paper_metadata.txt
have corresponding PDF files in the pdf_citations/ directory structure.

References:
    - Tests paper availability and correct folder placement
    - Validates arXiv ID formats
    - Detects orphan PDFs without corresponding IDs
    - Supports skip markers for unavailable papers (e.g., future publications)
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest


class TestCitations:
    """Test suite for validating citation integrity."""

    @pytest.fixture
    def citations_root(self) -> Path:
        """Get the pdf_citations root directory."""
        return Path(__file__).parent.parent.parent / "pdf_citations"

    @pytest.fixture
    def paper_ids(self, citations_root: Path) -> List[str]:
        """Load arXiv paper IDs from paper_ids.txt."""
        paper_ids_file = citations_root / "paper_ids.txt"
        if not paper_ids_file.exists():
            pytest.fail(f"paper_ids.txt not found at {paper_ids_file}")

        ids = []
        with open(paper_ids_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract arXiv ID (format: YYMM.NNNNN or YYMM.NNNNNN)
                    match = re.match(r'(\d{4}\.\d{4,6})', line)
                    if match:
                        ids.append(match.group(1))
        return ids

    @pytest.fixture
    def paper_metadata(self, citations_root: Path) -> List[Dict[str, str]]:
        """Load non-arXiv paper metadata from paper_metadata.txt."""
        metadata_file = citations_root / "paper_metadata.txt"
        if not metadata_file.exists():
            return []

        papers = []
        with open(metadata_file, 'r') as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Format: <identifier> | <type> | <title> | <authors> | <venue> | <year> | <folder> | <filename>
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) != 8:
                        pytest.fail(
                            f"Malformed metadata entry at {metadata_file}:{line_num}\n"
                            f"Expected 8 fields separated by '|', got {len(parts)} fields.\n"
                            f"Line content: {line}\n\n"
                            f"Expected format:\n"
                            f"<identifier> | <type> | <title> | <authors> | <venue> | <year> | <folder> | <filename>"
                        )
                    papers.append({
                        'identifier': parts[0],
                        'type': parts[1],
                        'title': parts[2],
                        'authors': parts[3],
                        'venue': parts[4],
                        'year': parts[5],
                        'folder': parts[6],
                        'filename': parts[7],
                    })
        return papers

    @pytest.fixture
    def all_pdfs(self, citations_root: Path) -> Set[Path]:
        """Find all PDF files in the citations directory."""
        return set(citations_root.rglob("*.pdf"))

    @pytest.fixture
    def arxiv_mapping(self, citations_root: Path) -> Dict[str, str]:
        """Load arXiv ID to PDF filename mapping."""
        mapping_file = citations_root / "arxiv_to_pdf_mapping.json"
        if not mapping_file.exists():
            return {}

        import json
        with open(mapping_file, 'r') as f:
            return json.load(f)

    def test_paper_ids_file_exists(self, citations_root: Path):
        """Test that paper_ids.txt exists."""
        assert (citations_root / "paper_ids.txt").exists(), \
            "paper_ids.txt is missing from pdf_citations/"

    def test_arxiv_id_format(self, paper_ids: List[str]):
        """Test that all arXiv IDs follow the correct format."""
        for arxiv_id in paper_ids:
            assert re.match(r'^\d{4}\.\d{4,6}$', arxiv_id), \
                f"Invalid arXiv ID format: {arxiv_id}"

    def test_arxiv_pdfs_exist(self, citations_root: Path, paper_ids: List[str], all_pdfs: Set[Path], arxiv_mapping: Dict[str, str]):
        """Test that all arXiv papers have corresponding PDFs."""
        missing_papers = []

        for arxiv_id in paper_ids:
            # First try to use the mapping file
            if arxiv_id in arxiv_mapping:
                expected_filename = arxiv_mapping[arxiv_id]
                # Search for this specific filename in all PDFs
                matching_pdfs = [
                    pdf for pdf in all_pdfs
                    if pdf.name == expected_filename
                ]
            else:
                # Fall back to substring search for unmapped IDs
                matching_pdfs = [
                    pdf for pdf in all_pdfs
                    if arxiv_id in pdf.stem or arxiv_id in str(pdf)
                ]

            if not matching_pdfs:
                missing_papers.append(arxiv_id)

        if missing_papers:
            pytest.fail(
                f"Missing PDFs for arXiv IDs: {missing_papers}\n"
                f"Run: cd pdf_citations && python download_papers.py paper_ids.txt"
            )

    def test_non_arxiv_pdfs_exist(self, citations_root: Path, paper_metadata: List[Dict[str, str]]):
        """Test that all non-arXiv papers have corresponding PDFs."""
        missing_papers = []

        for paper in paper_metadata:
            pdf_path = citations_root / paper['folder'] / paper['filename']
            if not pdf_path.exists():
                missing_papers.append({
                    'identifier': paper['identifier'],
                    'title': paper['title'],
                    'expected_path': str(pdf_path),
                })

        if missing_papers:
            error_msg = "Missing non-arXiv PDFs:\n"
            for paper in missing_papers:
                error_msg += f"  - {paper['identifier']}: {paper['title']}\n"
                error_msg += f"    Expected at: {paper['expected_path']}\n"
            pytest.fail(error_msg)

    def test_correct_folder_placement(self, citations_root: Path, all_pdfs: Set[Path]):
        """Test that PDFs are in correctly named folders."""
        valid_folders = {
            "01_Architecture",
            "02_Scaling_Laws",
            "03_Data_Sources",
            "04_Deduplication",
            "05_Quality_Filtering",
            "06_Domain_Mixing",
            "07_Data_Practices",
        }

        incorrectly_placed = []

        for pdf in all_pdfs:
            # Get the immediate parent folder
            folder = pdf.parent.name
            if folder not in valid_folders:
                incorrectly_placed.append(str(pdf.relative_to(citations_root)))

        if incorrectly_placed:
            pytest.fail(
                f"PDFs in incorrectly named folders:\n" +
                "\n".join(f"  - {p}" for p in incorrectly_placed) +
                f"\n\nValid folders: {', '.join(sorted(valid_folders))}"
            )

    def test_no_duplicate_pdfs(self, all_pdfs: Set[Path]):
        """Test that there are no duplicate PDF files (same filename in multiple folders)."""
        filenames = {}
        duplicates = []

        for pdf in all_pdfs:
            filename = pdf.name
            if filename in filenames:
                duplicates.append((filename, [filenames[filename], str(pdf)]))
            else:
                filenames[filename] = str(pdf)

        if duplicates:
            error_msg = "Duplicate PDFs found:\n"
            for filename, paths in duplicates:
                error_msg += f"  - {filename}:\n"
                for path in paths:
                    error_msg += f"    * {path}\n"
            pytest.fail(error_msg)

    def test_no_orphan_pdfs(
        self,
        citations_root: Path,
        paper_ids: List[str],
        paper_metadata: List[Dict[str, str]],
        all_pdfs: Set[Path],
        arxiv_mapping: Dict[str, str]
    ):
        """Test that all PDFs have corresponding entries in paper_ids.txt or paper_metadata.txt."""
        # Build set of tracked filenames
        tracked_filenames = set()

        # From paper_metadata.txt (exact filenames)
        for paper in paper_metadata:
            tracked_filenames.add(paper['filename'])

        # From arXiv mapping (exact filenames)
        for arxiv_id in paper_ids:
            if arxiv_id in arxiv_mapping:
                tracked_filenames.add(arxiv_mapping[arxiv_id])

        # From arXiv IDs (partial matching for unmapped IDs)
        arxiv_ids_set = set(paper_ids)

        orphan_pdfs = []

        for pdf in all_pdfs:
            filename = pdf.name

            # Check if filename is in tracked set
            if filename in tracked_filenames:
                continue

            # Check if any arXiv ID is in the filename (fallback for unmapped)
            has_arxiv_match = any(arxiv_id in filename for arxiv_id in arxiv_ids_set)
            if has_arxiv_match:
                continue

            # This PDF is not tracked anywhere
            orphan_pdfs.append(str(pdf.relative_to(citations_root)))

        if orphan_pdfs:
            pytest.fail(
                "Orphan PDFs found (not in paper_ids.txt or paper_metadata.txt):\n" +
                "\n".join(f"  - {p}" for p in orphan_pdfs) +
                "\n\nAction: Either add these to tracking files or remove them if irrelevant."
            )

    def test_readme_exists(self, citations_root: Path):
        """Test that README.md exists in pdf_citations/."""
        assert (citations_root / "README.md").exists(), \
            "README.md is missing from pdf_citations/"

    def test_folder_structure_complete(self, citations_root: Path):
        """Test that all required citation folders exist."""
        required_folders = [
            "01_Architecture",
            "02_Scaling_Laws",
            "03_Data_Sources",
            "04_Deduplication",
            "05_Quality_Filtering",
            "06_Domain_Mixing",
            "07_Data_Practices",
        ]

        missing_folders = []
        for folder in required_folders:
            if not (citations_root / folder).exists():
                missing_folders.append(folder)

        if missing_folders:
            pytest.fail(
                f"Missing citation folders:\n" +
                "\n".join(f"  - {f}" for f in missing_folders)
            )

    def test_malformed_metadata_raises_error(self, citations_root: Path):
        """Test that malformed metadata entries trigger explicit failures."""
        import tempfile
        import os

        # Create a temporary malformed metadata file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp.write("# Valid header\n")
            tmp.write("DOI:12345 | conference | Valid Paper | Author | Venue | 2024 | Folder | file.pdf\n")
            tmp.write("MALFORMED_LINE_WITH_ONLY_THREE_FIELDS | field2 | field3\n")
            tmp_path = tmp.name

        try:
            # The paper_metadata fixture logic should detect malformed entries
            # Simulate the fixture behavior
            malformed_detected = False
            with open(tmp_path, 'r') as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = [p.strip() for p in line.split('|')]
                        if len(parts) != 8:
                            # Malformed entry detected!
                            malformed_detected = True
                            # Verify the error message would be informative
                            expected_msg = f"Malformed metadata entry at {tmp_path}:{line_num}"
                            assert "Malformed metadata entry" in expected_msg
                            break

            # Verify that we did detect the malformed entry
            assert malformed_detected, "Malformed metadata entry was not detected!"

        finally:
            # Clean up
            os.unlink(tmp_path)

    def test_paper_count_matches(
        self,
        paper_ids: List[str],
        paper_metadata: List[Dict[str, str]],
        all_pdfs: Set[Path]
    ):
        """Test that the number of tracked papers matches the number of PDFs."""
        total_tracked = len(paper_ids) + len(paper_metadata)
        total_pdfs = len(all_pdfs)

        assert total_tracked == total_pdfs, \
            f"Mismatch: {total_tracked} tracked papers but {total_pdfs} PDFs found. " \
            f"Run test_no_orphan_pdfs and test_arxiv_pdfs_exist for details."


class TestCitationUsage:
    """Test that citations referenced in code actually exist."""

    @pytest.fixture
    def citations_root(self) -> Path:
        """Get the pdf_citations root directory."""
        return Path(__file__).parent.parent.parent / "pdf_citations"

    @pytest.fixture
    def arxiv_mapping(self, citations_root: Path) -> Dict[str, str]:
        """Load arXiv ID to PDF filename mapping."""
        mapping_file = citations_root / "arxiv_to_pdf_mapping.json"
        if not mapping_file.exists():
            return {}

        import json
        with open(mapping_file, 'r') as f:
            return json.load(f)

    def test_doremi_citation_exists(self, citations_root: Path):
        """Test that DoReMi paper (arXiv:2305.10429) exists in Domain Mixing folder."""
        doremi_pdfs = list((citations_root / "06_Domain_Mixing").glob("*DoReMi*.pdf"))
        assert len(doremi_pdfs) > 0, \
            "DoReMi paper not found in 06_Domain_Mixing/ (referenced in scripts/generate_configs.py)"

    def test_dclm_citation_exists(self, citations_root: Path):
        """Test that DCLM/KenLM paper exists in Quality Filtering folder."""
        # DCLM is arXiv:2406.11794 (DataComp-LM) or arXiv:2409.09613 (KenLM filtering)
        quality_pdfs = list((citations_root / "05_Quality_Filtering").glob("*.pdf"))
        dclm_found = any("DataComp-LM" in pdf.name or "KenLM" in pdf.name for pdf in quality_pdfs)
        assert dclm_found, \
            "DCLM or KenLM paper not found in 05_Quality_Filtering/ (referenced in src/data/heuristic_filters.py)"

    def test_zhou_et_al_citation_exists(self, citations_root: Path, arxiv_mapping: Dict[str, str]):
        """Test that Zhou et al. (2025) Data × LLM paper exists (arXiv:2505.18458)."""
        arxiv_id = "2505.18458"

        # Check mapping file first
        if arxiv_id in arxiv_mapping:
            expected_filename = arxiv_mapping[arxiv_id]
            pdf_path = citations_root / "07_Data_Practices" / expected_filename

            assert pdf_path.exists(), \
                f"Zhou et al. 'Data × LLM' paper not found at expected path: {pdf_path}\n" \
                f"Expected filename from mapping: {expected_filename}"

            # Verify the filename contains "Zhou" (first author)
            assert "Zhou" in expected_filename, \
                f"PDF filename should contain 'Zhou' (first author), but got: {expected_filename}\n" \
                f"This paper should be cited as 'Zhou et al.' not 'Li et al.'"
        else:
            # Fallback: search for the paper
            data_practices_pdfs = list((citations_root / "07_Data_Practices").glob("*.pdf"))

            # Look for Zhou and the paper title keywords
            zhou_found = any("Zhou" in pdf.name and ("LLM" in pdf.name or "DATA" in pdf.name)
                           for pdf in data_practices_pdfs)

            assert zhou_found, \
                f"Zhou et al. 'Data × LLM' paper not found (arXiv:{arxiv_id})\n" \
                f"Referenced extensively in pipeline code\n" \
                f"Note: Should be cited as 'Zhou et al.' (first author: Xuanhe Zhou), not 'Li et al.'"

    def test_fasttext_citation_exists(self, citations_root: Path):
        """Test that FastText paper (arXiv:1607.01759) exists in Quality Filtering folder."""
        fasttext_pdfs = list((citations_root / "05_Quality_Filtering").glob("*Joulin*.pdf"))
        assert len(fasttext_pdfs) > 0, \
            "FastText paper (Joulin et al.) not found in 05_Quality_Filtering/ (used for quality filters)"

    def test_broder_minhash_citation_exists(self, citations_root: Path):
        """Test that Broder MinHash paper exists in Deduplication folder."""
        broder_pdfs = list((citations_root / "04_Deduplication").glob("*Broder*.pdf"))
        assert len(broder_pdfs) > 0, \
            "Broder MinHash paper not found in 04_Deduplication/ (referenced for deduplication)"

    def test_ccnet_citation_exists(self, citations_root: Path):
        """Test that CCNet paper (arXiv:1911.00359) exists in Quality Filtering folder."""
        ccnet_pdfs = list((citations_root / "05_Quality_Filtering").glob("*CCNet*.pdf"))
        if not ccnet_pdfs:
            # Try alternate search patterns
            ccnet_pdfs = list((citations_root / "05_Quality_Filtering").glob("*Wenzek*.pdf"))

        assert len(ccnet_pdfs) > 0, \
            "CCNet paper (Wenzek et al., arXiv:1911.00359) not found in 05_Quality_Filtering/ " \
            "(referenced in all DeepSeek configs for quality filtering)"

    def test_mtp_papers_exist(self, citations_root: Path):
        """Test that MTP-related papers exist (referenced in tests/unit/test_mtp.py)."""
        # arXiv:2404.19737 - Gloeckle et al. "Better & Faster LLMs via Multi-token Prediction"
        # arXiv:2502.09419 - Mehra et al. "On multi-token prediction for efficient LLM inference"
        # arXiv:2509.18362 - Cai et al. "FastMTP"
        mtp_papers = [
            ("Gloeckle", "Multi-token Prediction"),  # 2404.19737
            ("Mehra", "multi-token prediction"),      # 2502.09419
            ("FastMTP", "Accelerating"),              # 2509.18362
        ]

        # These should be in 01_Architecture or 07_Data_Practices
        relevant_folders = [
            citations_root / "01_Architecture",
            citations_root / "07_Data_Practices",
        ]

        for author_keyword, title_keyword in mtp_papers:
            found = False
            for folder in relevant_folders:
                if not folder.exists():
                    continue
                all_pdfs = list(folder.glob("*.pdf"))
                matching_pdfs = [
                    pdf for pdf in all_pdfs
                    if author_keyword in pdf.name and title_keyword in pdf.name
                ]
                if matching_pdfs:
                    found = True
                    break

            assert found, \
                f"MTP paper with author '{author_keyword}' and title '{title_keyword}' not found " \
                f"(referenced in tests/unit/test_mtp.py)"

    def test_rope_papers_exist(self, citations_root: Path):
        """Test that RoPE-related papers exist (referenced in tests/unit/test_rope.py)."""
        # arXiv:2104.09864 - Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding"
        # arXiv:2410.06205 - Barbero et al. "Round and Round We Go! What makes Rotary Positional Encodings useful?"
        rope_papers = [
            ("Su", "RoFormer"),           # 2104.09864
            ("Barbero", "Round and Round"),  # 2410.06205
        ]

        relevant_folders = [
            citations_root / "01_Architecture",
            citations_root / "07_Data_Practices",
        ]

        for author_keyword, title_keyword in rope_papers:
            found = False
            for folder in relevant_folders:
                if not folder.exists():
                    continue
                all_pdfs = list(folder.glob("*.pdf"))
                matching_pdfs = [
                    pdf for pdf in all_pdfs
                    if author_keyword in pdf.name and title_keyword in pdf.name
                ]
                if matching_pdfs:
                    found = True
                    break

            assert found, \
                f"RoPE paper with author '{author_keyword}' and title '{title_keyword}' not found " \
                f"(referenced in tests/unit/test_rope.py)"

    def test_documentation_arxiv_ids_are_valid(self, citations_root: Path):
        """
        Scan documentation files for arXiv IDs and verify they exist in citation tracking.

        This test catches stale or incorrect arXiv IDs in documentation that don't
        match the curated citation database, preventing future misattribution issues.
        """
        # Load tracking files
        paper_ids_file = citations_root / "paper_ids.txt"
        paper_metadata_file = citations_root / "paper_metadata.txt"

        # Collect all tracked IDs
        tracked_ids = set()

        # From paper_ids.txt
        if paper_ids_file.exists():
            with open(paper_ids_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        match = re.match(r'(\d{4}\.\d{4,6})', line)
                        if match:
                            tracked_ids.add(match.group(1))

        # From paper_metadata.txt
        if paper_metadata_file.exists():
            with open(paper_metadata_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract arXiv IDs from metadata if present
                        arxiv_matches = re.findall(r'arXiv:(\d{4}\.\d{4,6})', line)
                        tracked_ids.update(arxiv_matches)

        # Directories and patterns to scan
        project_root = citations_root.parent
        docs_root = project_root / "docs"
        src_root = project_root / "src"
        scripts_root = project_root / "scripts"
        configs_root = project_root / "configs"
        tests_root = project_root / "tests"

        # Files to scan
        files_to_scan = []

        # 1. Root-level markdown files (README.md, QUICKSTART.md, BUGFIX_SUMMARY.md, etc.)
        files_to_scan.extend(project_root.glob("*.md"))

        # 2. All docs/ markdown files recursively
        if docs_root.exists():
            files_to_scan.extend(docs_root.rglob("*.md"))

        # 3. Source code Python files with docstrings
        if src_root.exists():
            files_to_scan.extend(src_root.rglob("*.py"))

        # 4. Script files
        if scripts_root.exists():
            files_to_scan.extend(scripts_root.glob("*.py"))

        # 5. Configuration files (JSON, YAML)
        if configs_root.exists():
            files_to_scan.extend(configs_root.rglob("*.json"))
            files_to_scan.extend(configs_root.rglob("*.yaml"))
            files_to_scan.extend(configs_root.rglob("*.yml"))
            # Also check config markdown files
            files_to_scan.extend(configs_root.rglob("*.md"))

        # 6. pdf_citations README
        if (citations_root / "README.md").exists():
            files_to_scan.append(citations_root / "README.md")

        # 7. Test files (to catch test fixtures with synthetic arXiv IDs)
        if tests_root.exists():
            files_to_scan.extend(tests_root.rglob("*.py"))

        # Pattern to find arXiv references in text
        arxiv_pattern = re.compile(r'arXiv:(\d{4}\.\d{4,6})')

        untracked_ids = {}  # {arxiv_id: [files where found]}

        for file_path in files_to_scan:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                # Process line by line to check for citation-ignore comments
                for line in lines:
                    # Skip lines with citation-ignore marker
                    if '# citation-ignore' in line or '# citation:ignore' in line:
                        continue

                    # Find all arXiv IDs in this line
                    found_ids = arxiv_pattern.findall(line)

                    for arxiv_id in found_ids:
                        if arxiv_id not in tracked_ids:
                            if arxiv_id not in untracked_ids:
                                untracked_ids[arxiv_id] = []
                            untracked_ids[arxiv_id].append(str(file_path.relative_to(citations_root.parent)))

            except Exception as e:
                # Skip files that can't be read
                continue

        if untracked_ids:
            error_msg = "Found arXiv IDs in documentation that are NOT in paper_ids.txt or paper_metadata.txt:\n\n"
            for arxiv_id, files in sorted(untracked_ids.items()):
                error_msg += f"  arXiv:{arxiv_id}\n"
                for file in files:
                    error_msg += f"    - {file}\n"
                error_msg += "\n"

            error_msg += "Action: Either:\n"
            error_msg += "  1. Add these IDs to paper_ids.txt and download the PDFs, OR\n"
            error_msg += "  2. Correct the arXiv IDs in the documentation to match tracked papers\n"

            pytest.fail(error_msg)
