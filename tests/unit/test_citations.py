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
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Format: <identifier> | <type> | <title> | <authors> | <venue> | <year> | <folder> | <filename>
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) == 8:
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

    def test_paper_ids_file_exists(self, citations_root: Path):
        """Test that paper_ids.txt exists."""
        assert (citations_root / "paper_ids.txt").exists(), \
            "paper_ids.txt is missing from pdf_citations/"

    def test_arxiv_id_format(self, paper_ids: List[str]):
        """Test that all arXiv IDs follow the correct format."""
        for arxiv_id in paper_ids:
            assert re.match(r'^\d{4}\.\d{4,6}$', arxiv_id), \
                f"Invalid arXiv ID format: {arxiv_id}"

    def test_arxiv_pdfs_exist(self, citations_root: Path, paper_ids: List[str], all_pdfs: Set[Path]):
        """Test that all arXiv papers have corresponding PDFs."""
        missing_papers = []

        for arxiv_id in paper_ids:
            # Search for PDF matching this arXiv ID in filename
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
        all_pdfs: Set[Path]
    ):
        """Test that all PDFs have corresponding entries in paper_ids.txt or paper_metadata.txt."""
        # Build set of tracked filenames
        tracked_filenames = set()

        # From paper_metadata.txt (exact filenames)
        for paper in paper_metadata:
            tracked_filenames.add(paper['filename'])

        # From arXiv IDs (partial matching)
        arxiv_ids_set = set(paper_ids)

        orphan_pdfs = []

        for pdf in all_pdfs:
            filename = pdf.name

            # Check if filename is in metadata
            if filename in tracked_filenames:
                continue

            # Check if any arXiv ID is in the filename
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

    def test_li_et_al_citation_exists(self, citations_root: Path):
        """Test that Li et al. (2025) Data × LLM paper exists."""
        # Should be in 07_Data_Practices/ or 05_Quality_Filtering/
        data_practices_pdfs = list((citations_root / "07_Data_Practices").glob("*.pdf"))
        quality_pdfs = list((citations_root / "05_Quality_Filtering").glob("*.pdf"))

        all_pdfs = data_practices_pdfs + quality_pdfs
        li_found = any("LLM" in pdf.name and "DATA" in pdf.name for pdf in all_pdfs)

        assert li_found, \
            "Li et al. 'Data × LLM' paper not found (referenced extensively in pipeline code)"

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
