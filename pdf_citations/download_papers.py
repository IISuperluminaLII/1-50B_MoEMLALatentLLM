import os, re, sys, arxiv

# Fix encoding issues on Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')

def fix_title(title: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", re.sub(r"\s*\n+\s*", " ", title))

def paper_to_filename(paper: arxiv.Result) -> str:
    author_str = str(paper.authors[0]) + " et al." * (len(paper.authors) > 1)
    return f"{author_str} - {fix_title(paper.title)}.pdf"

def parse_line(line: str):
    m = re.match(r".*(?P<paper_id>\d{4}\.\d{4,6}(v\d+)?)(\.pdf)?$", line)
    return m.group("paper_id") if m is not None else None

# Read from file if provided as argument, otherwise from stdin
if len(sys.argv) > 1:
    with open(sys.argv[1], 'r') as f:
        lines = f.readlines()
else:
    lines = sys.stdin.readlines()

paper_ids = [parse_line(line.strip()) for line in lines]
paper_ids = [x for x in paper_ids if x is not None]

print(f"Found {len(paper_ids)} paper IDs: {paper_ids}")

# Use the new Client API
client = arxiv.Client()
search = arxiv.Search(id_list=paper_ids)
papers = list(client.results(search))

print(f"Retrieved {len(papers)} papers from arXiv\n")

for paper, paper_id in zip(papers, paper_ids):
    src_filename = f"{paper_id}.pdf"
    dst_filename = paper_to_filename(paper)
    if os.path.exists(dst_filename):
        print(f"[TargetExists] {dst_filename}")
    elif os.path.exists(src_filename):
        print(f"[Rename] {src_filename}")
        os.rename(src_filename, dst_filename)
    else:
        print(f"[Download] Downloading {paper_id}...")
        paper.download_pdf(filename=dst_filename)
    print(f"file:    {dst_filename}")
    print(f"url:     {paper.entry_id}")
    print(f"authors: {[str(x) for x in paper.authors]}")
    print(f"title:   {paper.title}\n")
