"""Agent tools for experiments and research."""

from pathlib import Path
from typing import Optional
import re
import subprocess
import tempfile
import os
import base64
import toml
from duckduckgo_search import DDGS

from hackathon_science.git_ops import load_papers


def _get_working_dir() -> Optional[Path]:
    """Get working directory from team config if available."""
    # Try to find team config by looking for agents/*/config.toml in current working tree
    cwd = Path.cwd()

    # Check if we're in an agents directory
    if "agents" in cwd.parts:
        agent_idx = cwd.parts.index("agents")
        if len(cwd.parts) > agent_idx + 1:
            team_name = cwd.parts[agent_idx + 1]
            # Reconstruct path to repo root
            repo_root = Path(*cwd.parts[:agent_idx])
            config_path = repo_root / "agents" / team_name / "config.toml"

            if config_path.exists():
                try:
                    config = toml.load(config_path)
                    working_dir = config.get("team", {}).get("working_dir")
                    if working_dir:
                        # Make path absolute relative to repo root
                        return repo_root / working_dir
                except Exception:
                    pass

    return None


def extract_code_from_llm_response(text: str) -> str:
    """Extract Python code from LLM response that may contain markdown code blocks.

    Args:
        text: Raw LLM response text that may contain:
            - Markdown code blocks (```python ... ```)
            - Plain Python code
            - Mixed code and explanatory text

    Returns:
        Extracted Python code, or original text if no code blocks found

    Examples:
        >>> extract_code_from_llm_response("```python\\nprint('hello')\\n```")
        "print('hello')"
        >>> extract_code_from_llm_response("print('hello')")
        "print('hello')"
    """
    # Pattern to match markdown code blocks with optional language specifier
    # Matches: ```python\ncode\n``` or ```\ncode\n```
    code_block_pattern = r'```(?:python)?\s*\n(.*?)\n```'

    # Find all code blocks
    matches = re.findall(code_block_pattern, text, re.DOTALL)

    if matches:
        # If multiple code blocks found, join them with newlines
        # This handles cases where LLM splits code into multiple blocks
        return '\n\n'.join(matches)

    # No code blocks found - check if this looks like raw Python code
    # Simple heuristic: if it contains common Python keywords and no markdown formatting
    python_indicators = ['import ', 'def ', 'class ', 'if __name__', 'print(', 'return ']
    markdown_indicators = ['```', '## ', '*Note:', 'This code', 'Example usage']

    has_python = any(indicator in text for indicator in python_indicators)
    has_markdown = any(indicator in text for indicator in markdown_indicators)

    # If it looks like Python and doesn't have markdown, return as-is
    if has_python and not has_markdown:
        return text

    # If it has both, try to extract just the Python-looking parts
    # Split by common separators and filter for code-like content
    if has_python and has_markdown:
        lines = text.split('\n')
        code_lines = []
        in_code_section = False

        for line in lines:
            # Skip obvious prose/markdown lines
            if any(line.strip().startswith(marker) for marker in ['## ', '*', '-', '>', 'Note:', 'This ', 'The ']):
                in_code_section = False
                continue

            # Detect start of code section
            if any(indicator in line for indicator in python_indicators):
                in_code_section = True

            # Add lines that look like code
            if in_code_section or line.strip().startswith((' ', '\t')) or any(indicator in line for indicator in python_indicators):
                code_lines.append(line)

        if code_lines:
            return '\n'.join(code_lines).strip()

    # Fallback: return original text
    return text


def run_code(code: str, filename: str = "script.py", timeout: int = 300, working_dir: Optional[str] = None) -> str:
    """Execute Python code in a subprocess and return combined output.

    Args:
        code: Python code to execute. Can be:
            - Raw Python code
            - LLM response with markdown code blocks (```python ... ```)
            - Mixed text and code (will attempt to extract code)
        filename: Name for the script (default: "script.py")
        timeout: Maximum execution time in seconds (default: 300)
        working_dir: Directory to save code files (default: from team config)
            Set to None to disable saving and use temp files only

    Returns:
        Combined stdout/stderr output from code execution

    Note:
        The function will automatically extract code from markdown blocks
        if present, allowing direct use of LLM responses.
        By default, code is saved to the team's working directory from config.
    """
    # Extract code from potential LLM response with markdown
    extracted_code = extract_code_from_llm_response(code)

    # Determine working directory
    if working_dir is None:
        # Try to get from config
        working_dir = _get_working_dir()

    # Determine where to save/execute the file
    if working_dir:
        # Save to specified working directory
        save_path = Path(working_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        code_file = save_path / filename
        code_file.write_text(extracted_code)
        exec_file = code_file
        should_cleanup = False
    else:
        # Create temporary file for the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            exec_file = Path(f.name)
            f.write(extracted_code)
        should_cleanup = True

    try:
        # Execute the code using subprocess
        result = subprocess.run(
            ['python3', str(exec_file)],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        # Combine stdout and stderr
        output = result.stdout
        if result.stderr:
            output += "\n" + result.stderr

        # Add return code if non-zero
        if result.returncode != 0:
            output += f"\n\nProcess exited with code {result.returncode}"

        return output

    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing code: {str(e)}"
    finally:
        # Clean up temporary file only if we created one
        if should_cleanup:
            try:
                exec_file.unlink()
            except Exception:
                pass


def search_web(query: str, max_results: int = 10) -> list[dict]:
    """Search web via DuckDuckGo.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 10)

    Returns:
        List of dicts with keys: title, url, snippet

    Raises:
        Exception: If search fails
    """
    ddgs = DDGS()
    results = ddgs.text(query, max_results=max_results)

    # Transform results to consistent format
    formatted_results = []
    for result in results:
        formatted_results.append({
            'title': result.get('title', ''),
            'url': result.get('href', ''),
            'snippet': result.get('body', '')
        })

    return formatted_results


def get_paper(paper_id: str, papers_dir: Optional[Path] = None) -> Optional[dict]:
    """Load full paper by ID.

    Args:
        paper_id: Paper ID to search for
        papers_dir: Optional path to papers directory

    Returns:
        Dict with all paper fields or None if not found
    """
    if papers_dir is None:
        return None

    # papers_dir should point to the papers/ directory
    # If it's actually repo_path, adjust it
    if papers_dir.name != "papers":
        papers_dir = papers_dir / "papers"

    papers = load_papers(papers_dir.parent)

    # Find paper by ID
    for paper in papers:
        if paper.id == paper_id:
            # Return as dict with all fields
            return {
                'id': paper.id,
                'title': paper.title,
                'author': paper.author,
                'date': paper.date,
                'tags': paper.tags,
                'introduction': paper.introduction,
                'methods': paper.methods,
                'results': paper.results,
                'references': paper.references,
                'appendix': paper.appendix,
            }

    return None


def image_to_base64(image_path: str, alt_text: str = "Image") -> str:
    """Convert an image file to base64-encoded markdown for embedding in papers.

    Args:
        image_path: Path to the image file (JPEG or PNG). Can be relative or absolute.
        alt_text: Alternative text for the image (default: "Image")

    Returns:
        Markdown string with base64-encoded image that can be embedded in paper sections

    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If file is not a supported image format (jpg, jpeg, png)

    Example:
        # In your agent's run() function:
        import matplotlib.pyplot as plt

        # Create a plot
        plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])
        plt.savefig('results.png')
        plt.close()

        # Convert to base64 for embedding
        image_md = image_to_base64('results.png', 'Experimental Results')

        # Include in paper
        return Paper(
            title="...",
            results=f"Our experiments show:\\n\\n{image_md}",
            ...
        )
    """
    # Convert to Path object and resolve
    img_path = Path(image_path)

    # If relative path, try to resolve from working directory
    if not img_path.is_absolute():
        working_dir = _get_working_dir()
        if working_dir:
            img_path = working_dir / img_path

    # Check if file exists
    if not img_path.exists():
        raise FileNotFoundError(f"Image file not found: {img_path}")

    # Validate file extension
    ext = img_path.suffix.lower()
    if ext not in ['.jpg', '.jpeg', '.png']:
        raise ValueError(f"Unsupported image format: {ext}. Only .jpg, .jpeg, and .png are supported.")

    # Determine MIME type
    mime_type = "image/jpeg" if ext in ['.jpg', '.jpeg'] else "image/png"

    # Read and encode image
    with open(img_path, 'rb') as img_file:
        img_data = img_file.read()
        b64_data = base64.b64encode(img_data).decode('utf-8')

    # Return markdown with embedded base64 image
    return f"![{alt_text}](data:{mime_type};base64,{b64_data})"
