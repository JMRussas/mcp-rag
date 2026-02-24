#  mcp-rag - Test Fixtures
#
#  Shared pytest fixtures for chunker and pipeline tests.
#  Uses tmp_path (pytest built-in) for all temporary files.
#
#  Depends on: (none)
#  Used by:    all test files

import pytest


@pytest.fixture
def tmp_python_file(tmp_path):
    """A Python file with a class and two functions."""
    code = '''\
import os
from pathlib import Path

CONSTANT = 42


class MyService:
    """A sample service class."""

    def __init__(self, name: str):
        self.name = name

    def run(self):
        return f"running {self.name}"


def helper_one():
    """First helper."""
    return 1


def helper_two():
    """Second helper."""
    return 2
'''
    f = tmp_path / "sample.py"
    f.write_text(code, encoding="utf-8")
    return f


@pytest.fixture
def tmp_csharp_file(tmp_path):
    """A C# file with a namespace and a class."""
    code = """\
//  Sample - Test File
//
//  A test C# file for unit tests.
//
//  Depends on: nothing
//  Used by:    tests

using System;

namespace TestProject;

/// <summary>
/// A sample class for testing.
/// </summary>
public class SampleClass
{
    public string Name { get; set; }

    public void DoWork()
    {
        Console.WriteLine("working");
    }
}
"""
    f = tmp_path / "SampleClass.cs"
    f.write_text(code, encoding="utf-8")
    return f


@pytest.fixture
def tmp_markdown_file(tmp_path):
    """A markdown file with 3 headings and enough content."""
    md = """\
# Getting Started

This section explains how to get started with the project.
Follow the steps below to set up your environment.

## Installation

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

Make sure you have Python 3.11 or later installed.

## Configuration

Edit the config.json file to point at your codebase.
Set the repo paths, source tags, and chunker types.
"""
    f = tmp_path / "guide.md"
    f.write_text(md, encoding="utf-8")
    return f


@pytest.fixture
def tmp_config():
    """A minimal valid config dict for testing."""
    return {
        "ollama": {
            "host": "http://localhost:11434",
            "embed_model": "nomic-embed-text",
            "embed_timeout": 30.0,
        },
        "database": {"path": "data/test.db"},
        "search": {
            "default_top_k": 8,
            "max_top_k": 20,
            "embed_dimensions": 768,
        },
        "sources": {
            "repos_dir": "data/repos",
            "chunks_path": "data/chunks.jsonl",
        },
        "repos": [
            {
                "name": "test-src",
                "path": "/tmp/test-src",
                "type": "python",
                "source_tag": "test",
            }
        ],
    }
