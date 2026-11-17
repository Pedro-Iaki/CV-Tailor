import pytest

from cvtailor.master import parse_master
from cvtailor.tags import extract_tags, match_tags


@pytest.fixture
def tag_master(master_dict):
    d = master_dict
    d["jobs"][0]["bullets"][0]["tags"] += ["go", "csharp", "node-js", "r"]
    d["tag_aliases"]["go"] = ["golang"]
    d["tag_aliases"]["csharp"] = ["c#"]
    d["tag_aliases"]["node-js"] = ["node.js"]
    return parse_master(d)


def test_hyphen_space_and_joined_variants(tag_master):
    for text in ["Azure-Data-Factory", "azure data factory", "AzureDataFactory", "azure  data\nfactory"]:
        assert "azure-data-factory" in extract_tags(text, tag_master), text


def test_aliases(tag_master):
    tags = extract_tags("Strong PySpark and ADF experience; Postgres a plus", tag_master)
    assert {"spark", "azure-data-factory", "postgresql"} <= set(tags)


def test_word_boundaries(tag_master):
    assert "spark" not in extract_tags("We value a sparkle of creativity", tag_master)
    assert "spark" in extract_tags("Experience with Spark.", tag_master)
    assert "spark" in extract_tags("(spark)", tag_master)


def test_short_tags_only_via_alias(tag_master):
    assert "go" not in extract_tags("Ready to go the extra mile", tag_master)
    assert "go" in extract_tags("Backend in Golang", tag_master)
    assert "r" not in extract_tags("R and Python", tag_master)  # no alias defined


def test_symbols(tag_master):
    assert "csharp" in extract_tags("Backend in C# and .NET", tag_master)
    assert "ci-cd" in extract_tags("Own the CI/CD pipelines", tag_master)
    assert "node-js" in extract_tags("APIs in Node.js, React", tag_master)
    # 'cd' inside ci/cd must not create spurious matches; 'node' alone is not node.js
    assert "node-js" not in extract_tags("each node in the cluster", tag_master)


def test_output_sorted_and_matches_report_phrase(tag_master):
    matches = match_tags("PySpark on Databricks with Azure", tag_master)
    assert list(matches) == sorted(matches)
    assert matches["spark"] == "PySpark"
