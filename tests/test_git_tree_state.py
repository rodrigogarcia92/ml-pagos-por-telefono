import subprocess

from src.model_training import tracking


def _git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True,
                   capture_output=True)


def _repo(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / ".gitignore").write_text("gha-creds-*.json\n")
    (tmp_path / "a.txt").write_text("a")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_clean_tree_is_not_dirty(tmp_path):
    st = tracking.git_tree_state(_repo(tmp_path))
    assert st["tree_dirty"] is False and st["tree_changes"] == []
    assert st["git_sha"] and not st["git_sha"].endswith("-dirty")


def test_dirty_tree_keeps_the_bare_sha_and_lists_the_paths(tmp_path):
    repo = _repo(tmp_path)
    clean_sha = tracking.git_tree_state(repo)["git_sha"]
    (repo / "data" / "raw").mkdir(parents=True)
    (repo / "data" / "raw" / "new.json").write_text("{}")
    (repo / "a.txt").write_text("changed")
    st = tracking.git_tree_state(repo)
    assert st["git_sha"] == clean_sha
    assert st["tree_dirty"] is True
    assert st["tree_changes"] == ["a.txt", "data/raw/new.json"]


def test_credentials_file_written_by_the_auth_action_does_not_dirty_the_tree(tmp_path):
    repo = _repo(tmp_path)
    (repo / "gha-creds-2bf28456de611843.json").write_text("{}")
    assert tracking.git_tree_state(repo)["tree_dirty"] is False


def test_repo_gitignore_covers_the_credentials_file():
    out = subprocess.run(["git", "check-ignore", "gha-creds-abc123.json"], capture_output=True, text=True)
    assert out.returncode == 0


def test_not_a_repo_is_unknown(tmp_path):
    st = tracking.git_tree_state(tmp_path)
    assert st == {"git_sha": "unknown", "tree_dirty": None, "tree_changes": []}
