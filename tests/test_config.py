from smores import config


def test_project_root_is_the_repo():
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()


def test_data_directories_exist():
    for path in (
        config.RAW_DIR,
        config.INTERIM_DIR,
        config.PROCESSED_DIR,
        config.EXTERNAL_DIR,
        config.MODELS_DIR,
        config.FIGURES_DIR,
    ):
        assert path.is_dir(), f"missing directory: {path}"
