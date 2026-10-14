from pathlib import Path

project_root = Path(__file__).parent.parent

data_path = Path.joinpath(project_root, 'Data')
data_temp_path = Path.joinpath(data_path, 'temp')
results_path = Path.joinpath(project_root, 'Results')
figures_path = Path.joinpath(project_root, 'figures')
src_path = Path.joinpath(project_root, 'src')

lib_path = Path.joinpath(src_path, 'lib')
notebooks_path = Path.joinpath(src_path, 'notebooks')
tests_path = Path.joinpath(src_path, 'tests')

data_path.mkdir(parents=True, exist_ok=True)
data_temp_path.mkdir(parents=True, exist_ok=True)
results_path.mkdir(parents=True, exist_ok=True)
figures_path.mkdir(parents=True, exist_ok=True)
src_path.mkdir(parents=True, exist_ok=True)

lib_path.mkdir(parents=True, exist_ok=True)
notebooks_path.mkdir(parents=True, exist_ok=True)
tests_path.mkdir(parents=True, exist_ok=True)