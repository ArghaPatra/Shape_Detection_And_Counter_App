@echo off
REM Build ShapeCounter.exe. Run from D:\project_data with the "shapes" conda environment active.

REM PyInstaller imports many packages in helper processes; this stops the OpenMP clash from killing them.
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=1

echo Installing PyInstaller...
pip install pyinstaller
if errorlevel 1 goto fail

if not exist src\__init__.py type nul > src\__init__.py

if not exist models\shape_rf.joblib (
    echo Training the model first...
    python train_model.py
    if errorlevel 1 goto fail
)

REM The app does not use scikit-image, matplotlib, torch etc., so they are left out.
REM The array_api_compat torch/cupy/dask modules are optional backends that import torch.
pyinstaller --noconfirm --clean --windowed --name ShapeCounter ^
  --add-data "models\shape_rf.joblib;models" ^
  --hidden-import sklearn.ensemble._forest ^
  --hidden-import sklearn.tree._tree ^
  --exclude-module skimage --exclude-module matplotlib --exclude-module torch ^
  --exclude-module IPython --exclude-module sympy --exclude-module networkx ^
  --exclude-module PIL --exclude-module imageio --exclude-module tifffile ^
  --exclude-module scipy._lib.array_api_compat.torch ^
  --exclude-module scipy._lib.array_api_compat.cupy ^
  --exclude-module scipy._lib.array_api_compat.dask ^
  --exclude-module sklearn.externals.array_api_compat.torch ^
  --exclude-module sklearn.externals.array_api_compat.cupy ^
  --exclude-module sklearn.externals.array_api_compat.dask ^
  app.py
if errorlevel 1 goto fail

echo.
echo Done. Your app is: dist\ShapeCounter\ShapeCounter.exe
echo To share it, zip the whole dist\ShapeCounter folder.
pause
exit /b 0

:fail
echo.
echo Build failed. Scroll up to see the error.
pause
exit /b 1