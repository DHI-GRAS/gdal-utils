
REM Install to local python distribution (e.g. Anaconda)
python.exe setup.py install

REM Install to QGIS python distribution
python.exe setup.py install --prefix="C:\OSGeo4W64\apps\Python27" --install-data="~\.qgis\processing\scripts"

pause