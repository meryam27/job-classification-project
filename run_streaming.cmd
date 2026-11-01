@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%.venv-pyspark35"
set "SPARK_HOME=%VENV_DIR%\Lib\site-packages\pyspark"
set "PATH=%VENV_DIR%\Scripts;%PATH%"
set "PYSPARK_PYTHON=python.exe"
set "SPARK_VERSION=3.5.8"

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo Python environment not found: "%VENV_DIR%\Scripts\python.exe"
    exit /b 1
)

if not exist "%SPARK_HOME%\jars" (
    echo PySpark jars not found: "%SPARK_HOME%\jars"
    exit /b 1
)

cd /d "%PROJECT_DIR%"
call "%SPARK_HOME%\bin\spark-submit.cmd" --packages "org.apache.spark:spark-sql-kafka-0-10_2.12:%SPARK_VERSION%" "%PROJECT_DIR%streaming\streaming.py" %*
exit /b %ERRORLEVEL%
