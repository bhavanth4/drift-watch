@echo off
echo Initializing DVC...
dvc init

echo Configuring local remote...
mkdir \tmp\dvcstore
dvc remote add -d localremote \tmp\dvcstore

echo Tracking data directory...
dvc add data/
git add data.dvc .gitignore
git commit -m "Track data with DVC"

echo DVC setup complete.
