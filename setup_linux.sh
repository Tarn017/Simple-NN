ls
echo "[INFO] entered .venv/"
pip install --upgrade pip
echo "[INFO] upgraded pip"
pip install -r requirements_class.txt 
echo "[INFO] non-torch libraries installed"
mkdir -p tmp_pip
echo "[INFO] tmp_pip cache-folder created"
export TMPDIR=$(pwd)/tmp_pip
pip install --cache-dir="$TMPDIR" -r requirements_torch.txt 
echo "[INFO] torch libraries installed"
rm -r tmp_pip
echo "[INFO] deleted tmp cache"
echo "------------------------------------------------------------"
echo "fertig! :)"
