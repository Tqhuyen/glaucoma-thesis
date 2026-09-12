@echo off
cd /d "C:\Users\huyen\Downloads\master_thesis\glaucoma-thesis"
python scripts\run_denoise_full_cuda.py --execute --methods bilateral bm3d gaussian median tv >> "outputs\denoise_full_cuda\run.log" 2>> "outputs\denoise_full_cuda\run.err.log"
