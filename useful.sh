cd VAE-PU-label-shift/
conda activate vae-pu-env
rm -rf result-clean/
python ./create_results_copy.py
tar -czf results.tar.gz result-clean/
cp results.tar.gz results-2024-08-09.tar.gz

scp awawrzenczyk@eden.mini.pw.edu.pl:~/VAE-PU-label-shift/results.tar.gz \
results-2024-08-09.tar.gz

scp wawrzenczyka@ssh.mini.pw.edu.pl:~/results-2024-08-09.tar.gz results-2024-08-09.tar.gz
