cd VAE-PU-label-shift/
conda activate vae-pu-env
tar -czf results.tar.gz --exclude=**/*.pt result/
cp results.tar.gz results-2024-09-17.tar.gz

scp awawrzenczyk@eden.mini.pw.edu.pl:~/VAE-PU-label-shift/results.tar.gz \
results-2024-09-17.tar.gz

scp wawrzenczyka@ssh.mini.pw.edu.pl:~/results-2024-09-17.tar.gz results-2024-09-17.tar.gz
