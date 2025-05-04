cd VAE-PU-FOR-CTL/
conda activate vae-pu-env
tar -czf results.tar.gz --exclude=**/*.pt result/
cp results.tar.gz results-2025-05-04.tar.gz

scp awawrzenczyk@eden.mini.pw.edu.pl:~/VAE-PU-FOR-CTL/results-2025-05-04.tar.gz \
results-2025-05-04.tar.gz

scp wawrzenczyka@ssh.mini.pw.edu.pl:~/results-2025-05-04.tar.gz results-2025-05-04.tar.gz
