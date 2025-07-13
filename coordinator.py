import paramiko
import os
# only run this script on master node
# workers = ["172.31.46.106", "172.31.46.92"]
workers = ["172.31.46.106"]

def send_signal(ip):
    print(f"Sending SIGUSR1 to {ip}...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip, username="ubuntu", key_filename="./renming.pem")
    ssh.exec_command("kill -SIGUSR1 $(pgrep nccl_dynamic)")
    ssh.close()

for ip in workers:
    send_signal(ip)
