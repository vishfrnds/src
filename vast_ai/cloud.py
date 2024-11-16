import os
import re
import subprocess
import time

from vastai import VastAI


class VastAII:

  def __init__(self):
    self.vast_sdk = VastAI(api_key=os.getenv('VAST_API_KEY'))
    self.instance_id = None

  def parse_result(self, output):
    lines = output.strip().split('\n')

    # Extract header and rows
    header = re.split(r'\s{2,}', lines[0])
    rows = [re.split(r'\s{2,}', line) for line in lines[1:]]

    # Convert rows into list of dictionaries
    return [dict(zip(header, row)) for row in rows]

  def create_instance(self, dph=0.111, num_gpus=1, min_gpu_ram=11, max_gpu_ram=15, disk_space=5):
    query = ' '.join([
      f'dph<{dph}',
      'rentable=True', 
      f'num_gpus={num_gpus}',
      f'gpu_ram>{min_gpu_ram}',
      f'gpu_ram<{max_gpu_ram}',
      f'disk_space>{disk_space}',
      # 'dlperf>30',
      # https://en.wikipedia.org/wiki/GeForce_40_series https://en.wikipedia.org/wiki/GeForce_30_series
      # 'gpu_name in [RTX_4090_D,RTX_4090,RTX_3090]',
      # 'cuda_vers>=12.1'
    ])
    print(query)
    output = self.vast_sdk.search_offers(type='bid', query=query, storage=disk_space)
    # decode
    print(output)
    data_dicts = self.parse_result(output)

    self.instance_id = int(data_dicts[0]['ID'])
    onstart = 'apt install -y git pip;git clone https://github.com/tinygrad/tinygrad.git;cd tinygrad;python3 -m pip install -e .;python3 tinygrad/runtime/ops_cloud.py'
    self.vast_sdk.create_instance(ID=self.instance_id, disk=disk_space, image='nvidia/cuda:12.1.1-devel-ubuntu22.04', onstart_cmd=onstart)
    self.connect()

  def connect(self):
    while True:
      instances = self.vast_sdk.show_instances()
      data_dicts = self.parse_result(instances)
      print(data_dicts)
      if len(data_dicts) == 0: 
        time.sleep(10)
        continue
      status = data_dicts[0]['Status']
      if status == 'running': break
      print('Waiting for instance to start...', status)
      time.sleep(10)
    # if len(data_dicts) == 0:
    #   self.create_instance()
    #   return self.get_instance_ids()
    self.instance_id = int(data_dicts[0]['ID'])
    self.port = int(data_dicts[0]['SSH Port'])
    self.addr = data_dicts[0]['SSH Addr']
    # Setup SSH port forwarding
    subprocess.run(
      f"ssh -o StrictHostKeyChecking=no -f -N -L 6667:localhost:6667 root@{self.addr} -p {self.port}",
      shell=True,
      start_new_session=True
    )
    print(f'ssh -o StrictHostKeyChecking=no -N -L 6667:localhost:6667 root@{self.addr} -p {self.port}')
    return [int(data_dict['ID']) for data_dict in data_dicts]

  def export_keys(self, instance_id: int):
    self.vast_sdk.execute(ID=instance_id, COMMAND=f'export HF_API_KEY={os.getenv("HF_API_KEY")}')

  def sync_and_monitor(self):

    last_modified = {}
    src_dir = "../src"

    # Read rsync_ignore.txt
    with open("rsync_ignore.txt", "r") as f:
      ignore_patterns = [line.strip() for line in f if line.strip()]

    def should_ignore(file_path):
      rel_path = os.path.relpath(file_path, src_dir)
      return any(rel_path.startswith(pattern) or rel_path.endswith(pattern) for pattern in ignore_patterns)

    def perform_sync():
      self.get_instance_ids()  # This will set self.instance_id if an instance exists or create a new one
      if self.instance_id is not None:
        self.export_keys(self.instance_id)
        subprocess.run(
          args=f'rsync -avz --exclude-from "rsync_ignore.txt" -e "ssh -p {self.port}" {src_dir} root@{self.addr}:/root/srcs',
            shell=True)
      else:
        print("No instance ID available. Please create an instance first.")

    while True:
      changes_detected = False
      for root, _, files in os.walk(src_dir):
        for file in files:
          file_path = os.path.join(root, file)
          if should_ignore(file_path):
            continue

          current_mtime = os.path.getmtime(file_path)

          if file_path not in last_modified or current_mtime > last_modified[file_path]:
            print(f"File changed: {file_path}")
            changes_detected = True
            last_modified[file_path] = current_mtime

      if changes_detected:
        perform_sync()

      time.sleep(3)  # Wait for 3 seconds before checking again

  def destroy(self):
    instance_ids = self.get_instance_ids()
    if len(instance_ids) > 0:
      self.vast_sdk.destroy_instances(ids=instance_ids)


client = VastAII()
client.create_instance()
# client.sync_and_monitor()
